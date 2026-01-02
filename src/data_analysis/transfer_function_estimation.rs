// src/data_analysis/transfer_function_estimation.rs

use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::data_analysis::spectral_analysis::{
    coherence, to_magnitude_db, to_phase_deg, unwrap_phase, welch_cpsd, welch_psd, WelchConfig,
};
use crate::data_input::log_data::LogRowData;

/// Calculate signal variation (standard deviation)
fn calculate_signal_variation(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let mean: f64 = signal.iter().map(|&x| x as f64).sum::<f64>() / signal.len() as f64;
    let variance: f64 = signal
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / signal.len() as f64;

    variance.sqrt()
}

/// Transfer function estimation result with frequency response and quality metrics
#[derive(Debug, Clone)]
pub struct TransferFunctionResult {
    /// Frequency vector in Hz
    pub frequency_hz: Vec<f64>,
    /// Magnitude in dB
    pub magnitude_db: Vec<f64>,
    /// Phase in degrees (unwrapped)
    pub phase_deg: Vec<f64>,
    /// Coherence (0-1 quality metric)
    pub coherence: Vec<f64>,
    /// Sample rate used
    pub sample_rate_hz: f64,
    /// Axis name (Roll, Pitch, or Yaw) - retained for potential future use
    #[allow(dead_code)]
    pub axis_name: String,
}

impl TransferFunctionResult {
    /// Check if transfer function has valid data
    pub fn is_valid(&self) -> bool {
        !self.frequency_hz.is_empty() && self.frequency_hz.len() == self.magnitude_db.len()
    }

    /// Get number of frequency points
    pub fn len(&self) -> usize {
        self.frequency_hz.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.frequency_hz.is_empty()
    }

    /// Find index closest to a given frequency
    pub fn find_frequency_index(&self, target_freq: f64) -> Option<usize> {
        if self.frequency_hz.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_diff = (self.frequency_hz[0] - target_freq).abs();

        for (i, &freq) in self.frequency_hz.iter().enumerate() {
            let diff = (freq - target_freq).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        Some(best_idx)
    }

    /// Get magnitude at a specific frequency (with interpolation)
    pub fn get_magnitude_at_freq(&self, target_freq: f64) -> Option<f64> {
        if self.frequency_hz.len() < 2 {
            return None;
        }

        // Find bracketing indices
        let mut idx = 0;
        while idx < self.frequency_hz.len() - 1 && self.frequency_hz[idx + 1] < target_freq {
            idx += 1;
        }

        if idx >= self.frequency_hz.len() - 1 {
            return self.magnitude_db.last().copied();
        }

        // Linear interpolation
        let f1 = self.frequency_hz[idx];
        let f2 = self.frequency_hz[idx + 1];
        let m1 = self.magnitude_db[idx];
        let m2 = self.magnitude_db[idx + 1];

        let t = (target_freq - f1) / (f2 - f1);
        Some(m1 + t * (m2 - m1))
    }

    /// Get phase at a specific frequency (with interpolation)
    pub fn get_phase_at_freq(&self, target_freq: f64) -> Option<f64> {
        if self.frequency_hz.len() < 2 {
            return None;
        }

        // Find bracketing indices
        let mut idx = 0;
        while idx < self.frequency_hz.len() - 1 && self.frequency_hz[idx + 1] < target_freq {
            idx += 1;
        }

        if idx >= self.frequency_hz.len() - 1 {
            return self.phase_deg.last().copied();
        }

        // Linear interpolation
        let f1 = self.frequency_hz[idx];
        let f2 = self.frequency_hz[idx + 1];
        let p1 = self.phase_deg[idx];
        let p2 = self.phase_deg[idx + 1];

        let t = (target_freq - f1) / (f2 - f1);
        Some(p1 + t * (p2 - p1))
    }

    /// Get coherence at a specific frequency
    pub fn get_coherence_at_freq(&self, target_freq: f64) -> Option<f64> {
        let idx = self.find_frequency_index(target_freq)?;
        self.coherence.get(idx).copied()
    }
}

/// Estimates transfer function using H1 estimator: H1(f) = Sxy(f) / Sxx(f)
///
/// This computes the frequency response from setpoint (input) to gyro (output)
/// with coherence quality assessment.
pub fn estimate_transfer_function_h1(
    log_data: &[LogRowData],
    sample_rate: f64,
    axis_index: usize,
) -> Result<TransferFunctionResult, Box<dyn Error>> {
    // Validate axis index
    if axis_index >= 3 {
        return Err(format!("Invalid axis index: {}", axis_index).into());
    }

    let axis_name = AXIS_NAMES[axis_index].to_string();

    // Extract synchronized data for specified axis
    let mut setpoint_data = Vec::new();
    let mut gyro_data = Vec::new();

    for row in log_data {
        if let (Some(sp), Some(gy)) = (row.setpoint[axis_index], row.gyro[axis_index]) {
            setpoint_data.push(sp as f32);
            gyro_data.push(gy as f32);
        }
    }

    // Data quality checks
    if setpoint_data.len() < 2048 {
        return Err(format!(
            "Insufficient data for transfer function estimation: {} samples (minimum 2048)",
            setpoint_data.len()
        )
        .into());
    }

    // Check for non-constant signals
    let setpoint_variation = calculate_signal_variation(&setpoint_data);
    let gyro_variation = calculate_signal_variation(&gyro_data);

    if setpoint_variation < 1e-6 {
        return Err(format!("Setpoint signal is constant for {}", axis_name).into());
    }

    if gyro_variation < 1e-6 {
        return Err(format!("Gyro signal is constant for {}", axis_name).into());
    }

    // Configure Welch's method (default: segment_length = data_length / 8 for 8 averages)
    let config = WelchConfig::default();

    // Compute spectral estimates
    let cpsd_xy = welch_cpsd(
        &setpoint_data,
        &gyro_data,
        sample_rate,
        Some(config.clone()),
    )?;
    let psd_xx = welch_psd(&setpoint_data, sample_rate, Some(config.clone()))?;
    let psd_yy = welch_psd(&gyro_data, sample_rate, Some(config))?;

    // Compute H1 estimator: H1(f) = Sxy(f) / Sxx(f)
    let mut h1_complex = Vec::with_capacity(cpsd_xy.len());
    let mut frequency_hz = Vec::with_capacity(cpsd_xy.len());

    for (i, (freq, cpsd)) in cpsd_xy.iter().enumerate() {
        let psd_input = psd_xx[i].1;

        if psd_input > 1e-12 {
            let h1 = cpsd / psd_input;
            h1_complex.push(h1);
            frequency_hz.push(*freq);
        }
    }

    if h1_complex.is_empty() {
        return Err(format!("Failed to compute transfer function for {}", axis_name).into());
    }

    // Convert complex H1 to magnitude (dB) and phase (degrees)
    let magnitude_linear: Vec<f64> = h1_complex.iter().map(|h| h.norm()).collect();
    let magnitude_db: Vec<f64> = magnitude_linear
        .iter()
        .map(|&m| to_magnitude_db(m))
        .collect();

    let phase_rad: Vec<f64> = h1_complex.iter().map(|h| h.arg()).collect();
    let phase_deg: Vec<f64> = phase_rad.iter().map(|&p| to_phase_deg(p)).collect();
    let phase_deg_unwrapped = unwrap_phase(&phase_deg);

    // Calculate coherence for quality assessment
    let coherence_result = coherence(&cpsd_xy, &psd_xx, &psd_yy)?;
    let coherence_values: Vec<f64> = coherence_result.iter().map(|(_, coh)| *coh).collect();

    Ok(TransferFunctionResult {
        frequency_hz,
        magnitude_db,
        phase_deg: phase_deg_unwrapped,
        coherence: coherence_values,
        sample_rate_hz: sample_rate,
        axis_name,
    })
}

/// Stability margin confidence level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

/// Stability margins computed from transfer function
#[derive(Debug, Clone)]
pub struct StabilityMargins {
    /// Phase margin in degrees
    pub phase_margin_deg: Option<f64>,
    /// Gain margin in dB
    pub gain_margin_db: Option<f64>,
    /// Gain crossover frequency (0 dB) in Hz
    pub gain_crossover_hz: Option<f64>,
    /// Phase crossover frequency (-180°) in Hz
    pub phase_crossover_hz: Option<f64>,
    /// -3dB bandwidth in Hz
    pub bandwidth_hz: Option<f64>,
    /// Confidence level based on coherence at crossovers
    pub confidence: Confidence,
    /// Warning messages for user
    pub warnings: Vec<String>,
}

impl Default for StabilityMargins {
    fn default() -> Self {
        Self {
            phase_margin_deg: None,
            gain_margin_db: None,
            gain_crossover_hz: None,
            phase_crossover_hz: None,
            bandwidth_hz: None,
            confidence: Confidence::Low,
            warnings: Vec::new(),
        }
    }
}

impl StabilityMargins {
    /// Check if margins indicate stable system
    pub fn is_stable(&self) -> bool {
        if let Some(pm) = self.phase_margin_deg {
            if let Some(gm) = self.gain_margin_db {
                return pm > 0.0 && gm > 0.0;
            }
        }
        false
    }

    /// Check if margins are tight (potentially problematic)
    pub fn has_tight_margins(&self) -> bool {
        if let Some(pm) = self.phase_margin_deg {
            if pm < 30.0 {
                return true;
            }
        }
        if let Some(gm) = self.gain_margin_db {
            if gm < 6.0 {
                return true;
            }
        }
        false
    }
}

/// Calculate stability margins from transfer function result
pub fn calculate_stability_margins(
    tf: &TransferFunctionResult,
) -> Result<StabilityMargins, Box<dyn Error>> {
    if !tf.is_valid() {
        return Err("Invalid transfer function data".into());
    }

    let mut margins = StabilityMargins::default();

    // Find gain crossover frequency (magnitude crosses 0 dB)
    let gain_crossover = find_crossover(&tf.frequency_hz, &tf.magnitude_db, 0.0);

    if let Some((f_c, _)) = gain_crossover {
        margins.gain_crossover_hz = Some(f_c);

        // Calculate phase margin at gain crossover
        if let Some(phase_at_fc) = tf.get_phase_at_freq(f_c) {
            margins.phase_margin_deg = Some(180.0 + phase_at_fc);

            // Check coherence at gain crossover
            if let Some(coh_at_fc) = tf.get_coherence_at_freq(f_c) {
                if coh_at_fc < 0.4 {
                    margins.confidence = Confidence::Low;
                    margins
                        .warnings
                        .push(format!("Low coherence at gain crossover: {:.2}", coh_at_fc));
                } else if coh_at_fc < 0.7 {
                    margins.confidence = Confidence::Medium;
                } else {
                    margins.confidence = Confidence::High;
                }
            }
        }
    } else {
        margins
            .warnings
            .push("No gain crossover found (0 dB crossing)".to_string());
    }

    // Find phase crossover frequency (phase crosses -180°)
    let phase_crossover = find_crossover(&tf.frequency_hz, &tf.phase_deg, -180.0);

    if let Some((f_p, _)) = phase_crossover {
        margins.phase_crossover_hz = Some(f_p);

        // Calculate gain margin at phase crossover
        if let Some(mag_at_fp) = tf.get_magnitude_at_freq(f_p) {
            margins.gain_margin_db = Some(-mag_at_fp);

            // Check coherence at phase crossover
            if let Some(coh_at_fp) = tf.get_coherence_at_freq(f_p) {
                if coh_at_fp < 0.4 && margins.confidence != Confidence::Low {
                    margins.confidence = Confidence::Low;
                    margins.warnings.push(format!(
                        "Low coherence at phase crossover: {:.2}",
                        coh_at_fp
                    ));
                } else if coh_at_fp < 0.7 && margins.confidence == Confidence::High {
                    margins.confidence = Confidence::Medium;
                }
            }
        }
    } else {
        margins
            .warnings
            .push("No phase crossover found (-180° crossing)".to_string());
    }

    // Estimate bandwidth (-3dB point)
    let bandwidth = find_crossover(&tf.frequency_hz, &tf.magnitude_db, -3.0);
    if let Some((f_bw, _)) = bandwidth {
        margins.bandwidth_hz = Some(f_bw);
    }

    // Add warnings for tight margins (only if coherence is reasonable)
    if margins.has_tight_margins() && margins.confidence != Confidence::Low {
        if let Some(pm) = margins.phase_margin_deg {
            if pm < 30.0 {
                margins
                    .warnings
                    .push(format!("Low phase margin: {:.1}° (recommended > 30°)", pm));
            }
        }
        if let Some(gm) = margins.gain_margin_db {
            if gm < 6.0 {
                margins.warnings.push(format!(
                    "Low gain margin: {:.1} dB (recommended > 6 dB)",
                    gm
                ));
            }
        }
    }

    // Check for unstable system (only warn if coherence is reasonable)
    if !margins.is_stable() && margins.confidence != Confidence::Low {
        margins
            .warnings
            .push("System may be unstable (negative margins detected)".to_string());
    }

    // Add note about low coherence affecting reliability
    if margins.confidence == Confidence::Low {
        margins.warnings.push(
            "Note: Low coherence at critical frequencies - margin values may be unreliable"
                .to_string(),
        );
    }

    Ok(margins)
}

/// Find the first crossing of a target value with linear interpolation
///
/// Returns (frequency, value_at_crossing) if found
fn find_crossover(frequencies: &[f64], values: &[f64], target: f64) -> Option<(f64, f64)> {
    if frequencies.len() < 2 || frequencies.len() != values.len() {
        return None;
    }

    for i in 0..values.len() - 1 {
        let v1 = values[i];
        let v2 = values[i + 1];

        // Check if target is between v1 and v2
        if (v1 <= target && target <= v2) || (v2 <= target && target <= v1) {
            let f1 = frequencies[i];
            let f2 = frequencies[i + 1];

            // Linear interpolation
            let t = if (v2 - v1).abs() > 1e-12 {
                (target - v1) / (v2 - v1)
            } else {
                0.5
            };

            let f_cross = f1 + t * (f2 - f1);
            return Some((f_cross, target));
        }
    }

    None
}
