// src/data_analysis/transfer_function_estimation.rs

use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COHERENCE_HIGH_THRESHOLD, COHERENCE_MEDIUM_THRESHOLD, FREQUENCY_EPSILON, PSD_EPSILON,
    VALUE_EPSILON,
};
use crate::data_analysis::spectral_analysis::{
    coherence, to_magnitude_db, to_phase_deg, welch_cpsd, welch_psd, WelchConfig,
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
    /// Phase in degrees (wrapped to [-180°, +180°])
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
        !self.frequency_hz.is_empty()
            && self.frequency_hz.len() == self.magnitude_db.len()
            && self.frequency_hz.len() == self.phase_deg.len()
            && self.frequency_hz.len() == self.coherence.len()
    }

    /// Get number of frequency points
    pub fn len(&self) -> usize {
        self.frequency_hz.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.frequency_hz.is_empty()
    }

    /// Find index closest to a given frequency using binary search
    pub fn find_frequency_index(&self, target_freq: f64) -> Option<usize> {
        if self.frequency_hz.is_empty() {
            return None;
        }

        // Handle NaN target frequency
        if target_freq.is_nan() {
            return None;
        }

        // Binary search for exact match or insertion point
        match self.frequency_hz.binary_search_by(|freq| {
            freq.partial_cmp(&target_freq)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => Some(idx), // Exact match found
            Err(idx) => {
                // idx is the insertion point; compare neighbors for closest match
                if idx == 0 {
                    // target_freq is less than all frequencies
                    Some(0)
                } else if idx >= self.frequency_hz.len() {
                    // target_freq is greater than all frequencies
                    Some(self.frequency_hz.len() - 1)
                } else {
                    // target_freq is between idx-1 and idx; pick the closer one
                    let left_diff = (self.frequency_hz[idx - 1] - target_freq).abs();
                    let right_diff = (self.frequency_hz[idx] - target_freq).abs();
                    if left_diff <= right_diff {
                        Some(idx - 1)
                    } else {
                        Some(idx)
                    }
                }
            }
        }
    }

    /// Get magnitude at a specific frequency (with interpolation)
    ///
    /// Returns the value at the edge if target_freq is outside the frequency range.
    pub fn get_magnitude_at_freq(&self, target_freq: f64) -> Option<f64> {
        if self.frequency_hz.len() < 2 {
            return None;
        }

        // Handle out-of-bounds cases explicitly
        if target_freq < self.frequency_hz[0] {
            return self.magnitude_db.first().copied();
        }
        if target_freq > *self.frequency_hz.last().unwrap() {
            return self.magnitude_db.last().copied();
        }

        // Find bracketing indices
        let mut idx = 0;
        while idx < self.frequency_hz.len() - 1 && self.frequency_hz[idx + 1] < target_freq {
            idx += 1;
        }

        // Linear interpolation
        let f1 = self.frequency_hz[idx];
        let f2 = self.frequency_hz[idx + 1];
        let m1 = self.magnitude_db[idx];
        let m2 = self.magnitude_db[idx + 1];

        let t = if (f2 - f1).abs() > FREQUENCY_EPSILON {
            (target_freq - f1) / (f2 - f1)
        } else {
            0.5
        };
        Some(m1 + t * (m2 - m1))
    }

    /// Get phase at a specific frequency (with wrap-aware interpolation)
    ///
    /// Returns the value at the edge if target_freq is outside the frequency range.
    /// Uses wrap-aware interpolation to correctly handle ±180° discontinuities.
    pub fn get_phase_at_freq(&self, target_freq: f64) -> Option<f64> {
        if self.frequency_hz.len() < 2 {
            return None;
        }

        // Handle out-of-bounds cases explicitly
        if target_freq < self.frequency_hz[0] {
            return self.phase_deg.first().copied();
        }
        if target_freq > *self.frequency_hz.last().unwrap() {
            return self.phase_deg.last().copied();
        }

        // Find bracketing indices
        let mut idx = 0;
        while idx < self.frequency_hz.len() - 1 && self.frequency_hz[idx + 1] < target_freq {
            idx += 1;
        }

        // Wrap-aware phase interpolation
        let f1 = self.frequency_hz[idx];
        let f2 = self.frequency_hz[idx + 1];
        let p1 = self.phase_deg[idx];
        let p2 = self.phase_deg[idx + 1];

        let t = if (f2 - f1).abs() > FREQUENCY_EPSILON {
            (target_freq - f1) / (f2 - f1)
        } else {
            0.5
        };
        Some(interpolate_phase(p1, p2, t))
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
    // H(f) = Output/Input = Gyro/Setpoint
    // H1 estimator: H1(f) = Syx(f) / Sxx(f) where Syx = conj(X) * Y = conj(Setpoint) * Gyro
    let cpsd_yx = welch_cpsd(
        &gyro_data,     // signal1 (output Y)
        &setpoint_data, // signal2 (input X)
        sample_rate,
        Some(config.clone()),
    )?;
    let psd_xx = welch_psd(&setpoint_data, sample_rate, Some(config.clone()))?;
    let psd_yy = welch_psd(&gyro_data, sample_rate, Some(config))?;

    // Compute H1 estimator: H1(f) = Syx(f) / Sxx(f)
    let mut h1_complex = Vec::with_capacity(cpsd_yx.len());
    let mut frequency_hz = Vec::with_capacity(cpsd_yx.len());
    let mut coherence_values = Vec::with_capacity(cpsd_yx.len());

    // Calculate coherence for quality assessment
    let coherence_result = coherence(&cpsd_yx, &psd_xx, &psd_yy)?;

    for (i, (freq, cpsd)) in cpsd_yx.iter().enumerate() {
        let psd_input = psd_xx[i].1;

        if psd_input > PSD_EPSILON {
            // cpsd_yx from welch_cpsd computes: Gyro(f) * conj(Setpoint(f))
            // But we need: conj(Setpoint(f)) * Gyro(f) for correct phase
            // These differ by conjugation, so conjugate cpsd to get correct phase sign
            let h1 = cpsd.conj() / psd_input;
            h1_complex.push(h1);
            frequency_hz.push(*freq);
            // Keep coherence value aligned with frequency and magnitude
            coherence_values.push(coherence_result[i].1);
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
    // Note: Phase is kept wrapped in -180° to +180° range for typical Bode plot display
    // Unwrapping can cause issues with noisy/low-coherence data from non-ideal test signals

    Ok(TransferFunctionResult {
        frequency_hz,
        magnitude_db,
        phase_deg, // Use wrapped phase instead of unwrapped
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
    let gain_crossover = find_crossover(&tf.frequency_hz, &tf.magnitude_db, 0.0, WrapMode::Linear);

    if let Some((f_c, _)) = gain_crossover {
        margins.gain_crossover_hz = Some(f_c);

        // Calculate phase margin at gain crossover
        if let Some(phase_at_fc) = tf.get_phase_at_freq(f_c) {
            margins.phase_margin_deg = Some(180.0 + phase_at_fc);

            // Check coherence at gain crossover
            if let Some(coh_at_fc) = tf.get_coherence_at_freq(f_c) {
                if coh_at_fc < COHERENCE_MEDIUM_THRESHOLD {
                    margins.confidence = Confidence::Low;
                    margins
                        .warnings
                        .push(format!("Low coherence at gain crossover: {:.2}", coh_at_fc));
                } else if coh_at_fc < COHERENCE_HIGH_THRESHOLD {
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
    let phase_crossover =
        find_crossover(&tf.frequency_hz, &tf.phase_deg, -180.0, WrapMode::Circular);

    if let Some((f_p, _)) = phase_crossover {
        margins.phase_crossover_hz = Some(f_p);

        // Calculate gain margin at phase crossover
        if let Some(mag_at_fp) = tf.get_magnitude_at_freq(f_p) {
            margins.gain_margin_db = Some(-mag_at_fp);

            // Check coherence at phase crossover
            if let Some(coh_at_fp) = tf.get_coherence_at_freq(f_p) {
                if coh_at_fp < COHERENCE_MEDIUM_THRESHOLD && margins.confidence != Confidence::Low {
                    margins.confidence = Confidence::Low;
                    margins.warnings.push(format!(
                        "Low coherence at phase crossover: {:.2}",
                        coh_at_fp
                    ));
                } else if coh_at_fp < COHERENCE_HIGH_THRESHOLD
                    && margins.confidence == Confidence::High
                {
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
    // For systems with passband ripple, use the last -3dB crossing to get accurate bandwidth
    let bandwidth = find_last_crossover(&tf.frequency_hz, &tf.magnitude_db, -3.0);
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

/// Interpolates phase values using wrap-aware shortest-path interpolation
///
/// Handles ±180° discontinuities correctly by:
/// 1. Computing the signed shortest angular difference between p1 and p2
/// 2. Adjusting diff by ±360° if abs(diff) > 180° to take the shortest path
/// 3. Interpolating along that shortest path using parameter t
/// 4. Re-wrapping the result back into [-180.0, 180.0]
///
/// # Arguments
/// * `p1` - First phase value (degrees) in [-180.0, 180.0]
/// * `p2` - Second phase value (degrees) in [-180.0, 180.0]
/// * `t` - Interpolation parameter (0.0 = p1, 1.0 = p2)
///
/// # Returns
/// Interpolated phase value in [-180.0, 180.0]
fn interpolate_phase(p1: f64, p2: f64, t: f64) -> f64 {
    // Use helper for shortest angular difference
    let diff = wrap_aware_diff(p1, p2);

    // Interpolate along shortest path
    let interpolated = p1 + t * diff;

    // Re-wrap result back into [-180.0, 180.0] using modular arithmetic
    ((interpolated + 180.0).rem_euclid(360.0)) - 180.0
}

/// Find the first crossing of a target value with wrap-aware detection
///
/// Finds the first frequency at which a signal crosses a target value.
/// For phase-like values (abs(target) > 90°), uses wrap-aware circular comparison
/// to correctly detect crossings at ±180° discontinuities.
/// For other values, uses standard linear interval testing.
///
/// Returns (frequency, value_at_crossing) if found.
///
/// # Arguments
/// * `frequencies` - Sorted array of frequency values (Hz)
/// * `values` - Array of magnitude or phase values to search
/// * `target` - Target value to cross
///
/// # Returns
/// `Some((crossover_frequency, target_value))` if a crossing is found,
/// or `None` if no crossing found or arrays are invalid.
/// Mode for phase crossing detection
///
/// Controls whether to use wrap-aware circular arithmetic (for phase values)
/// or standard linear comparison (for magnitude values).
#[derive(Debug, Clone, Copy)]
enum WrapMode {
    /// Standard linear interval testing for magnitude-like values
    Linear,
    /// Wrap-aware circular comparison for phase values at ±180° boundary
    Circular,
}

/// Compute shortest signed angular difference from v1 to v2
///
/// Normalizes the difference to stay within [-180.0, 180.0] to represent
/// the shortest angular path. Used for wrap-aware phase comparisons and interpolations.
///
/// # Arguments
/// * `v1` - First angle (degrees)
/// * `v2` - Second angle (degrees)
///
/// # Returns
/// Shortest signed angular difference in [-180.0, 180.0]
fn wrap_aware_diff(v1: f64, v2: f64) -> f64 {
    let mut diff = v2 - v1;
    if diff > 180.0 {
        diff -= 360.0;
    } else if diff < -180.0 {
        diff += 360.0;
    }
    diff
}

fn find_crossover(
    frequencies: &[f64],
    values: &[f64],
    target: f64,
    mode: WrapMode,
) -> Option<(f64, f64)> {
    if frequencies.len() < 2 || frequencies.len() != values.len() {
        return None;
    }

    for i in 0..values.len() - 1 {
        let v1 = values[i];
        let v2 = values[i + 1];

        // Skip pairs containing NaN to prevent silent comparison failures
        if v1.is_nan() || v2.is_nan() {
            continue;
        }

        // Determine crossing logic based on WrapMode
        let crosses_target = match mode {
            WrapMode::Circular => {
                // Normalize angles to [0, 360) for circular arithmetic
                let normalize = |angle: f64| {
                    let mut a = angle % 360.0;
                    if a < 0.0 {
                        a += 360.0;
                    }
                    a
                };

                let v1_norm = normalize(v1);
                let target_norm = normalize(target);

                // Compute shortest signed angular difference from v1 to v2
                let diff = wrap_aware_diff(v1, v2);

                // Compute target offset from v1 to target in the [0, 360) modular space
                let target_offset = (target_norm - v1_norm + 360.0) % 360.0;

                // Test crossing based on traversal direction:
                // If diff >= 0 (clockwise): target is crossed if 0 <= target_offset <= diff
                // If diff < 0 (counterclockwise): target is crossed if target_offset >= (360.0 + diff)
                if diff >= 0.0 {
                    0.0 <= target_offset && target_offset <= diff
                } else {
                    target_offset >= (360.0 + diff)
                }
            }
            WrapMode::Linear => {
                // Standard case: magnitude or mid-range phase
                (v1 <= target && target <= v2) || (v2 <= target && target <= v1)
            }
        };

        if crosses_target {
            let f1 = frequencies[i];
            let f2 = frequencies[i + 1];

            // Calculate interpolation parameter based on WrapMode
            let t = match mode {
                WrapMode::Circular => {
                    // Phase crossover: use wrap-aware numerator and denominator
                    let diff = wrap_aware_diff(v1, v2);

                    // Compute wrap-aware numerator (distance from v1 to target)
                    let mut num = target - v1;
                    if num > 180.0 {
                        num -= 360.0;
                    } else if num < -180.0 {
                        num += 360.0;
                    }

                    if diff.abs() > VALUE_EPSILON {
                        num / diff
                    } else {
                        0.5
                    }
                }
                WrapMode::Linear => {
                    // Standard interpolation
                    if (v2 - v1).abs() > VALUE_EPSILON {
                        (target - v1) / (v2 - v1)
                    } else {
                        0.5
                    }
                }
            };

            let f_cross = f1 + t * (f2 - f1);
            return Some((f_cross, target));
        }
    }

    None
}

/// Find the last crossing of a target value (for bandwidth with passband ripple)
///
/// Finds the final crossing point where the values cross the target level.
/// Useful for bandwidth estimation when passband ripple exists — returns the
/// last -3dB crossing instead of the first, providing accurate bandwidth.
///
/// **Note**: Uses standard linear interpolation without wrap-aware handling.
/// This is correct for magnitude-based bandwidth estimation. For phase values
/// that may wrap at ±180°, use `find_crossover` instead, which handles
/// wrap-aware circular detection.
///
/// # Arguments
/// * `frequencies` - Sorted array of frequency values (Hz)
/// * `values` - Array of magnitude or phase values
/// * `target` - Target value to cross
///
/// # Returns
/// `Some((crossover_frequency, target_value))` for the last crossing,
/// or `None` if no crossing found or arrays are invalid.
fn find_last_crossover(frequencies: &[f64], values: &[f64], target: f64) -> Option<(f64, f64)> {
    if frequencies.len() < 2 || frequencies.len() != values.len() {
        return None;
    }

    let mut last_crossing = None;

    for i in 0..values.len() - 1 {
        let v1 = values[i];
        let v2 = values[i + 1];

        // Check if target is between v1 and v2
        if (v1 <= target && target <= v2) || (v2 <= target && target <= v1) {
            let f1 = frequencies[i];
            let f2 = frequencies[i + 1];

            // Linear interpolation
            let t = if (v2 - v1).abs() > VALUE_EPSILON {
                (target - v1) / (v2 - v1)
            } else {
                0.5
            };

            let f_cross = f1 + t * (f2 - f1);
            last_crossing = Some((f_cross, target));
        }
    }

    last_crossing
}
