// src/data_analysis/spectral_analysis.rs

use ndarray::Array1;
use num_complex::Complex64;
use std::error::Error;

use crate::data_analysis::{calc_step_response, fft_utils};

/// Configuration for Welch's method spectral analysis
#[derive(Debug, Clone)]
pub struct WelchConfig {
    /// Segment length in samples (default: data_length / 8 for minimum 8 averages)
    pub segment_length: usize,
    /// Overlap percentage (default: 50%)
    pub overlap_percent: f64,
}

impl Default for WelchConfig {
    fn default() -> Self {
        Self {
            segment_length: 0, // Will be calculated based on data length
            overlap_percent: 0.5,
        }
    }
}

/// Generates a Hanning window of specified length
fn hanning_window(length: usize) -> Array1<f32> {
    // Hanning window is Tukey with alpha=1.0
    calc_step_response::tukeywin(length, 1.0)
}

/// Calculates the frequency vector for FFT results
pub fn frequency_vector(nfft: usize, sample_rate: f64) -> Vec<f64> {
    let num_freqs = nfft / 2 + 1;
    (0..num_freqs)
        .map(|i| (i as f64 * sample_rate) / (nfft as f64))
        .collect()
}

/// Converts magnitude to decibels
pub fn to_magnitude_db(magnitude: f64) -> f64 {
    if magnitude > 0.0 {
        20.0 * magnitude.log10()
    } else {
        -100.0 // Floor for zero/negative values
    }
}

/// Converts phase from radians to degrees
pub fn to_phase_deg(phase_rad: f64) -> f64 {
    phase_rad.to_degrees()
}

/// Unwraps phase to remove 360° discontinuities
#[allow(dead_code)]
pub fn unwrap_phase(phase_deg: &[f64]) -> Vec<f64> {
    if phase_deg.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = Vec::with_capacity(phase_deg.len());
    unwrapped.push(phase_deg[0]);

    let mut cumulative_offset = 0.0;

    for i in 1..phase_deg.len() {
        let mut diff = phase_deg[i] - phase_deg[i - 1];

        // Detect and correct jumps > 180°
        while diff > 180.0 {
            diff -= 360.0;
            cumulative_offset -= 360.0;
        }
        while diff < -180.0 {
            diff += 360.0;
            cumulative_offset += 360.0;
        }

        unwrapped.push(phase_deg[i] + cumulative_offset);
    }

    unwrapped
}

/// Computes Power Spectral Density using Welch's method
///
/// Segments the signal with overlap, applies windowing, computes FFT,
/// and averages power across segments.
pub fn welch_psd(
    signal: &[f32],
    sample_rate: f64,
    config: Option<WelchConfig>,
) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    if signal.is_empty() {
        return Err("Empty signal provided".into());
    }

    let mut cfg = config.unwrap_or_default();

    // Calculate segment length if not specified (aim for 8 averages minimum)
    if cfg.segment_length == 0 {
        cfg.segment_length = (signal.len() / 8).max(256).next_power_of_two();
    }

    let segment_length = cfg.segment_length;
    let hop_size = ((1.0 - cfg.overlap_percent) * segment_length as f64) as usize;

    if hop_size == 0 {
        return Err("Invalid overlap percentage".into());
    }

    // Generate window
    let window = hanning_window(segment_length);
    let window_power: f64 = window.iter().map(|&w| (w as f64) * (w as f64)).sum();

    // Calculate number of segments
    let num_segments = if signal.len() >= segment_length {
        (signal.len() - segment_length) / hop_size + 1
    } else {
        return Err("Signal too short for specified segment length".into());
    };

    let nfft = segment_length.next_power_of_two();
    let num_freqs = nfft / 2 + 1;

    // Accumulator for averaged PSD
    let mut psd_sum = vec![0.0f64; num_freqs];
    let mut segment_count = 0;

    // Process each segment
    for seg_idx in 0..num_segments {
        let start = seg_idx * hop_size;
        let end = start + segment_length;

        if end > signal.len() {
            break;
        }

        let segment_slice = &signal[start..end];

        // Apply window and zero-pad
        let mut padded = Array1::<f32>::zeros(nfft);
        for i in 0..segment_length {
            padded[i] = segment_slice[i] * window[i];
        }

        // Compute FFT
        let spectrum = fft_utils::fft_forward(&padded);

        if spectrum.is_empty() {
            continue;
        }

        // Accumulate power spectral density
        for i in 0..num_freqs.min(spectrum.len()) {
            let magnitude_sqr = spectrum[i].norm_sqr() as f64;

            // Proper PSD calculation with normalization
            let mut psd = magnitude_sqr / (sample_rate * segment_length as f64 * window_power);

            // One-sided spectrum: double power for positive frequencies (except DC and Nyquist)
            let is_nyquist = (nfft % 2 == 0) && (i == num_freqs - 1);
            if i > 0 && !is_nyquist {
                psd *= 2.0;
            }

            psd_sum[i] += psd;
        }

        segment_count += 1;
    }

    if segment_count == 0 {
        return Err("No valid segments processed".into());
    }

    // Average and create frequency-PSD pairs
    let frequencies = frequency_vector(nfft, sample_rate);
    let psd_avg: Vec<(f64, f64)> = frequencies
        .iter()
        .zip(psd_sum.iter())
        .map(|(&freq, &psd)| (freq, psd / segment_count as f64))
        .collect();

    Ok(psd_avg)
}

/// Computes Cross-Power Spectral Density using Welch's method
///
/// Returns complex-valued CPSD averaged across segments
pub fn welch_cpsd(
    signal1: &[f32],
    signal2: &[f32],
    sample_rate: f64,
    config: Option<WelchConfig>,
) -> Result<Vec<(f64, Complex64)>, Box<dyn Error>> {
    if signal1.is_empty() || signal2.is_empty() {
        return Err("Empty signal provided".into());
    }

    if signal1.len() != signal2.len() {
        return Err("Signals must have equal length".into());
    }

    let mut cfg = config.unwrap_or_default();

    // Calculate segment length if not specified
    if cfg.segment_length == 0 {
        cfg.segment_length = (signal1.len() / 8).max(256).next_power_of_two();
    }

    let segment_length = cfg.segment_length;
    let hop_size = ((1.0 - cfg.overlap_percent) * segment_length as f64) as usize;

    if hop_size == 0 {
        return Err("Invalid overlap percentage".into());
    }

    // Generate window
    let window = hanning_window(segment_length);
    let window_power: f64 = window.iter().map(|&w| (w as f64) * (w as f64)).sum();

    // Calculate number of segments
    let num_segments = if signal1.len() >= segment_length {
        (signal1.len() - segment_length) / hop_size + 1
    } else {
        return Err("Signal too short for specified segment length".into());
    };

    let nfft = segment_length.next_power_of_two();
    let num_freqs = nfft / 2 + 1;

    // Accumulator for averaged CPSD
    let mut cpsd_sum = vec![Complex64::new(0.0, 0.0); num_freqs];
    let mut segment_count = 0;

    // Process each segment
    for seg_idx in 0..num_segments {
        let start = seg_idx * hop_size;
        let end = start + segment_length;

        if end > signal1.len() {
            break;
        }

        let segment1_slice = &signal1[start..end];
        let segment2_slice = &signal2[start..end];

        // Apply window and zero-pad for both signals
        let mut padded1 = Array1::<f32>::zeros(nfft);
        let mut padded2 = Array1::<f32>::zeros(nfft);
        for i in 0..segment_length {
            padded1[i] = segment1_slice[i] * window[i];
            padded2[i] = segment2_slice[i] * window[i];
        }

        // Compute FFTs
        let spectrum1 = fft_utils::fft_forward(&padded1);
        let spectrum2 = fft_utils::fft_forward(&padded2);

        if spectrum1.is_empty() || spectrum2.is_empty() {
            continue;
        }

        // Accumulate cross-power spectral density
        for i in 0..num_freqs.min(spectrum1.len()).min(spectrum2.len()) {
            // CPSD: X(f) * conj(Y(f))
            let cross_power = Complex64::new(spectrum1[i].re as f64, spectrum1[i].im as f64)
                * Complex64::new(spectrum2[i].re as f64, -spectrum2[i].im as f64); // conjugate

            // Normalize
            let mut cpsd = cross_power / (sample_rate * segment_length as f64 * window_power);

            // One-sided spectrum: double for positive frequencies
            let is_nyquist = (nfft % 2 == 0) && (i == num_freqs - 1);
            if i > 0 && !is_nyquist {
                cpsd *= 2.0;
            }

            cpsd_sum[i] += cpsd;
        }

        segment_count += 1;
    }

    if segment_count == 0 {
        return Err("No valid segments processed".into());
    }

    // Average and create frequency-CPSD pairs
    let frequencies = frequency_vector(nfft, sample_rate);
    let cpsd_avg: Vec<(f64, Complex64)> = frequencies
        .iter()
        .zip(cpsd_sum.iter())
        .map(|(&freq, &cpsd)| (freq, cpsd / segment_count as f64))
        .collect();

    Ok(cpsd_avg)
}

/// Calculates coherence: γ²(f) = |Sxy(f)|² / (Sxx(f) × Syy(f))
pub fn coherence(
    cpsd: &[(f64, Complex64)],
    psd1: &[(f64, f64)],
    psd2: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    if cpsd.len() != psd1.len() || cpsd.len() != psd2.len() {
        return Err("Input arrays must have equal length".into());
    }

    let mut coh = Vec::with_capacity(cpsd.len());

    for i in 0..cpsd.len() {
        let freq = cpsd[i].0;
        let cpsd_mag_sqr = cpsd[i].1.norm_sqr();
        let psd_product = psd1[i].1 * psd2[i].1;

        let coherence_val = if psd_product > 1e-12 {
            (cpsd_mag_sqr / psd_product).min(1.0) // Clamp to [0, 1]
        } else {
            0.0
        };

        coh.push((freq, coherence_val));
    }

    Ok(coh)
}
