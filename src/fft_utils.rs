//! Utilities for FFT operations and throttle-spectrogram PSD calculation.
//!
//! This module provides:
//!  - fft_forward: real→complex forward FFT
//!  - fft_inverse: complex→real inverse FFT
//!  - fft_rfftfreq: frequency bin generation for real FFT
//!  - calculate_throttle_psd: power spectral density over throttle bins
// src/fft_utils.rs

use ndarray::{Array1, Array2};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::fs::File;
use std::io::Write;
use log::warn;
const TUKEY_ALPHA: f32 = 1.0; // may tune <1.0 to reduce spectral leakage

use crate::step_response::tukeywin;
use crate::constants::{SPECTROGRAM_FFT_OVERLAP_FACTOR};

pub fn fft_forward(data: &Array1<f32>) -> Array1<Complex32> {
    if data.is_empty() {
        return Array1::zeros(0);
    }
    let n = data.len();
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut output = planner.make_output_vec();
    let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    if let Err(err) = planner.process(&mut input, &mut output) {
        warn!("FFT forward processing failed: {}", err);
        return Array1::zeros(expected_complex_len);
    }
    Array1::from(output)
}

pub fn fft_inverse(data: &Array1<Complex32>, original_length_n: usize) -> Array1<f32> {
    if data.is_empty() || original_length_n == 0 {
        return Array1::zeros(original_length_n);
    }
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec();

    let expected_complex_len = if original_length_n % 2 == 0 {
        original_length_n / 2 + 1
    } else {
        (original_length_n + 1) / 2
    };

    if input.len() != expected_complex_len {
        warn!(
            "FFT inverse length mismatch. Expected complex length {}, got {}. Returning zeros.",
            expected_complex_len, input.len()
        );
        return Array1::zeros(original_length_n);
    }

    if let Err(err) = planner.process(&mut input, &mut output) {
        warn!("FFT inverse processing failed: {}", err);
        return Array1::zeros(original_length_n);
    }
    let scale = 1.0 / original_length_n as f32;
    let mut output_arr = Array1::from(output);
    output_arr.mapv_inplace(|x| x * scale);
    output_arr
}

#[allow(dead_code)]
pub fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> {
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let mut freqs = Array1::<f32>::zeros(num_freqs);
    let sample_rate = 1.0f32 / d;
    if num_freqs <= 1 {
        if num_freqs == 1 { freqs[0] = 0.0; }
        return freqs;
    }
    for i in 0..num_freqs {
        freqs[i] = i as f32 * sample_rate / n as f32;
    }
    freqs
}

/// Calculate the power spectral density (PSD) of a gyroscope signal as a function of throttle bins.
///
/// # Arguments
///
/// * `gyro_signal` - Input gyroscope signal (time-domain) as an `Array1<f32>`.
/// * `throttle_signal` - Corresponding throttle signal (0-100%) as an `Array1<f32>`.
/// * `sample_rate` - Sampling rate in Hz as `f64`.
/// * `num_throttle_bins` - Number of throttle bins to categorize throttle signal durations.
/// * `fft_window_time_ms` - FFT window length in milliseconds as `f64`.
/// * `diag_file` - Optional mutable reference to a `File` for writing diagnostic information.
///
/// # Returns
///
/// A `Result` containing:
///
/// * `averaged_amplitude_spectrum_matrix` - 2D array (`Array2<f32>`) of averaged amplitude spectra (frequency bins × throttle bins).
/// * `frequency_bins` - 1D array (`Array1<f32>`) of frequency bin centers in Hz.
/// * `throttle_bin_centers` - 1D array (`Array1<f32>`) of throttle bin center values in percent.
/// * `overall_peak_raw_segment_magnitude` - The peak raw magnitude observed across all FFT segments.
///
/// On error, returns an `Err` encapsulating a boxed error (`Box<dyn std::error::Error>`).
pub fn calculate_throttle_psd(
    gyro_signal: &Array1<f32>,
    throttle_signal: &Array1<f32>,
    sample_rate: f64,
    num_throttle_bins: usize,
    fft_window_time_ms: f64,
    mut diag_file: Option<&mut File>,
) -> Result<(Array2<f32>, Array1<f32>, Array1<f32>, f32), Box<dyn std::error::Error>> {
    if gyro_signal.len() != throttle_signal.len() {
        return Err("Gyro and throttle signals must have the same length.".into());
    }
    if gyro_signal.is_empty() || num_throttle_bins == 0 || fft_window_time_ms <= 0.0 || sample_rate <= 0.0 {
        return Err("Empty signals, zero bins/window time, or invalid sample rate.".into());
    }

    let mut fft_window_size = (fft_window_time_ms / 1000.0 * sample_rate).round() as usize;
    if fft_window_size == 0 { fft_window_size = 2; }
    fft_window_size = fft_window_size.next_power_of_two();

    if fft_window_size == 0 || fft_window_size > gyro_signal.len() {
        return Err(format!(
            "Calculated FFT window size ({}) is invalid or larger than signal length ({}).",
            fft_window_size, gyro_signal.len()
        ).into());
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut sum_amplitude_spectrum_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut counts_matrix = Array2::<usize>::zeros((num_freq_bins_output, num_throttle_bins));

    let hanning_window = tukeywin(fft_window_size, 1.0);
    let hop_size = (fft_window_size / SPECTROGRAM_FFT_OVERLAP_FACTOR).max(1);
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;

    let mut overall_peak_raw_segment_magnitude = 0.0f32;

    let mut current_pos = 0;
    while current_pos + fft_window_size <= gyro_signal.len() {
        let gyro_segment_slice = &gyro_signal.slice(ndarray::s![current_pos..(current_pos + fft_window_size)]);
        let throttle_segment_slice = &throttle_signal.slice(ndarray::s![current_pos..(current_pos + fft_window_size)]);

        let avg_throttle_in_segment: f32 = throttle_segment_slice.mean().unwrap_or(0.0);
        let mut throttle_bin_index = (avg_throttle_in_segment.clamp(0.0, 100.0) / throttle_bin_width).floor() as usize;
        if throttle_bin_index >= num_throttle_bins {
            throttle_bin_index = num_throttle_bins - 1;
        }

        let mut gyro_segment_windowed = gyro_segment_slice.to_owned();
        gyro_segment_windowed = gyro_segment_windowed * &hanning_window;

        let spectrum_complex = fft_forward(&gyro_segment_windowed);

        if spectrum_complex.len() == num_freq_bins_output {
            for freq_idx in 0..num_freq_bins_output {
                let raw_magnitude = spectrum_complex[freq_idx].norm();

                if raw_magnitude > overall_peak_raw_segment_magnitude {
                    overall_peak_raw_segment_magnitude = raw_magnitude;
                }

                let mut amplitude_spectrum_val = raw_magnitude / (fft_window_size as f32);
                if freq_idx != 0 && freq_idx != num_freq_bins_output - 1 {
                    amplitude_spectrum_val *= 2.0;
                }

                sum_amplitude_spectrum_matrix[[freq_idx, throttle_bin_index]] += amplitude_spectrum_val;
                counts_matrix[[freq_idx, throttle_bin_index]] += 1;
            }
        } else {
            return Err(format!(
                "FFT output length ({}) did not match expected ({})",
                spectrum_complex.len(),
                num_freq_bins_output
            ).into());
        }
        current_pos += hop_size;
    }

    let mut averaged_amplitude_spectrum_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut max_val_in_averaged_matrix_diag = 0.0f32; 

    for freq_idx in 0..num_freq_bins_output {
        for bin_idx in 0..num_throttle_bins {
            if counts_matrix[[freq_idx, bin_idx]] > 0 {
                let avg_val = sum_amplitude_spectrum_matrix[[freq_idx, bin_idx]] / counts_matrix[[freq_idx, bin_idx]] as f32;
                averaged_amplitude_spectrum_matrix[[freq_idx, bin_idx]] = avg_val;
                if avg_val > max_val_in_averaged_matrix_diag { // Only for diagnostics
                    max_val_in_averaged_matrix_diag = avg_val;
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- PSD (Averaged N-Norm & x2 Amplitude Spectrum) Matrix Diagnostic (fft_utils.rs) ---")?;
        writeln!(file, "FFT Window Time (ms): {}, Calculated FFT Window Size (samples): {}", fft_window_time_ms, fft_window_size)?;
        writeln!(file, "Overall Peak Raw Segment Magnitude (for text display 'peak_mag_seg'): {:.2}", overall_peak_raw_segment_magnitude)?;
        writeln!(file, "Max value in Averaged Amplitude Spectrum Matrix (for diagnostics): {:.6}", max_val_in_averaged_matrix_diag)?;
        // ... other diagnostics ...
        writeln!(file, "--------------------------------------------------------------------")?;
    }

    Ok((
        averaged_amplitude_spectrum_matrix,
        freq_bins,
        throttle_bin_centers,
        overall_peak_raw_segment_magnitude
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn fft_identity_roundtrip() {
        let data = arr1(&[0.5, 1.5, -0.5, 2.0]);
        let spectrum = fft_forward(&data);
        let recovered = fft_inverse(&spectrum, data.len());
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }

    #[test]
    fn fft_rfftfreq_counts() {
        let freqs = fft_rfftfreq(4, 0.5);
        assert_eq!(freqs.len(), 3);
        assert_eq!(freqs[0], 0.0);
    }
}

// src/fft_utils.rs
/// * `original_length_n` - The length of the original time-domain signal.
///
/// # Returns
/// An `Array1<f32>` containing the reconstructed real signal.
/// If length mismatch or processing error occurs, returns zeros of length `original_length_n`.
pub fn fft_inverse(data: &Array1<Complex32>, original_length_n: usize) -> Array1<f32> {
    if data.is_empty() || original_length_n == 0 {
        return Array1::zeros(original_length_n);
    }
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec();
    let expected_complex_len = if original_length_n % 2 == 0 {
        original_length_n / 2 + 1
    } else {
        (original_length_n + 1) / 2
    };
    if input.len() != expected_complex_len {
        warn!(
            "FFT inverse length mismatch (expected {}, got {}). Returning zeros.",
            expected_complex_len, input.len()
        );
        return Array1::zeros(original_length_n);
    }
    if let Err(err) = planner.process(&mut input, &mut output) {
        warn!("FFT inverse processing failed: {}. Returning zeros.", err);
        return Array1::zeros(original_length_n);
    }
    let scale = 1.0 / original_length_n as f32;
    let mut output_arr = Array1::from(output);
    output_arr.mapv_inplace(|x| x * scale);
    output_arr
}

/// Generates frequency bins for a real-valued FFT of length `n`.
///
/// # Arguments
/// * `n` - The length of the time-domain signal.
/// * `d` - The sample spacing (inverse of sample rate).
///
/// # Returns
/// An `Array1<f32>` of frequency bin centers. Returns an empty array if `n == 0` or `d <= 0.0`.
pub fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> {
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let mut freqs = Array1::<f32>::zeros(num_freqs);
    let sample_rate = 1.0f32 / d;
    if num_freqs <= 1 {
        if num_freqs == 1 { freqs[0] = 0.0; }
        return freqs;
    }
    for i in 0..num_freqs {
        freqs[i] = i as f32 * sample_rate / n as f32;
    }
    freqs
}

/// Computes the power spectral density (PSD) of a gyro signal binned by throttle levels.
///
/// # Arguments
/// * `gyro_signal` - A reference to a 1-D array of gyro measurements.
/// * `throttle_signal` - A reference to a 1-D array of throttle values (0–100).
/// * `sample_rate` - Sampling rate in Hz.
/// * `num_throttle_bins` - Number of discrete throttle bins.
/// * `fft_window_time_ms` - Window length in milliseconds for FFT.
/// * `diag_file` - Optional mutable reference to a `File` for writing diagnostics.
///
/// # Returns
/// `Result<(Array2<f32>, Array1<f32>, Array1<f32>, f32), Box<dyn std::error::Error>>`
///
/// On success, returns a tuple containing:
/// 1. Averaged amplitude spectrum matrix (frequency × throttle).
/// 2. Frequency bins for the FFT.
/// 3. Throttle bin center values.
/// 4. Peak raw magnitude from any segment (for diagnostic display).
///
/// Returns an `Err` if input lengths mismatch or invalid parameters.
pub fn calculate_throttle_psd(
    gyro_signal: &Array1<f32>,
    throttle_signal: &Array1<f32>,
    sample_rate: f64,
    num_throttle_bins: usize,
    fft_window_time_ms: f64,
    mut diag_file: Option<&mut File>,
) -> Result<(Array2<f32>, Array1<f32>, Array1<f32>, f32), Box<dyn std::error::Error>> {
    if gyro_signal.len() != throttle_signal.len() {
        return Err("Gyro and throttle signals must have the same length.".into());
    }
    if gyro_signal.is_empty() || num_throttle_bins == 0 || fft_window_time_ms <= 0.0 || sample_rate <= 0.0 {
        return Err("Empty signals, zero bins/window time, or invalid sample rate.".into());
    }

    let mut fft_window_size = (fft_window_time_ms / 1000.0 * sample_rate).round() as usize;
    if fft_window_size == 0 { fft_window_size = 2; }
    fft_window_size = fft_window_size.next_power_of_two();
    if fft_window_size == 0 || fft_window_size > gyro_signal.len() {
        return Err(format!(
            "Calculated FFT window size ({}) is invalid or larger than signal length ({}).",
            fft_window_size, gyro_signal.len()
        ).into());
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut sum_amplitude_spectrum_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut counts_matrix = Array2::<usize>::zeros((num_freq_bins_output, num_throttle_bins));

    let hanning_window = tukeywin(fft_window_size, TUKEY_ALPHA);
    let hop_size = (fft_window_size / SPECTROGRAM_FFT_OVERLAP_FACTOR).max(1);
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;
    let mut overall_peak_raw_segment_magnitude = 0.0f32;
    let mut current_pos = 0;

    while current_pos + fft_window_size <= gyro_signal.len() {
        let gyro_segment_slice = &gyro_signal.slice(ndarray::s![current_pos..current_pos + fft_window_size]);
        let throttle_segment_slice = &throttle_signal.slice(ndarray::s![current_pos..current_pos + fft_window_size]);

        let avg_throttle_in_segment: f32 = throttle_segment_slice.mean().unwrap_or(0.0);
        let mut throttle_bin_index = (avg_throttle_in_segment.clamp(0.0, 100.0) / throttle_bin_width).floor() as usize;
        if throttle_bin_index >= num_throttle_bins {
            throttle_bin_index = num_throttle_bins - 1;
        }

        let mut gyro_segment_windowed = gyro_segment_slice.to_owned() * &hanning_window;
        let spectrum_complex = fft_forward(&gyro_segment_windowed);

        if spectrum_complex.len() == num_freq_bins_output {
            for freq_idx in 0..num_freq_bins_output {
                let raw_magnitude = spectrum_complex[freq_idx].norm();
                if raw_magnitude > overall_peak_raw_segment_magnitude {
                    overall_peak_raw_segment_magnitude = raw_magnitude;
                }
                let mut amplitude_spectrum_val = raw_magnitude / (fft_window_size as f32);
                if freq_idx != 0 && freq_idx != num_freq_bins_output - 1 {
                    amplitude_spectrum_val *= 2.0;
                }
                sum_amplitude_spectrum_matrix[[freq_idx, throttle_bin_index]] += amplitude_spectrum_val;
                counts_matrix[[freq_idx, throttle_bin_index]] += 1;
            }
        } else {
            return Err(format!(
                "FFT output length ({}) did not match expected ({})",
                spectrum_complex.len(),
                num_freq_bins_output
            ).into());
        }
        current_pos += hop_size;
    }

    let mut averaged_amplitude_spectrum_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut max_val_in_averaged_matrix_diag = 0.0f32;
    for freq_idx in 0..num_freq_bins_output {
        for bin_idx in 0..num_throttle_bins {
            if counts_matrix[[freq_idx, bin_idx]] > 0 {
                let avg_val =
                    sum_amplitude_spectrum_matrix[[freq_idx, bin_idx]] / counts_matrix[[freq_idx, bin_idx]] as f32;
                averaged_amplitude_spectrum_matrix[[freq_idx, bin_idx]] = avg_val;
                if avg_val > max_val_in_averaged_matrix_diag {
                    max_val_in_averaged_matrix_diag = avg_val;
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| (i as f32 + 0.5) * throttle_bin_width);

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- PSD (Averaged N-Norm & x2 Amplitude Spectrum) Matrix Diagnostic (fft_utils.rs) ---")?;
        writeln!(file, "FFT Window Time (ms): {}, Calculated FFT Window Size (samples): {}", fft_window_time_ms, fft_window_size)?;
        writeln!(file, "Overall Peak Raw Segment Magnitude (for text display 'peak_mag_seg'): {:.2}", overall_peak_raw_segment_magnitude)?;
        writeln!(file, "Max value in Averaged Amplitude Spectrum Matrix (for diagnostics): {:.6}", max_val_in_averaged_matrix_diag)?;
        // ... other diagnostics ...
        writeln!(file, "--------------------------------------------------------------------")?;
    }

    Ok((
        averaged_amplitude_spectrum_matrix,
        freq_bins,
        throttle_bin_centers,
        overall_peak_raw_segment_magnitude,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn fft_identity_roundtrip() {
        let data = arr1(&[0.5, 1.5, -0.5, 2.0]);
        let spectrum = fft_forward(&data);
        let recovered = fft_inverse(&spectrum, data.len());
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }

    #[test]
    fn fft_rfftfreq_counts() {
        let freqs = fft_rfftfreq(4, 0.5);
        assert_eq!(freqs.len(), 3);
        assert_eq!(freqs[0], 0.0);
    }
}

// src/fft_utils.rs