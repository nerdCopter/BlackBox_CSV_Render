// src/fft_utils.rs

use ndarray::{Array1, Array2};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::error::Error;
use std::fs::File; // For diagnostic file
use std::io::Write; // For diagnostic file

use crate::step_response::tukeywin; // For Hanning window
/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
/// Returns the complex frequency spectrum. Handles empty input.
pub fn fft_forward(data: &Array1<f32>) -> Array1<Complex32> {
    if data.is_empty() {
        return Array1::zeros(0);
    }
    let n = data.len();
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut output = planner.make_output_vec();
    if planner.process(&mut input, &mut output).is_err() {
        // Optionally write to diag_file if passed here
        eprintln!("Warning: FFT forward processing failed.");
        let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
        return Array1::zeros(expected_complex_len);
    }
    Array1::from(output)
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of a complex spectrum.
/// Returns the reconstructed real-valued signal. Requires the original signal length N.
/// Normalizes the output. Handles empty input or length mismatches.
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
        eprintln!(
            "Warning: FFT inverse length mismatch. Expected complex length {}, got {}. Returning zeros.",
            expected_complex_len,
            input.len()
        );
        return Array1::zeros(original_length_n);
    }

    if planner.process(&mut input, &mut output).is_ok() {
        let scale = 1.0 / original_length_n as f32;
        let mut output_arr = Array1::from(output);
        output_arr.mapv_inplace(|x| x * scale);
        output_arr
    } else {
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        Array1::zeros(original_length_n)
    }
}

/// Calculates the frequencies for the real FFT output.
#[allow(dead_code)]
pub fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> { // n is fft_window_size, d is sample_interval (1.0/sample_rate)
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let mut freqs = Array1::<f32>::zeros(num_freqs);
    let sample_rate = 1.0 / d;
    if num_freqs <= 1 {
        if num_freqs == 1 { freqs[0] = 0.0; }
        return freqs;
    }
    for i in 0..num_freqs {
        freqs[i] = i as f32 * sample_rate / n as f32;
    }
    freqs
}

/// Calculates Power Spectral Density (PSD) for gyro data, binned by throttle.
/// Uses Welch's method idea by averaging PSDs of overlapping (50%) Hanning-windowed segments.
pub fn calculate_throttle_psd(
    gyro_signal: &Array1<f32>,
    throttle_signal: &Array1<f32>,
    sample_rate: f64,
    num_throttle_bins: usize,
    fft_window_size: usize, // This is SPECTROGRAM_FFT_WINDOW_SIZE_TARGET
    mut diag_file: Option<&mut File>, // Added for diagnostic output
) -> Result<(Array2<f32>, Array1<f32>, Array1<f32>), Box<dyn Error>> {
    if gyro_signal.len() != throttle_signal.len() {
        return Err("Gyro and throttle signals must have the same length.".into());
    }
    if gyro_signal.is_empty() || num_throttle_bins == 0 || fft_window_size == 0 || sample_rate <= 0.0 {
        return Err("Empty signals, zero bins/window size, or invalid sample rate.".into());
    }
    if fft_window_size % 2 != 0 {
        let msg = format!("Warning: fft_window_size {} is odd, even is preferred. Results might be unexpected.", fft_window_size);
        eprintln!("{}", msg);
        if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?; }
    }


    let mut binned_gyro_samples: Vec<Vec<f32>> = vec![Vec::new(); num_throttle_bins];
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;

    for i in 0..gyro_signal.len() {
        let throttle_val = throttle_signal[i].clamp(0.0, 100.0);
        let mut bin_index = (throttle_val / throttle_bin_width).floor() as usize;
        if bin_index >= num_throttle_bins {
            bin_index = num_throttle_bins - 1;
        }
        binned_gyro_samples[bin_index].push(gyro_signal[i]);
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut psd_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let hanning_window = tukeywin(fft_window_size, 1.0);
    let hop_size = fft_window_size / 2; 

    if hop_size == 0 {
        return Err("FFT window size is too small for 50% overlap (hop_size is 0).".into());
    }

    for bin_idx in 0..num_throttle_bins {
        let samples_in_bin = &binned_gyro_samples[bin_idx];
        let num_samples_in_bin = samples_in_bin.len();


        if num_samples_in_bin >= fft_window_size {
            let mut averaged_psd_for_bin = Array1::<f32>::zeros(num_freq_bins_output);
            let mut num_segments_averaged = 0;

            let mut current_pos = 0;
            while current_pos + fft_window_size <= num_samples_in_bin {
                let segment_slice = &samples_in_bin[current_pos..(current_pos + fft_window_size)];
                let mut segment = Array1::from_iter(segment_slice.iter().cloned());

                segment = segment * &hanning_window;

                let spectrum_complex = fft_forward(&segment);
                if spectrum_complex.len() == num_freq_bins_output {
                    for freq_idx in 0..num_freq_bins_output {
                        averaged_psd_for_bin[freq_idx] += spectrum_complex[freq_idx].norm(); 
                    }
                    num_segments_averaged += 1;
                } else {
                     let msg = format!("Warning: FFT output length mismatch for a segment in throttle bin {}. Expected {}, got {}.", bin_idx, num_freq_bins_output, spectrum_complex.len());
                     eprintln!("{}", msg);
                     if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?; }
                }
                current_pos += hop_size; 
            }

            if num_segments_averaged > 0 {
                for freq_idx in 0..num_freq_bins_output {
                    psd_matrix[[freq_idx, bin_idx]] = averaged_psd_for_bin[freq_idx] / num_segments_averaged as f32;
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });

    // --- TEMPORARY DIAGNOSTIC ---
    if let Some(file) = diag_file.as_mut() {
        if !psd_matrix.is_empty() {
            let mut max_psd_val = 0.0f32;
            let mut min_nz_psd_val = f32::MAX; 
            let mut sum_psd_val = 0.0f32;
            let mut count_nz = 0;
            let mut typical_mid_values = Vec::new();
            let (num_freq_bins_diag, num_throttle_bins_diag) = psd_matrix.dim();

            for (idx_tuple, &val) in psd_matrix.indexed_iter() {
                let (freq_idx_diag, throttle_idx_diag) = idx_tuple;
                if val > max_psd_val {
                    max_psd_val = val;
                }
                if val > 1e-9 { 
                    if val < min_nz_psd_val {
                        min_nz_psd_val = val;
                    }
                    sum_psd_val += val;
                    count_nz += 1;

                    if throttle_idx_diag > num_throttle_bins_diag / 3 && throttle_idx_diag < 2 * num_throttle_bins_diag / 3 &&
                       freq_idx_diag > num_freq_bins_diag / 4 && freq_idx_diag < 3 * num_freq_bins_diag / 4 &&
                       typical_mid_values.len() < 20 {
                        typical_mid_values.push(val);
                    }
                }
            }
            writeln!(file, "--- PSD Matrix Diagnostic (fft_utils.rs) ---")?;
            writeln!(file, "FFT Window Size Used: {}", fft_window_size)?;
            writeln!(file, "Dimensions: {} freq_bins x {} throttle_bins", num_freq_bins_diag, num_throttle_bins_diag)?;
            writeln!(file, "Max PSD value: {}", max_psd_val)?;
            if count_nz > 0 {
                writeln!(file, "Min Non-Zero PSD value: {}", min_nz_psd_val)?;
                writeln!(file, "Avg Non-Zero PSD value: {}", sum_psd_val / count_nz as f32)?;
                if !typical_mid_values.is_empty() {
                    writeln!(file, "Some typical mid-range PSD values: {:?}", typical_mid_values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<String>>())?;
                } else {
                     writeln!(file, "No typical mid-range PSD values collected (check conditions or data sparsity).")?;
                }
            } else {
                writeln!(file, "PSD Matrix contains all zero or near-zero values.")?;
            }
            writeln!(file, "Relevant constants from constants.rs for plotting:")?;
            writeln!(file, "  SPECTROGRAM_POWER_CLIP_MAX: {}", crate::constants::SPECTROGRAM_POWER_CLIP_MAX)?;
            writeln!(file, "  MIN_POWER_FOR_LOG_SCALE: {}", crate::constants::MIN_POWER_FOR_LOG_SCALE)?;
            writeln!(file, "------------------------------------------")?;
        }
    }
    // --- END TEMPORARY DIAGNOSTIC ---

    Ok((psd_matrix, freq_bins, throttle_bin_centers))
}

// src/fft_utils.rs
