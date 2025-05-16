// src/fft_utils.rs

use ndarray::{Array1, Array2};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::error::Error;

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
pub fn calculate_throttle_psd(
    gyro_signal: &Array1<f32>,
    throttle_signal: &Array1<f32>,
    sample_rate: f64,
    num_throttle_bins: usize,
    fft_window_size: usize,
) -> Result<(Array2<f32>, Array1<f32>, Array1<f32>), Box<dyn Error>> {
    if gyro_signal.len() != throttle_signal.len() {
        return Err("Gyro and throttle signals must have the same length.".into());
    }
    if gyro_signal.is_empty() || num_throttle_bins == 0 || fft_window_size == 0 || sample_rate <= 0.0 {
        return Err("Empty signals, zero bins/window size, or invalid sample rate.".into());
    }

    let mut binned_gyro_samples: Vec<Vec<f32>> = vec![Vec::new(); num_throttle_bins];
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;

    for i in 0..gyro_signal.len() {
        let throttle_val = throttle_signal[i].clamp(0.0, 100.0);
        let mut bin_index = (throttle_val / throttle_bin_width).floor() as usize;
        if bin_index >= num_throttle_bins {
            bin_index = num_throttle_bins - 1;
        }
        // Diagnostic print for first few samples going into first few bins
        // if i < 20 && bin_index < 2 {
        //     println!("Debug binning: gyro: {:.2}, throttle_val: {:.2}, bin_width: {:.2}, bin_idx: {}", gyro_signal[i], throttle_val, throttle_bin_width, bin_index);
        // }
        binned_gyro_samples[bin_index].push(gyro_signal[i]);
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut psd_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let hanning_window = tukeywin(fft_window_size, 1.0);

    for bin_idx in 0..num_throttle_bins {
        let samples_in_bin = &binned_gyro_samples[bin_idx];
        let num_samples_in_bin = samples_in_bin.len();

        if num_samples_in_bin > 0 {
             println!("Diag: Throttle Bin {}: {} samples", bin_idx, num_samples_in_bin);
        }

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
                     eprintln!("Warning: FFT output length mismatch for a segment in throttle bin {}. Expected {}, got {}.", bin_idx, num_freq_bins_output, spectrum_complex.len());
                }
                current_pos += fft_window_size;
            }

            if num_segments_averaged > 0 {
                println!("Diag: Throttle Bin {}: Averaged {} segments.", bin_idx, num_segments_averaged);
                for freq_idx in 0..num_freq_bins_output {
                    psd_matrix[[freq_idx, bin_idx]] = averaged_psd_for_bin[freq_idx] / num_segments_averaged as f32;
                }
                if bin_idx < 2 || bin_idx >= num_throttle_bins.saturating_sub(2) {
                     println!("Diag: PSD for bin {} (first 5 vals): {:?}", bin_idx, psd_matrix.column(bin_idx).iter().take(5).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });

    Ok((psd_matrix, freq_bins, throttle_bin_centers))
}

// src/fft_utils.rs
