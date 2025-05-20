// src/fft_utils.rs

use ndarray::{Array1, Array2};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::error::Error;
use std::fs::File;
use std::io::Write;

use crate::step_response::tukeywin;
use crate::constants::{SPECTROGRAM_FFT_OVERLAP_FACTOR, MIN_POWER_FOR_LOG_SCALE};


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


#[allow(dead_code)]
pub fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> {
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

// Returns: (
// Averaged Normalized Magnitude Matrix (for heatmap coloring),
// Frequency Bins,
// Throttle Bin Centers,
// Peak Normalized Magnitude from any individual Segment (for "peak_mag_seg" text display)
// )
// Note: The 4th element (peak_normalized_segment_magnitude) is used for the text display.
// The averaged_normalized_magnitudes_matrix is used for coloring, scaled against BBE_SCALE_HEATMAP.
pub fn calculate_throttle_psd(
    gyro_signal: &Array1<f32>,
    throttle_signal: &Array1<f32>,
    sample_rate: f64,
    num_throttle_bins: usize,
    fft_window_time_ms: f64,
    mut diag_file: Option<&mut File>,
) -> Result<(Array2<f32>, Array1<f32>, Array1<f32>, f32), Box<dyn Error>> {
    if gyro_signal.len() != throttle_signal.len() {
        return Err("Gyro and throttle signals must have the same length.".into());
    }
    if gyro_signal.is_empty() || num_throttle_bins == 0 || fft_window_time_ms <= 0.0 || sample_rate <= 0.0 {
        return Err("Empty signals, zero bins/window time, or invalid sample rate.".into());
    }

    let mut fft_window_size = (fft_window_time_ms / 1000.0 * sample_rate).round() as usize;
    if fft_window_size == 0 { fft_window_size = 2; }  // Ensure not zero
    fft_window_size = fft_window_size.next_power_of_two();

    if fft_window_size == 0 || fft_window_size > gyro_signal.len() {
        return Err(format!(
            "Calculated FFT window size ({}) is invalid or larger than signal length ({}).",
            fft_window_size, gyro_signal.len()
        ).into());
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut sum_normalized_magnitudes_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut counts_matrix = Array2::<usize>::zeros((num_freq_bins_output, num_throttle_bins));

    let hanning_window = tukeywin(fft_window_size, 1.0);
    let hop_size = (fft_window_size / SPECTROGRAM_FFT_OVERLAP_FACTOR).max(1);
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;

    let mut overall_peak_normalized_segment_magnitude = 0.0f32;

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
                let mut normalized_magnitude = raw_magnitude / (fft_window_size as f32);

                if freq_idx != 0 && freq_idx != num_freq_bins_output -1 {
                    normalized_magnitude *= 2.0;
                }

                if normalized_magnitude > overall_peak_normalized_segment_magnitude {
                    overall_peak_normalized_segment_magnitude = normalized_magnitude;
                }
                sum_normalized_magnitudes_matrix[[freq_idx, throttle_bin_index]] += normalized_magnitude;
                counts_matrix[[freq_idx, throttle_bin_index]] += 1;
            }
        } else {
            let msg = format!(
                "Warning: FFT output length mismatch for a segment. Expected {}, got {}. Throttle Bin: {}",
                num_freq_bins_output, spectrum_complex.len(), throttle_bin_index
            );
            eprintln!("{}", msg);
            if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?; }
        }
        current_pos += hop_size;
    }

    let mut averaged_normalized_magnitudes_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut max_val_in_averaged_matrix = 0.0f32; // Track max for diagnostics, not for primary color scaling anymore

    for freq_idx in 0..num_freq_bins_output {
        for bin_idx in 0..num_throttle_bins {
            if counts_matrix[[freq_idx, bin_idx]] > 0 {
                let avg_val = sum_normalized_magnitudes_matrix[[freq_idx, bin_idx]] / counts_matrix[[freq_idx, bin_idx]] as f32;
                averaged_normalized_magnitudes_matrix[[freq_idx, bin_idx]] = avg_val;
                if avg_val > max_val_in_averaged_matrix {
                    max_val_in_averaged_matrix = avg_val;
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- PSD (Normalized Magnitude, x2 for non-DC/Nyquist) Matrix Diagnostic (fft_utils.rs) ---")?;
        writeln!(file, "FFT Window Time (ms): {}, Calculated FFT Window Size (samples): {}", fft_window_time_ms, fft_window_size)?;
        writeln!(file, "Hop Size (samples): {}, Overlap Factor: {}", hop_size, SPECTROGRAM_FFT_OVERLAP_FACTOR)?;
        writeln!(file, "Dimensions: {} freq_bins x {} throttle_bins", averaged_normalized_magnitudes_matrix.shape()[0], averaged_normalized_magnitudes_matrix.shape()[1])?;
        
        let mut min_nz_avg_norm_mag = f32::MAX;
        let mut sum_avg_norm_mag = 0.0f32;
        let mut count_nz = 0;
        
        averaged_normalized_magnitudes_matrix.iter().for_each(|&val| {
            if val > MIN_POWER_FOR_LOG_SCALE { 
                if val < min_nz_avg_norm_mag { min_nz_avg_norm_mag = val; }
                sum_avg_norm_mag += val;
                count_nz +=1;
            }
        });

        writeln!(file, "Overall Peak Normalized Segment Magnitude (for text display 'peak_mag_seg'): {:.2}", overall_peak_normalized_segment_magnitude)?;
        writeln!(file, "Max value in Averaged Normalized Magnitude Matrix (for diagnostics): {:.6}", max_val_in_averaged_matrix)?;
        if count_nz > 0 {
            writeln!(file, "Min Non-Zero Averaged Normalized Magnitude in Matrix: {:.6}", min_nz_avg_norm_mag)?;
            writeln!(file, "Avg Non-Zero Averaged Normalized Magnitude in Matrix: {:.6}", sum_avg_norm_mag / count_nz as f32)?;
        } else {
            writeln!(file, "Averaged Normalized Magnitude Matrix effectively empty or all very small values.")?;
        }
        writeln!(file, "Constant MIN_POWER_FOR_LOG_SCALE (used as floor for plotting): {}", MIN_POWER_FOR_LOG_SCALE)?;
        writeln!(file, "--------------------------------------------------------------------")?;
    }

    // The 4th element is overall_peak_normalized_segment_magnitude, which BBE seems to display as "peak_lin".
    // The averaged_normalized_magnitudes_matrix is used for the heatmap colors, scaled against BBE_SCALE_HEATMAP.
    Ok((
        averaged_normalized_magnitudes_matrix,
        freq_bins,
        throttle_bin_centers,
        overall_peak_normalized_segment_magnitude 
    ))
}
// src/fft_utils.rs
