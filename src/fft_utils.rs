// src/fft_utils.rs

use ndarray::{Array1, Array2};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::error::Error;
use std::fs::File;
use std::io::Write;

use crate::step_response::tukeywin;
use crate::constants::{SPECTROGRAM_FFT_OVERLAP_FACTOR, AUTO_CLIP_MAX_SCALE_FACTOR, MIN_POWER_FOR_LOG_SCALE};


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

// Returns: (Averaged PSD Matrix, Frequency Bins, Throttle Bin Centers, Max Power from Averaged PSD for Auto-Clipping)
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
    if fft_window_size == 0 { fft_window_size = 2; }
    fft_window_size = fft_window_size.next_power_of_two();

    if fft_window_size == 0 || fft_window_size > gyro_signal.len() {
        return Err(format!(
            "Calculated FFT window size ({}) is invalid or larger than signal length ({}).",
            fft_window_size, gyro_signal.len()
        ).into());
    }

    let num_freq_bins_output = fft_window_size / 2 + 1;
    let mut psd_matrix_sum = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut psd_matrix_counts = Array2::<usize>::zeros((num_freq_bins_output, num_throttle_bins));

    let hanning_window = tukeywin(fft_window_size, 1.0); // Alpha 1.0 for Hanning
    let hop_size = (fft_window_size / SPECTROGRAM_FFT_OVERLAP_FACTOR).max(1);
    let throttle_bin_width = 100.0 / num_throttle_bins as f32;

    let mut current_pos = 0;
    while current_pos + fft_window_size <= gyro_signal.len() {
        let gyro_segment_slice = &gyro_signal.slice(ndarray::s![current_pos..(current_pos + fft_window_size)]);
        let throttle_segment_slice = &throttle_signal.slice(ndarray::s![current_pos..(current_pos + fft_window_size)]);

        // Calculate average throttle for this segment
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
                let power = spectrum_complex[freq_idx].norm(); // Magnitude of complex number
                psd_matrix_sum[[freq_idx, throttle_bin_index]] += power;
                psd_matrix_counts[[freq_idx, throttle_bin_index]] += 1;
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

    let mut averaged_psd_matrix = Array2::<f32>::zeros((num_freq_bins_output, num_throttle_bins));
    let mut overall_max_averaged_psd_val = 0.0f32;

    for freq_idx in 0..num_freq_bins_output {
        for bin_idx in 0..num_throttle_bins {
            if psd_matrix_counts[[freq_idx, bin_idx]] > 0 {
                let avg_power = psd_matrix_sum[[freq_idx, bin_idx]] / psd_matrix_counts[[freq_idx, bin_idx]] as f32;
                averaged_psd_matrix[[freq_idx, bin_idx]] = avg_power;
                if avg_power > overall_max_averaged_psd_val {
                    overall_max_averaged_psd_val = avg_power;
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });

    // This is the value used for the "peak_lin" display and for auto-ranging the colormap.
    let effective_clip_max_for_plot = overall_max_averaged_psd_val * AUTO_CLIP_MAX_SCALE_FACTOR;

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- PSD Matrix Diagnostic (fft_utils.rs) ---")?;
        writeln!(file, "FFT Window Time (ms): {}, Calculated FFT Window Size (samples): {}", fft_window_time_ms, fft_window_size)?;
        writeln!(file, "Hop Size (samples): {}, Overlap Factor: {}", hop_size, SPECTROGRAM_FFT_OVERLAP_FACTOR)?;
        writeln!(file, "Dimensions: {} freq_bins x {} throttle_bins", averaged_psd_matrix.shape()[0], averaged_psd_matrix.shape()[1])?;
        
        let mut min_nz_psd_val = f32::MAX;
        let mut sum_psd_val = 0.0f32;
        let mut count_nz = 0;
        averaged_psd_matrix.iter().for_each(|&val| {
            if val > 1e-9 { // Consider non-zero
                if val < min_nz_psd_val { min_nz_psd_val = val; }
                sum_psd_val += val;
                count_nz +=1;
            }
        });

        writeln!(file, "Max Averaged PSD value (overall_max_averaged_psd_val): {}", overall_max_averaged_psd_val)?;
        writeln!(file, "Effective Clip Max for Plotting (based on averaged PSD, before log): {}", effective_clip_max_for_plot)?;
        if count_nz > 0 {
            writeln!(file, "Min Non-Zero Averaged PSD value in Matrix: {}", min_nz_psd_val)?;
            writeln!(file, "Avg Non-Zero Averaged PSD value in Matrix: {}", sum_psd_val / count_nz as f32)?;
        } else {
            writeln!(file, "Averaged PSD Matrix effectively empty or all zeros.")?;
        }
        writeln!(file, "Constant MIN_POWER_FOR_LOG_SCALE: {}", MIN_POWER_FOR_LOG_SCALE)?;
        writeln!(file, "Constant AUTO_CLIP_MAX_SCALE_FACTOR: {}", AUTO_CLIP_MAX_SCALE_FACTOR)?;
        writeln!(file, "------------------------------------------")?;
    }

    Ok((averaged_psd_matrix, freq_bins, throttle_bin_centers, effective_clip_max_for_plot))
}
// src/fft_utils.rs
