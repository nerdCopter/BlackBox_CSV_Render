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

// Returns: (PSD Matrix, Frequency Bins, Throttle Bin Centers, Detected Max Linear Power for Auto-Clipping)
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
    if fft_window_size == 0 { // Ensure fft_window_size is not zero before next_power_of_two
        fft_window_size = 2; // A very small default if calculated as zero
    }
    fft_window_size = fft_window_size.next_power_of_two(); 

    if fft_window_size == 0 { // Should be caught by above, but as a failsafe
        return Err("Calculated FFT window size is zero after attempting to make it power of two.".into());
    }
    // Note: fft_window_size % 2 != 0 check is now redundant due to next_power_of_two (unless size is 1, which is unlikely)

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
    
    let hop_size = (fft_window_size / SPECTROGRAM_FFT_OVERLAP_FACTOR).max(1); 

    let mut overall_max_psd_val = 0.0f32;

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
                        let power = spectrum_complex[freq_idx].norm();
                        averaged_psd_for_bin[freq_idx] += power; 
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
                    let avg_power_for_bin_freq = averaged_psd_for_bin[freq_idx] / num_segments_averaged as f32;
                    psd_matrix[[freq_idx, bin_idx]] = avg_power_for_bin_freq;
                    if avg_power_for_bin_freq > overall_max_psd_val {
                        overall_max_psd_val = avg_power_for_bin_freq;
                    }
                }
            }
        }
    }

    let freq_bins = fft_rfftfreq(fft_window_size, 1.0 / sample_rate as f32);
    let throttle_bin_centers = Array1::from_shape_fn(num_throttle_bins, |i| {
        (i as f32 + 0.5) * throttle_bin_width
    });
    
    let auto_ranged_clip_max = overall_max_psd_val * AUTO_CLIP_MAX_SCALE_FACTOR;

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- PSD Matrix Diagnostic (fft_utils.rs) ---")?;
        writeln!(file, "FFT Window Time (ms): {}, Calculated FFT Window Size (samples): {}", fft_window_time_ms, fft_window_size)?;
        writeln!(file, "Hop Size (samples): {}, Overlap Factor: {}", hop_size, SPECTROGRAM_FFT_OVERLAP_FACTOR)?;
        writeln!(file, "Dimensions: {} freq_bins x {} throttle_bins", psd_matrix.shape()[0], psd_matrix.shape()[1])?;
        
        let mut min_nz_psd_val = f32::MAX; 
        let mut sum_psd_val = 0.0f32;
        let mut count_nz = 0;
        psd_matrix.iter().for_each(|&val| {
            if val > 1e-9 {
                if val < min_nz_psd_val { min_nz_psd_val = val; }
                sum_psd_val += val;
                count_nz +=1;
            }
        });

        writeln!(file, "Max Calculated PSD value (overall_max_psd_val): {}", overall_max_psd_val)?;
        writeln!(file, "Auto-Ranged Clip Max (for plotting, before log): {}", auto_ranged_clip_max)?;
        if count_nz > 0 {
            writeln!(file, "Min Non-Zero PSD value in Matrix: {}", min_nz_psd_val)?;
            writeln!(file, "Avg Non-Zero PSD value in Matrix: {}", sum_psd_val / count_nz as f32)?;
        } else {
            writeln!(file, "PSD Matrix effectively empty or all zeros.")?;
        }
        writeln!(file, "Constant MIN_POWER_FOR_LOG_SCALE: {}", MIN_POWER_FOR_LOG_SCALE)?;
        writeln!(file, "Constant AUTO_CLIP_MAX_SCALE_FACTOR: {}", AUTO_CLIP_MAX_SCALE_FACTOR)?;
        writeln!(file, "------------------------------------------")?;
    }

    Ok((psd_matrix, freq_bins, throttle_bin_centers, auto_ranged_clip_max))
}

// src/fft_utils.rs
