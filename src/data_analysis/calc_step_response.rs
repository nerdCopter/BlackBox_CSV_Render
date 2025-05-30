// src/data_analysis/calc_step_response.rs

use ndarray::{Array1, Array2, s};
use std::collections::VecDeque;
use std::error::Error;

use crate::constants::{
    FRAME_LENGTH_S, RESPONSE_LENGTH_S, SUPERPOSITION_FACTOR, TUKEY_ALPHA,
    INITIAL_GYRO_SMOOTHING_WINDOW, // New
    APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION, // New
    Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS, // New
    NORMALIZED_STEADY_STATE_MIN_VAL, // New or re-purposed
    NORMALIZED_STEADY_STATE_MAX_VAL, // New or re-purposed
    ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK, //New
    NORMALIZED_STEADY_STATE_MEAN_MIN, // New
    NORMALIZED_STEADY_STATE_MEAN_MAX, // New
    STEADY_STATE_START_S, STEADY_STATE_END_S,
    MOVEMENT_THRESHOLD_DEG_S,
};

use crate::data_analysis::fft_utils;

// tukeywin, winstacker_contiguous, wiener_deconvolution_window, cumulative_sum are assumed to be the same as before.
// moving_average_smooth_f32 will be a new helper for initial gyro smoothing.

/// Makes a Tukey window for enveloping. (Same as before)
pub fn tukeywin(num: usize, alpha: f64) -> Array1<f32> {
    if alpha <= 0.0 { return Array1::ones(num); }
    else if alpha >= 1.0 {
        let mut window = Array1::<f32>::zeros(num);
        for i in 0..num { window[i] = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (num as f64 - 1.0)).cos()) as f32; }
        return window;
    }
    let mut window = Array1::<f32>::ones(num);
    let alpha_half = alpha / 2.0; let n_alpha = (alpha_half * (num as f64 - 1.0)).floor() as usize;
    for i in 0..n_alpha {
        window[i] = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / (n_alpha as f64)).cos()) as f32;
        window[num - 1 - i] = window[i];
    }
    window
}

/// Generates overlapping windows. (Same as before)
fn winstacker_contiguous( input_data: &Array1<f32>, output_data: &Array1<f32>, frame_length_samples: usize, superposition_factor: usize) -> (Array2<f32>, Array2<f32>) {
    let total_len = input_data.len();
    if total_len == 0 || frame_length_samples == 0 || superposition_factor == 0 { return (Array2::zeros((0, 0)), Array2::zeros((0, 0))); }
    let shift = frame_length_samples / superposition_factor;
    if shift == 0 { eprintln!("Warning: Window shift is zero."); return (Array2::zeros((0, 0)), Array2::zeros((0, 0))); }
    let num_windows = if total_len >= frame_length_samples { (total_len - frame_length_samples) / shift + 1 } else { 0 };
    if num_windows == 0 { return (Array2::zeros((0, 0)), Array2::zeros((0, 0))); }
    let mut stacked_input = Array2::<f32>::zeros((num_windows, frame_length_samples));
    let mut stacked_output = Array2::<f32>::zeros((num_windows, frame_length_samples));
    for i in 0..num_windows {
        let start = i * shift; let end = start + frame_length_samples;
        stacked_input.row_mut(i).assign(&input_data.slice(s![start..end]));
        stacked_output.row_mut(i).assign(&output_data.slice(s![start..end]));
    }
    (stacked_input, stacked_output)
}

/// Wiener deconvolution. (Same as before)
fn wiener_deconvolution_window( input_window: &Array1<f32>, output_window: &Array1<f32>, _sample_rate: f64) -> Array1<f32> {
    let n = input_window.len(); if n == 0 { return Array1::zeros(n); }
    let padded_n = n.next_power_of_two();
    let mut input_padded_vec = vec![0.0f32; padded_n]; input_padded_vec[0..n].copy_from_slice(input_window.as_slice().unwrap());
    let input_padded = Array1::from(input_padded_vec);
    let mut output_padded_vec = vec![0.0f32; padded_n]; output_padded_vec[0..n].copy_from_slice(output_window.as_slice().unwrap());
    let output_padded = Array1::from(output_padded_vec);
    let h_spec = fft_utils::fft_forward(&input_padded); let g_spec = fft_utils::fft_forward(&output_padded);
    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() { eprintln!("Warning: FFT output empty/mismatch in Wiener deconvolution."); return Array1::zeros(n); }
    let regularization_term = 0.0001; let epsilon = 1e-9;
    let mut deconvolved_spec = Array1::<num_complex::Complex32>::zeros(h_spec.len());
    for i in 0..h_spec.len() {
        let h = h_spec[i]; let g = g_spec[i]; let h_conj = h.conj();
        let denominator = (h * h_conj).re + regularization_term;
        if denominator.abs() > epsilon { deconvolved_spec[i] = (g * h_conj) / denominator; }
        else { deconvolved_spec[i] = num_complex::Complex32::new(0.0, 0.0); }
    }
    let deconvolved_impulse = fft_utils::fft_inverse(&deconvolved_spec, padded_n);
    deconvolved_impulse.slice(s![0..n]).to_owned()
}

/// Cumulative sum. (Same as before)
fn cumulative_sum(data: &Array1<f32>) -> Array1<f32> {
    let mut cumulative = Array1::<f32>::zeros(data.len()); let mut current_sum = 0.0;
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() { current_sum += val; } else { eprintln!("Warning: Non-finite impulse value ({}) at index {}.", val, i); }
        cumulative[i] = current_sum;
    }
    cumulative
}

/// Applies a moving average filter to smooth a 1D array of f32. (New helper, or use existing if it matches)
pub fn moving_average_smooth_f32(data: &Array1<f32>, window_size: usize) -> Array1<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_owned();
    }
    let mut smoothed_data = Array1::<f32>::zeros(data.len());
    let mut current_sum: f32 = 0.0;
    let mut history: VecDeque<f32> = VecDeque::with_capacity(window_size);
    for i in 0..data.len() {
        let val = data[i]; history.push_back(val); current_sum += val;
        if history.len() > window_size { if let Some(old_val) = history.pop_front() { current_sum -= old_val; }}
        let current_window_len = history.len() as f32;
        if current_window_len > 0.0 { smoothed_data[i] = current_sum / current_window_len; } else { smoothed_data[i] = 0.0; }
    }
    smoothed_data
}


/// Calculates the mean of the step responses (used by plotting, can remain f64 if preferred there)
pub fn average_responses( stacked_responses: &Array2<f32>, weights: &Array1<f32>, response_len: usize) -> Result<Array1<f64>, Box<dyn Error>> {
    let num_windows = stacked_responses.shape()[0];
    if num_windows == 0 || response_len == 0 || weights.len() != num_windows || stacked_responses.shape()[1] != response_len {
        return Err("Input data mismatch for average_responses".into());
    }
    let mut averaged_response = Array1::<f64>::zeros(response_len);
    let mut active_window_counts = Array1::<f64>::zeros(response_len);
    for i in 0..num_windows {
        let weight = weights[i] as f64; if weight <= 1e-9 { continue; }
        for j in 0..response_len {
            let response_value = stacked_responses[[i, j]] as f64;
            if response_value.is_finite() { averaged_response[j] += response_value * weight; active_window_counts[j] += weight; }
        }
    }
    for j in 0..response_len { if active_window_counts[j] > 1e-9 { averaged_response[j] /= active_window_counts[j]; } }
    Ok(averaged_response)
}

pub fn calculate_step_response(
    _time: &Array1<f64>,
    setpoint: &Array1<f32>,
    gyro_filtered_input: &Array1<f32>, // Renamed to avoid confusion
    sample_rate: f64,
) -> Result<(Array1<f64>, Array2<f32>, Array1<f32>), Box<dyn Error>> {
    if setpoint.is_empty() || gyro_filtered_input.is_empty() || setpoint.len() != gyro_filtered_input.len() || sample_rate <= 0.0 {
        return Err("Invalid input to calculate_step_response".into());
    }

    // Step 1: Optional Initial Gyro Smoothing
    let gyro_processed = if INITIAL_GYRO_SMOOTHING_WINDOW > 1 {
        moving_average_smooth_f32(gyro_filtered_input, INITIAL_GYRO_SMOOTHING_WINDOW)
    } else {
        gyro_filtered_input.to_owned() // Or just pass by reference if no copy needed
    };

    let frame_length_samples = (FRAME_LENGTH_S * sample_rate).ceil() as usize;
    let response_length_samples = (RESPONSE_LENGTH_S * sample_rate).ceil() as usize;

    if frame_length_samples == 0 || response_length_samples == 0 {
         return Err("Calculated window length is zero.".into());
    }

    let ss_start_sample = (STEADY_STATE_START_S * sample_rate).floor() as usize;
    let ss_end_sample = (STEADY_STATE_END_S * sample_rate).ceil() as usize;
    let effective_ss_start_sample = ss_start_sample.min(response_length_samples.saturating_sub(1)); // ensure it's a valid index
    let effective_ss_end_sample = ss_end_sample.min(response_length_samples).max(effective_ss_start_sample + 1); // ensure end > start and within bounds


    let mut stacked_step_responses_qc: Vec<Array1<f32>> = Vec::new();
    let mut window_max_setpoints_qc: Vec<f32> = Vec::new();

    let (stacked_input_raw, stacked_output_raw) = winstacker_contiguous(
        setpoint,
        &gyro_processed, // Use the (potentially) smoothed gyro data
        frame_length_samples,
        SUPERPOSITION_FACTOR,
    );

    let num_windows = stacked_input_raw.shape()[0];
    if num_windows == 0 { return Err("No complete windows generated.".into()); }

    let window_func = tukeywin(frame_length_samples, TUKEY_ALPHA);

    for i in 0..num_windows {
        let input_window_raw = stacked_input_raw.row(i).to_owned();
        let output_window_raw = stacked_output_raw.row(i).to_owned();

        let max_setpoint_in_window = input_window_raw.iter().fold(0.0f32, |max_val, &v| max_val.max(v.abs()));
        if max_setpoint_in_window < MOVEMENT_THRESHOLD_DEG_S as f32 { // Simplified check, PTB also checks gyro
             continue;
        }

        let input_window_windowed = &input_window_raw * &window_func;
        let output_window_windowed = &output_window_raw * &window_func;

        let impulse_response = wiener_deconvolution_window(&input_window_windowed, &output_window_windowed, sample_rate);
        let impulse_response_truncated = impulse_response.slice(s![0..frame_length_samples]).to_owned();
        if impulse_response_truncated.is_empty() { continue; }

        let unnormalized_step_response = cumulative_sum(&impulse_response_truncated);
        if unnormalized_step_response.len() < response_length_samples { continue; }
        let truncated_unnormalized_response = unnormalized_step_response.slice(s![0..response_length_samples]).to_owned();

        let mut response_for_qc = truncated_unnormalized_response.clone(); // Start with unnormalized
        let mut y_correction_applied_successfully = false;

        if APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION {
            if effective_ss_start_sample < effective_ss_end_sample && truncated_unnormalized_response.len() >= effective_ss_end_sample {
                let unnorm_ss_segment = truncated_unnormalized_response.slice(s![effective_ss_start_sample..effective_ss_end_sample]);
                if let Some(unnorm_ss_mean) = unnorm_ss_segment.mean() {
                    if unnorm_ss_mean.is_finite() && unnorm_ss_mean.abs() > Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS {
                        // PTB's specific scaling: val * (2 - mean)
                        let scaling_factor = 2.0 - unnorm_ss_mean;
                        response_for_qc = truncated_unnormalized_response.mapv(|v| v * scaling_factor);
                        y_correction_applied_successfully = true;
                    } else {
                        // eprintln!("Debug: Window {} Y-correction skipped: unnormalized mean {:.2} too small or not finite.", i, unnorm_ss_mean);
                    }
                }
            }
        }

        // --- Quality Control ---
        let mut current_window_passes_qc = false;
        if effective_ss_start_sample < effective_ss_end_sample && response_for_qc.len() >= effective_ss_end_sample {
            let qc_ss_segment = response_for_qc.slice(s![effective_ss_start_sample..effective_ss_end_sample]);
            if qc_ss_segment.is_empty() { // Should not happen if indices are correct
                 eprintln!("Warning: QC steady-state segment empty for window {}. Skipping QC.", i);
            } else {
                let min_val_ss = qc_ss_segment.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val_ss = qc_ss_segment.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mean_val_ss_opt = qc_ss_segment.mean();

                if min_val_ss.is_finite() && max_val_ss.is_finite() &&
                   min_val_ss > NORMALIZED_STEADY_STATE_MIN_VAL && max_val_ss < NORMALIZED_STEADY_STATE_MAX_VAL {
                    if ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK {
                        if let Some(mean_val_ss) = mean_val_ss_opt {
                            if mean_val_ss.is_finite() &&
                               mean_val_ss > NORMALIZED_STEADY_STATE_MEAN_MIN &&
                               mean_val_ss < NORMALIZED_STEADY_STATE_MEAN_MAX {
                                current_window_passes_qc = true;
                            } else {
                                // eprintln!("Debug: Window {} failed QC: Y-corrected mean {:.2} out of bounds ({:.2} - {:.2}).",
                                //    i, mean_val_ss.unwrap_or(f32::NAN), NORMALIZED_STEADY_STATE_MEAN_MIN, NORMALIZED_STEADY_STATE_MEAN_MAX);
                            }
                        } else { /* Mean calculation failed */ }
                    } else {
                        current_window_passes_qc = true; // Passed min/max, and mean check is disabled
                    }
                } else {
                     // eprintln!("Debug: Window {} failed QC: Y-corrected min/max ({:.2}/{:.2}) out of bounds ({:.2} - {:.2}). Applied Y-corr: {}",
                     //    i, min_val_ss, max_val_ss, NORMALIZED_STEADY_STATE_MIN_VAL, NORMALIZED_STEADY_STATE_MAX_VAL, y_correction_applied_successfully);
                }
            }
        } else { /* effective_ss window invalid or response too short */ }


        if current_window_passes_qc {
             stacked_step_responses_qc.push(response_for_qc); // This is the (potentially) Y-corrected response
             window_max_setpoints_qc.push(max_setpoint_in_window);
        }
    }

    if stacked_step_responses_qc.is_empty() {
        return Err("No windows passed quality control.".into());
    }

    let valid_stacked_responses = Array2::from_shape_fn((stacked_step_responses_qc.len(), response_length_samples), |(r, c)| {
        stacked_step_responses_qc[r][c]
    });
    let valid_window_max_setpoints = Array1::from(window_max_setpoints_qc);
    let response_time = Array1::linspace(0.0, RESPONSE_LENGTH_S, response_length_samples);

    Ok((response_time.mapv(|t| t as f64), valid_stacked_responses, valid_window_max_setpoints))
}


// moving_average_smooth_f64 (for plotting) would remain the same if it's in this file,
// or be used from where it's defined if `plot_step_response.rs` calls it.
// For brevity, not re-listing it if it's unchanged and used elsewhere.
pub fn moving_average_smooth_f64(data: &Array1<f64>, window_size: usize) -> Array1<f64> {
    if window_size <= 1 || data.is_empty() { return data.to_owned(); }
    let mut smoothed_data = Array1::<f64>::zeros(data.len());
    let mut current_sum: f64 = 0.0;
    let mut history: VecDeque<f64> = VecDeque::with_capacity(window_size);
    for i in 0..data.len() {
        let val = data[i]; history.push_back(val); current_sum += val;
        if history.len() > window_size { if let Some(old_val) = history.pop_front() { current_sum -= old_val; }}
        let current_window_len = history.len() as f64;
        if current_window_len > 0.0 { smoothed_data[i] = current_sum / current_window_len; } else { smoothed_data[i] = 0.0; }
    }
    smoothed_data
}

// src/data_analysis/calc_step_response.rs
