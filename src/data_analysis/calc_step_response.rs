// src/data_analysis/calc_step_response.rs

use ndarray::{Array1, Array2, s};
use std::collections::VecDeque;
use std::error::Error;

use crate::constants::{
    FRAME_LENGTH_S, RESPONSE_LENGTH_S, SUPERPOSITION_FACTOR, TUKEY_ALPHA,
    INITIAL_GYRO_SMOOTHING_WINDOW,
    APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION,
    Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS,
    NORMALIZED_STEADY_STATE_MIN_VAL,
    NORMALIZED_STEADY_STATE_MAX_VAL,
    ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK,
    NORMALIZED_STEADY_STATE_MEAN_MIN,
    NORMALIZED_STEADY_STATE_MEAN_MAX,
    STEADY_STATE_START_S, STEADY_STATE_END_S,
    MOVEMENT_THRESHOLD_DEG_S,
};

use crate::data_analysis::fft_utils;

// tukeywin, winstacker_contiguous, wiener_deconvolution_window, cumulative_sum,
// moving_average_smooth_f32, moving_average_smooth_f64, average_responses
// are assumed to be the same as in the previous full file output provided.
// For brevity, only calculate_step_response is shown with changes.

/// Creates a Tukey window for signal tapering with configurable alpha parameter.
pub fn tukeywin(num: usize, alpha: f64) -> Array1<f32> {
    if alpha <= 0.0 { return Array1::ones(num); }
    else if alpha >= 1.0 {
        // Full Hanning window when alpha = 1.0
        let mut window = Array1::<f32>::zeros(num);
        for i in 0..num { window[i] = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (num as f64 - 1.0)).cos()) as f32; }
        return window;
    }
    let mut window = Array1::<f32>::ones(num);
    let alpha_half = alpha / 2.0; let n_alpha = (alpha_half * (num as f64 - 1.0)).floor() as usize;
    // Apply cosine tapering to both ends
    for i in 0..n_alpha {
        window[i] = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / (n_alpha as f64)).cos()) as f32;
        window[num - 1 - i] = window[i];
    }
    window
}

/// Generates overlapping windowed segments from input/output data for parallel analysis.
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

/// Performs Wiener deconvolution to extract system impulse response from input/output signals.
fn wiener_deconvolution_window( input_window: &Array1<f32>, output_window: &Array1<f32>, _sample_rate: f64) -> Array1<f32> {
    let n = input_window.len(); if n == 0 { return Array1::zeros(n); }
    let padded_n = n.next_power_of_two();
    let mut input_padded_vec = vec![0.0f32; padded_n]; if n > 0 {input_padded_vec[0..n].copy_from_slice(input_window.as_slice().unwrap_or_default());}
    let input_padded = Array1::from(input_padded_vec);
    let mut output_padded_vec = vec![0.0f32; padded_n]; if n > 0 {output_padded_vec[0..n].copy_from_slice(output_window.as_slice().unwrap_or_default());}
    let output_padded = Array1::from(output_padded_vec);
    let h_spec = fft_utils::fft_forward(&input_padded); let g_spec = fft_utils::fft_forward(&output_padded);
    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() { eprintln!("Warning: FFT output empty/mismatch in Wiener deconvolution."); return Array1::zeros(n); }
    // Apply regularization to prevent division by near-zero values
    let regularization_term = 0.0001; let epsilon = 1e-9;
    let mut deconvolved_spec = Array1::<num_complex::Complex32>::zeros(h_spec.len());
    for i in 0..h_spec.len() {
        let h = h_spec[i]; let g = g_spec[i]; let h_conj = h.conj();
        let denominator = (h * h_conj).re + regularization_term;
        if denominator.abs() > epsilon { deconvolved_spec[i] = (g * h_conj) / denominator; }
        else { deconvolved_spec[i] = num_complex::Complex32::new(0.0, 0.0); }
    }
    let deconvolved_impulse = fft_utils::fft_inverse(&deconvolved_spec, padded_n);
    if n > 0 { deconvolved_impulse.slice(s![0..n]).to_owned() } else { Array1::zeros(0) }
}

/// Converts impulse response to step response via cumulative integration.
fn cumulative_sum(data: &Array1<f32>) -> Array1<f32> {
    let mut cumulative = Array1::<f32>::zeros(data.len()); let mut current_sum = 0.0;
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() { current_sum += val; } else { eprintln!("Warning: Non-finite impulse value ({}) at index {}.", val, i); }
        cumulative[i] = current_sum;
    }
    cumulative
}

/// Applies a moving average filter to smooth a 1D array of f32. (Same as before)
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

/// Applies a moving average filter to smooth a 1D array of f64. (Same as before)
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

/// Computes weighted average of multiple step responses for final result.
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


/// Main function to calculate step response from setpoint and gyro data using windowed Wiener deconvolution.
/// Returns time array, stacked responses matrix, and max setpoints for each valid window.
pub fn calculate_step_response(
    _time: &Array1<f64>,
    setpoint: &Array1<f32>,
    gyro_filtered_input: &Array1<f32>,
    sample_rate: f64,
) -> Result<(Array1<f64>, Array2<f32>, Array1<f32>), Box<dyn Error>> {
    if setpoint.is_empty() || gyro_filtered_input.is_empty() || setpoint.len() != gyro_filtered_input.len() || sample_rate <= 0.0 {
        return Err("Invalid input to calculate_step_response".into());
    }

    // Apply initial smoothing to gyro data if configured
    let gyro_processed = if INITIAL_GYRO_SMOOTHING_WINDOW > 1 {
        moving_average_smooth_f32(gyro_filtered_input, INITIAL_GYRO_SMOOTHING_WINDOW)
    } else {
        gyro_filtered_input.to_owned()
    };

    let frame_length_samples = (FRAME_LENGTH_S * sample_rate).ceil() as usize;
    let response_length_samples = (RESPONSE_LENGTH_S * sample_rate).ceil() as usize;

    if frame_length_samples == 0 || response_length_samples == 0 {
         return Err("Calculated window length is zero.".into());
    }

    // Define steady-state region for quality control checks
    let ss_start_sample = (STEADY_STATE_START_S * sample_rate).floor() as usize;
    let ss_end_sample = (STEADY_STATE_END_S * sample_rate).ceil() as usize;
    let effective_ss_start_sample = ss_start_sample.min(response_length_samples.saturating_sub(1));
    let effective_ss_end_sample = ss_end_sample.min(response_length_samples).max(effective_ss_start_sample + 1);

    let mut stacked_step_responses_qc: Vec<Array1<f32>> = Vec::new();
    let mut window_max_setpoints_qc: Vec<f32> = Vec::new();

    // Generate overlapping windows for parallel processing
    let (stacked_input_raw, stacked_output_raw) = winstacker_contiguous(
        setpoint, &gyro_processed, frame_length_samples, SUPERPOSITION_FACTOR);

    let num_windows = stacked_input_raw.shape()[0];
    if num_windows == 0 { return Err("No complete windows generated.".into()); }

    let window_func = tukeywin(frame_length_samples, TUKEY_ALPHA);

    // Process each window through deconvolution and quality control
    for i in 0..num_windows {
        let input_window_raw = stacked_input_raw.row(i).to_owned();
        let output_window_raw = stacked_output_raw.row(i).to_owned();

        // Filter out low-activity windows based on setpoint magnitude
        let max_setpoint_in_window = input_window_raw.iter().fold(0.0f32, |max_val, &v| max_val.max(v.abs()));
        if max_setpoint_in_window < MOVEMENT_THRESHOLD_DEG_S as f32 {
             continue;
        }

        // Apply windowing function to reduce spectral leakage
        let input_window_windowed = &input_window_raw * &window_func;
        let output_window_windowed = &output_window_raw * &window_func;

        // Extract impulse response via Wiener deconvolution
        let impulse_response = wiener_deconvolution_window(&input_window_windowed, &output_window_windowed, sample_rate);
        let impulse_response_truncated = if impulse_response.len() >= frame_length_samples {
            impulse_response.slice(s![0..frame_length_samples]).to_owned()
        } else {
            // eprintln!("Warning: Impulse response too short ({} vs {}) for window {}. Skipping.", impulse_response.len(), frame_length_samples, i);
            continue;
        };
        if impulse_response_truncated.is_empty() { continue; }

        // Convert impulse to step response and truncate to desired length
        let unnormalized_step_response = cumulative_sum(&impulse_response_truncated);
        if unnormalized_step_response.len() < response_length_samples { continue; }
        let truncated_unnormalized_response = unnormalized_step_response.slice(s![0..response_length_samples]).to_owned();

        let mut response_for_qc = truncated_unnormalized_response.clone();
        let mut y_correction_attempted_and_valid_for_qc = false;

        // Apply Y-correction (normalization) if enabled and conditions are met
        if APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION {
            if effective_ss_start_sample < effective_ss_end_sample && truncated_unnormalized_response.len() >= effective_ss_end_sample {
                let unnorm_ss_segment = truncated_unnormalized_response.slice(s![effective_ss_start_sample..effective_ss_end_sample]);
                if let Some(unnorm_ss_mean) = unnorm_ss_segment.mean() {
                    if unnorm_ss_mean.is_finite() && unnorm_ss_mean.abs() > Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS {
                        // ** MODIFIED: Standard Normalization for Y-Correction **
                        // Normalize response to target steady-state of 1.0
                        response_for_qc = truncated_unnormalized_response.mapv(|v| v / unnorm_ss_mean);
                        y_correction_attempted_and_valid_for_qc = true;
                    } else {
                        // Unnormalized mean is too small or not finite, Y-correction cannot be reliably applied.
                        // Window will proceed to QC with its unnormalized response if APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION is false,
                        // or be effectively skipped for Y-corrected QC if APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION is true
                        // (as y_correction_attempted_and_valid_for_qc remains false).
                    }
                }
            }
        } else {
            // If Y-correction is not applied, QC will run on the unnormalized response.
            // The NORMALIZED_STEADY_STATE constants might not be appropriate in this case.
            // This path assumes QC constants are set considering unnormalized data if Y-correction is off.
            y_correction_attempted_and_valid_for_qc = true; // Allow QC to proceed on unnormalized data
        }

        // --- Quality Control ---
        // If Y-correction is enabled, QC should only run if Y-correction was successfully applied.
        // If Y-correction is disabled, QC runs on the unnormalized data.
        // Quality control: check steady-state values against thresholds
        let mut current_window_passes_qc = false;
        if (APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION && y_correction_attempted_and_valid_for_qc) || !APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION {
            if effective_ss_start_sample < effective_ss_end_sample && response_for_qc.len() >= effective_ss_end_sample {
                let qc_ss_segment = response_for_qc.slice(s![effective_ss_start_sample..effective_ss_end_sample]);
                if !qc_ss_segment.is_empty() {
                    let min_val_ss = qc_ss_segment.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max_val_ss = qc_ss_segment.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mean_val_ss_opt = qc_ss_segment.mean();

                    // Check min/max bounds for steady-state region
                    if min_val_ss.is_finite() && max_val_ss.is_finite() &&
                       min_val_ss > NORMALIZED_STEADY_STATE_MIN_VAL && max_val_ss < NORMALIZED_STEADY_STATE_MAX_VAL {
                        // Optional additional check on steady-state mean
                        if ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK {
                            if let Some(mean_val_ss) = mean_val_ss_opt {
                                if mean_val_ss.is_finite() &&
                                   mean_val_ss > NORMALIZED_STEADY_STATE_MEAN_MIN &&
                                   mean_val_ss < NORMALIZED_STEADY_STATE_MEAN_MAX {
                                    current_window_passes_qc = true;
                                }
                            }
                        } else {
                            current_window_passes_qc = true;
                        }
                    }
                }
            }
        }


        // Collect windows that pass quality control
        if current_window_passes_qc {
             stacked_step_responses_qc.push(response_for_qc);
             window_max_setpoints_qc.push(max_setpoint_in_window);
        }
    }

    if stacked_step_responses_qc.is_empty() {
        return Err("No windows passed quality control.".into());
    }

    let rows = stacked_step_responses_qc.len();
    // Assuming all inner vectors should have response_length_samples.
    // If not, this needs to be the maximum length or handled differently.
    let cols = response_length_samples; 

    // Convert validated responses to Array2 format for output
    let valid_stacked_responses = Array2::from_shape_fn((rows, cols), |(r, c)| {
        stacked_step_responses_qc.get(r)
            .and_then(|row| row.get(c))
            .copied()
            .unwrap_or_else(|| {
                // This block will execute if either `r` or `c` is out of bounds.
                // eprintln!(
                //     "Warning: Accessing out of bounds in stacked_step_responses_qc at ({}, {}). Max dims ({}, {}). Using 0.0.",
                //     r, c, rows, cols 
                // );
                0.0 // Default value if out of bounds
            })
    });
    let valid_window_max_setpoints = Array1::from(window_max_setpoints_qc);
    let response_time = Array1::linspace(0.0, RESPONSE_LENGTH_S, response_length_samples);

    Ok((response_time.mapv(|t| t as f64), valid_stacked_responses, valid_window_max_setpoints))
}

/// Calculates Delay Time (Td) - the time to reach 50% of the final value.
/// Uses linear interpolation for precise threshold crossing.
/// Retains PTB's convention of subtracting 1ms from the calculated time in milliseconds.
///
/// # Arguments
/// * `normalized_response` - A 1D array of the normalized step response (expected to go from ~0 to ~1).
/// * `sample_rate` - The sample rate of the response data in Hz.
///
/// # Returns
/// `Some(f64)` containing the delay time in seconds, or `None` if the response is empty,
/// sample rate is invalid, or the 50% threshold is not crossed.
pub fn calculate_delay_time(normalized_response: &Array1<f64>, sample_rate: f64) -> Option<f64> {
    if normalized_response.is_empty() || sample_rate <= 0.0 {
        return None;
    }

    let threshold = 0.5;
    let mut interpolated_idx_0_based: Option<f64> = None;

    // Check if the first point is at or above the threshold
    if normalized_response[0] >= threshold {
        interpolated_idx_0_based = Some(0.0);
    } else {
        for idx in 1..normalized_response.len() {
            let y0 = normalized_response[idx - 1];
            let y1 = normalized_response[idx];

            if y0 < threshold && y1 >= threshold {
                // Found a crossing
                let x0 = (idx - 1) as f64; // 0-based index of point before crossing
                let x1 = idx as f64;       // 0-based index of point at or after crossing

                if (y1 - y0).abs() < 1e-9 { // Avoid division by zero if y1 is very close to y0
                                            // This implies the segment is flat and at/crossing the threshold.
                                            // Take the first point of this segment that is >= threshold.
                    interpolated_idx_0_based = Some(x1);
                } else {
                    // Linear interpolation: x = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
                    let t = (threshold - y0) / (y1 - y0);
                    interpolated_idx_0_based = Some(x0 + t * (x1 - x0));
                }
                break; // Found the first crossing
            }
        }
    }

    if let Some(idx_val) = interpolated_idx_0_based {
        // The PTB formula `(find(m>.5,1) / A_lograte(f)) - 1` implies:
        // 1. `find(m>.5,1)` gives a 1-based sample index.
        //    Our `idx_val` is a 0-based (potentially fractional) index.
        //    So, `idx_val + 1.0` corresponds to the 1-based sample number.
        let time_samples_1_based = idx_val + 1.0;
        
        // 2. `A_lograte(f)` is sample_rate / 1000 (samples per millisecond).
        let sample_rate_per_ms = sample_rate / 1000.0;
        if sample_rate_per_ms <= 1e-9 { // Avoid division by zero or issues with very low sample rates
            return None;
        }
        
        // 3. Calculate time in milliseconds.
        let time_ms = time_samples_1_based / sample_rate_per_ms;
        
        // 4. Subtract 1ms offset as per PTB convention.
        let delay_ms = time_ms - 1.0;
        
        // 5. Convert to seconds and ensure non-negative.
        Some((delay_ms / 1000.0).max(0.0))
    } else {
        None // Threshold not crossed
    }
}

/// Finds the peak (maximum) value in a 1D response data array.
/// Returns None if the array is empty or contains no finite values.
pub fn find_peak_value(response_data: &Array1<f64>) -> Option<f64> {
    if response_data.is_empty() {
        return None;
    }
    let mut peak_val = f64::NEG_INFINITY;
    let mut found_finite = false;
    for &val in response_data.iter() {
        if val.is_finite() {
            if !found_finite || val > peak_val {
                peak_val = val;
            }
            found_finite = true;
        }
    }
    if found_finite { Some(peak_val) } else { None }
}

// src/data_analysis/calc_step_response.rs
