// src/step_response.rs

use ndarray::{Array1, Array2, s};
use std::collections::VecDeque;
use std::error::Error;

use crate::constants::{
    FRAME_LENGTH_S, RESPONSE_LENGTH_S, SUPERPOSITION_FACTOR, TUKEY_ALPHA,
    STEADY_STATE_START_S, STEADY_STATE_END_S, STEADY_STATE_MIN_VAL, STEADY_STATE_MAX_VAL
};

use crate::fft_utils; // Import the new FFT utility module


/// Makes a Tukey window for enveloping.
fn tukeywin(num: usize, alpha: f64) -> Array1<f32> {
    if alpha <= 0.0 {
        return Array1::ones(num); // rectangular window
    } else if alpha >= 1.0 {
        // Hanning window is a special case of Tukey with alpha=1.0
        let mut window = Array1::<f32>::zeros(num);
        for i in 0..num {
            window[i] = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (num as f64 - 1.0)).cos()) as f32;
        }
        return window;
    }

    let mut window = Array1::<f32>::ones(num);
    let alpha_half = alpha / 2.0;
    let n_alpha = (alpha_half * (num as f64 - 1.0)).floor() as usize;

    for i in 0..n_alpha {
        window[i] = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / (n_alpha as f64)).cos()) as f32;
        window[num - 1 - i] = window[i];
    }
    window
}


/// Generates overlapping windows from input data.
/// Returns a tuple of stacked input and output windows.
fn winstacker(
    input_data: &Array1<f32>,
    output_data: &Array1<f32>,
    frame_length_samples: usize,
    superposition_factor: usize,
) -> (Array2<f32>, Array2<f32>) {
    let total_len = input_data.len();
    if total_len == 0 || frame_length_samples == 0 || superposition_factor == 0 {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let shift = frame_length_samples / superposition_factor;
    if shift == 0 {
         eprintln!("Warning: Window shift is zero. Adjust frame_length_samples or superposition_factor.");
         return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let num_windows = if total_len >= frame_length_samples {
        (total_len - frame_length_samples) / shift + 1
    } else {
        0
    };

    if num_windows == 0 {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let mut stacked_input = Array2::<f32>::zeros((num_windows, frame_length_samples));
    let mut stacked_output = Array2::<f32>::zeros((num_windows, frame_length_samples));

    for i in 0..num_windows {
        let start = i * shift;
        let end = start + frame_length_samples;
        if end <= total_len {
            stacked_input.row_mut(i).assign(&input_data.slice(s![start..end]));
            stacked_output.row_mut(i).assign(&output_data.slice(s![start..end]));
        } else {
             eprintln!("Warning: Window end index out of bounds. Window {} from {} to {} out of {}", i, start, end, total_len);
        }
    }

    (stacked_input, stacked_output)
}

/// Performs Wiener deconvolution on a single input/output window.
/// Returns the deconvolved impulse response.
fn wiener_deconvolution_window(
    input_window: &Array1<f32>,
    output_window: &Array1<f32>,
    sample_rate: f64,
) -> Array1<f32> {
    let n = input_window.len();
    if n == 0 || sample_rate <= 0.0 {
        return Array1::zeros(n);
    }

    // Pad to next power of 2 for efficient FFT
    let padded_n = n.next_power_of_two();
    let input_padded = {
        let mut padded = Array1::<f32>::zeros(padded_n);
        padded.slice_mut(s![0..n]).assign(input_window);
        padded
    };
    // Corrected assignment:
    let mut output_padded = Array1::<f32>::zeros(padded_n);
    output_padded.slice_mut(s![0..n]).assign(output_window);


    let h_spec = fft_utils::fft_forward(&input_padded); // Use fft_utils
    let g_spec = fft_utils::fft_forward(&output_padded); // Use fft_utils

    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() {
         eprintln!("Warning: FFT output empty or length mismatch in Wiener deconvolution.");
         return Array1::zeros(n);
    }

    // Use a small constant regularization term (from PTstepcalc.m 0.0001)
    let regularization_term = 0.0001;
    let epsilon = 1e-9; // Small value to prevent division by zero

    let mut deconvolved_spec = Array1::<num_complex::Complex32>::zeros(h_spec.len()); // Use num_complex::Complex32 explicitly

    for i in 0..h_spec.len() {
        let h = h_spec[i];
        let g = g_spec[i];
        let h_conj = h.conj();

        // Wiener filter formula: G * H* / (|H|^2 + regularization_term)
        let denominator = (h * h_conj).re + regularization_term;

        if denominator.abs() > epsilon {
            deconvolved_spec[i] = (g * h_conj) / denominator;
        } else {
            deconvolved_spec[i] = num_complex::Complex32::new(0.0, 0.0); // Use num_complex::Complex32 explicitly
        }
    }

    // Perform inverse FFT and truncate to original length
    let deconvolved_impulse = fft_utils::fft_inverse(&deconvolved_spec, padded_n); // Use fft_utils

    // Return the impulse response truncated to the original window length
    deconvolved_impulse.slice(s![0..n]).to_owned()
}


/// Calculates the cumulative sum of an array to get the step response from impulse response.
fn cumulative_sum(data: &Array1<f32>) -> Array1<f32> {
    let mut cumulative = Array1::<f32>::zeros(data.len());
    let mut current_sum = 0.0;
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() {
            current_sum += val;
        } else {
             eprintln!("Warning: Non-finite value ({}) detected in impulse response at index {}. Skipping.", val, i);
        }
        cumulative[i] = current_sum;
    }
    cumulative
}

/// Applies a moving average filter to smooth a 1D array of f64.
pub fn moving_average_smooth_f64(data: &Array1<f64>, window_size: usize) -> Array1<f64> {
    if window_size <= 1 || data.is_empty() {
        return data.to_owned(); // No smoothing needed or possible.
    }

    let mut smoothed_data = Array1::<f64>::zeros(data.len());
    let mut current_sum: f64 = 0.0;
    let mut history: VecDeque<f64> = VecDeque::with_capacity(window_size);

    for i in 0..data.len() {
        let val = data[i];
        history.push_back(val);
        current_sum += val;

        if history.len() > window_size {
            if let Some(old_val) = history.pop_front() {
                current_sum -= old_val;
            }
        }

        let current_window_len = history.len() as f64;
        if current_window_len > 0.0 {
            smoothed_data[i] = current_sum / current_window_len;
        } else {
            smoothed_data[i] = 0.0;
        }
    }
    smoothed_data
}


/// Calculates the mean of the step responses from the stacked windows,
/// considering only windows with a weight > 0.
/// Returns the averaged step response curve.
/// This replaces the histogram-based mode averaging with a simpler mean.
pub fn average_responses( // Renamed from weighted_mode_avr
    stacked_responses: &Array2<f32>, // Stack of step responses (windows x time)
    weights: &Array1<f32>, // Weight for each window (0.0 or 1.0 based on mask)
    response_len: usize, // Length of the response (number of time points)
) -> Result<Array1<f64>, Box<dyn Error>> { // Returns the averaged response curve (Array1<f64>)
    let num_windows = stacked_responses.shape()[0];

    if num_windows == 0 || response_len == 0 || weights.len() != num_windows || stacked_responses.shape()[1] != response_len {
        return Err("Input data mismatch for average_responses".into());
    }

    let mut averaged_response = Array1::<f64>::zeros(response_len);
    let mut active_window_counts = Array1::<f64>::zeros(response_len); // Count how many weighted windows contribute to each time point

    for i in 0..num_windows { // Iterate over windows
        let weight = weights[i] as f64;
        if weight <= 1e-9 { continue; } // Skip windows with near-zero weight

        for j in 0..response_len { // Iterate over time points in the response
            let response_value = stacked_responses[[i, j]] as f64;

            if response_value.is_finite() {
                 // Add the value to the sum for this time point
                 averaged_response[j] += response_value * weight; // Apply weight (currently 0 or 1)
                 active_window_counts[j] += weight; // Count active windows (currently 0 or 1)
            }
        }
    }

    // Divide the sum by the count of active windows for each time point to get the mean
    for j in 0..response_len {
        if active_window_counts[j] > 1e-9 { // Avoid division by zero
            averaged_response[j] /= active_window_counts[j];
        } else {
            // If no active windows contributed to this time point, leave it as 0.0 (or handle differently if needed)
        }
    }

    Ok(averaged_response)
}


/// Calculates the system's step response using windowing and deconvolution,
/// returning the stacked, quality-controlled step responses and their corresponding
/// max setpoints for later averaging.
/// Input: Raw time, setpoint and gyro data, sample rate.
/// Output: Tuple of (response_time, stacked_qc_responses, qc_window_max_setpoints) or an error.
/// Note: The responses returned here are UN-NORMALIZED. Averaging and final normalization happen in main.rs.
pub fn calculate_step_response_python_style(
    time: &Array1<f64>, // Need time for relative time and windowing
    setpoint: &Array1<f32>,
    gyro_filtered: &Array1<f32>,
    sample_rate: f64,
) -> Result<(Array1<f64>, Array2<f32>, Array1<f32>), Box<dyn Error>> { // Updated return type
    if time.is_empty() || setpoint.is_empty() || gyro_filtered.is_empty() || setpoint.len() != gyro_filtered.len() || time.len() != setpoint.len() || sample_rate <= 0.0 {
        return Err("Invalid input to calculate_step_response_python_style: Empty data, length mismatch, or invalid sample rate.".into());
    }

    // Calculate window lengths in samples.
    let frame_length_samples = (FRAME_LENGTH_S * sample_rate).ceil() as usize;
    let response_length_samples = (RESPONSE_LENGTH_S * sample_rate).ceil() as usize;

    if frame_length_samples == 0 || response_length_samples == 0 {
         return Err("Calculated window length is zero. Adjust constants or sample rate.".into());
    }

    // 1. Generate overlapping windows.
    let (stacked_input, stacked_output) = winstacker(
        setpoint,
        gyro_filtered,
        frame_length_samples,
        SUPERPOSITION_FACTOR,
    );

    let num_windows = stacked_input.shape()[0];
    if num_windows == 0 {
        return Err("No complete windows could be generated.".into());
    }

    // Apply window function (e.g., Hanning) to each window.
    let window_func = tukeywin(frame_length_samples, TUKEY_ALPHA);
    let stacked_input_windowed = &stacked_input * &window_func;
    let stacked_output_windowed = &stacked_output * &window_func;


    // Prepare storage for step responses from each window that pass quality control.
    let mut stacked_step_responses_qc: Vec<Array1<f32>> = Vec::new();
    let mut window_max_setpoints_qc: Vec<f32> = Vec::new(); // Store max setpoint for QC windows

    // Calculate steady-state window indices for quality control
    let ss_start_sample = (STEADY_STATE_START_S * sample_rate).floor() as usize;
    let ss_end_sample = (STEADY_STATE_END_S * sample_rate).ceil() as usize;
    let ss_start_sample = ss_start_sample.min(response_length_samples);
    let ss_end_sample = ss_end_sample.min(response_length_samples);

    if ss_start_sample >= ss_end_sample {
         eprintln!("Warning: Steady-state window for quality control is invalid (start >= end). Quality control will be skipped.");
    }


    // 2. Perform Wiener deconvolution and cumulative sum for each window, apply QC.
    for i in 0..num_windows {
        let input_window = stacked_input_windowed.row(i).to_owned();
        let output_window = stacked_output_windowed.row(i).to_owned();

        // Calculate max setpoint in the original input window for masking later.
        let original_input_window = stacked_input.row(i).to_owned();
        let max_setpoint_in_window = original_input_window.iter().fold(0.0f32, |max_val, &v| max_val.max(v.abs()));


        let impulse_response = wiener_deconvolution_window(
            &input_window,
            &output_window,
            sample_rate,
        );

        // Truncate impulse response to frame length before cumulative sum
        let impulse_response_truncated = impulse_response.slice(s![0..frame_length_samples]).to_owned();

        if impulse_response_truncated.is_empty() || impulse_response_truncated.len() != frame_length_samples {
             eprintln!("Warning: Truncated impulse response length mismatch for window {}. Skipping.", i);
             continue;
        }

        let step_response = cumulative_sum(&impulse_response_truncated);

        // Truncate the step response to the desired response length *before* QC check
        if step_response.len() < response_length_samples {
             eprintln!("Warning: Step response shorter than response_length_samples for window {}. Skipping.", i);
             continue;
        }
        let truncated_step_response = step_response.slice(s![0..response_length_samples]).to_owned();


        // --- Individual Response Quality Control (based on steady-state value range) ---
        let mut passes_qc = false;
        if ss_start_sample < ss_end_sample { // Only perform QC if steady-state window is valid
            let steady_state_segment = truncated_step_response.slice(s![ss_start_sample..ss_end_sample]);

            // Use .mean() from ndarray
            if let Some(steady_state_mean) = steady_state_segment.mean() {
                 // Check if the UN-NORMALIZED steady-state mean is within the acceptable range
                 if steady_state_mean.is_finite() && steady_state_mean >= STEADY_STATE_MIN_VAL as f32 && steady_state_mean <= STEADY_STATE_MAX_VAL as f32 {
                     passes_qc = true; // Window passes QC based on steady-state value range
                 } else {
                     // Optionally print why a window failed QC
                     // eprintln!("Debug: Window {} failed QC. Steady-state mean: {:.2}", i, steady_state_mean.unwrap_or(f32::NAN));
                 }
            } else {
                 eprintln!("Warning: Could not calculate steady-state mean for window {}. Skipping QC.", i);
            }
        } else {
             // If steady-state window is invalid, maybe skip QC or pass all? Let's skip QC for now.
             // passes_qc = true; // Or decide to pass all if QC window is invalid
        }


        if passes_qc {
             // Push the UN-NORMALIZED truncated response
             stacked_step_responses_qc.push(truncated_step_response);
             window_max_setpoints_qc.push(max_setpoint_in_window);
        }
    }

    let num_qc_windows = stacked_step_responses_qc.len();
    if num_qc_windows == 0 {
        return Err("No windows passed filtering and quality control.".into());
    }

    // Convert Vec<Array1> to Array2 for averaging
    let valid_stacked_responses = Array2::from_shape_fn((num_qc_windows, response_length_samples), |(i, j)| {
        stacked_step_responses_qc[i][j]
    });
    let valid_window_max_setpoints = Array1::from(window_max_setpoints_qc);

    // Time vector for the response plot (0 to RESPONSE_LENGTH_S).
    let response_time = Array1::linspace(0.0, RESPONSE_LENGTH_S, response_length_samples);


    // Return the time vector, stacked QC responses, and max setpoints
    Ok((response_time.mapv(|t| t as f64), valid_stacked_responses, valid_window_max_setpoints)) // Updated return tuple
}