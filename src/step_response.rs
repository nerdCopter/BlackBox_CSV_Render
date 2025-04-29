// src/step_response.rs

use ndarray::{Array1, Array2, s};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::error::Error;

use crate::constants::*;

/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
/// Returns the complex frequency spectrum. Handles empty input.
fn fft_forward(data: &Array1<f32>) -> Array1<Complex32> {
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
fn fft_inverse(data: &Array1<Complex32>, original_length_n: usize) -> Array1<f32> {
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


    let h_spec = fft_forward(&input_padded);
    let g_spec = fft_forward(&output_padded);

    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() {
         eprintln!("Warning: FFT output empty or length mismatch in Wiener deconvolution.");
         return Array1::zeros(n);
    }

    // Use a small constant regularization term (from PTstepcalc.m 0.0001)
    let regularization_term = 0.0001;
    let epsilon = 1e-9; // Small value to prevent division by zero

    let mut deconvolved_spec = Array1::<Complex32>::zeros(h_spec.len());

    for i in 0..h_spec.len() {
        let h = h_spec[i];
        let g = g_spec[i];
        let h_conj = h.conj();

        // Wiener filter formula: G * H* / (|H|^2 + regularization_term)
        let denominator = (h * h_conj).re + regularization_term;

        if denominator.abs() > epsilon {
            deconvolved_spec[i] = (g * h_conj) / denominator;
        } else {
            deconvolved_spec[i] = Complex32::new(0.0, 0.0);
        }
    }

    // Perform inverse FFT and truncate to original length
    let deconvolved_impulse = fft_inverse(&deconvolved_spec, padded_n);

    // Return the impulse response truncated to the original window length
    deconvolved_impulse.slice(s![0..n]).to_owned()
}

/// Calculates the frequencies for the real FFT output. (Currently unused in deconvolution logic)
#[allow(dead_code)]
fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> {
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let mut freqs = Array1::<f32>::zeros(num_freqs);
    let nyquist = 0.5 / d;
    for i in 0..num_freqs {
        freqs[i] = i as f32 * nyquist / (num_freqs - 1) as f32;
    }
    freqs
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

    for (i, &val) in data.iter().enumerate() {
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


/// Struct to hold 2D histogram data and scales needed for mode averaging.
struct Hist2D {
    hist2d_norm: Array2<f64>, // Normalized histogram (used for finding mode)
    yscale: Array1<f64>, // Vertical scale (bin centers)
}

/// Creates a weighted 2D histogram from a stack of response values (y) against time (x, implicit indices).
/// Weights are applied to each window.
fn create_weighted_hist2d(
    stacked_responses: &Array2<f32>, // Stack of step responses (windows x time)
    weights: &Array1<f32>, // Weight for each window
    response_time: &Array1<f64>, // Time vector for the response (0 to RESPONSE_LENGTH_S)
    vertrange: [f64; 2], // [ymin, ymax] for the vertical axis (response value)
    vertbins: usize, // Number of vertical bins
) -> Result<Hist2D, Box<dyn Error>> {
    let num_windows = stacked_responses.shape()[0];
    let response_len = stacked_responses.shape()[1];

    if num_windows == 0 || response_len == 0 || weights.len() != num_windows || response_time.len() != response_len {
        return Err("Input data mismatch for create_weighted_hist2d".into());
    }

    let nx = response_len; // Number of horizontal bins equals the response length
    let ny = vertbins;
    let ymin = vertrange[0];
    let ymax = vertrange[1];

    let mut hist2d = Array2::<f64>::zeros((ny, nx)); // Note: histogram2d returns shape (ny, nx)
    let mut xhist = Array1::<f64>::zeros(nx); // This will count how many weighted values fall into each time bin

    let dy = (ymax - ymin) / ny as f64;

    for i in 0..num_windows { // Iterate over windows
        let weight = weights[i] as f64;
        if weight <= 1e-9 { continue; } // Skip windows with near-zero weight

        for j in 0..response_len { // Iterate over time points in the response
            let response_value = stacked_responses[[i, j]] as f64;

            // Check if the response value is within the vertical range
            if response_value >= ymin && response_value <= ymax {
                let y_bin = ((response_value - ymin) / dy).floor() as usize;
                let y_bin = y_bin.min(ny - 1); // Clamp to the last bin if exactly on the upper edge

                // The horizontal bin is simply the time index j
                let x_bin = j;

                hist2d[[y_bin, x_bin]] += weight; // Add the window's weight to the bin
                xhist[x_bin] += weight; // Add the window's weight to the total weight for this time bin
            }
        }
    }

    // Calculate normalized histogram (normalize each time slice by the total weight in that slice)
    let mut hist2d_norm = Array2::<f64>::zeros((ny, nx));
    for j in 0..nx { // Iterate over time bins (columns)
        if xhist[j] > 1e-9 { // Avoid division by zero
            for i in 0..ny { // Iterate over response value bins (rows)
                hist2d_norm[[i, j]] = hist2d[[i, j]] / xhist[j];
            }
        }
    }

    // Generate yscale (using bin centers for mode finding)
    let yscale = Array1::linspace(ymin + dy/2.0, ymax - dy/2.0, ny); // Y-scale represents bin centers


    Ok(Hist2D {
        hist2d_norm,
        yscale,
    })
}


/// Calculates the weighted mode average of a stack of step responses using histogram-based mode finding.
/// Returns the averaged step response curve.
fn weighted_mode_avr(
    stacked_responses: &Array2<f32>, // Stack of step responses (windows x time)
    weights: &Array1<f32>, // Weight for each window
    response_time: &Array1<f64>, // Time vector for the response (0 to RESPONSE_LENGTH_S)
    vertrange: [f64; 2], // [ymin, ymax] for the vertical axis (response value)
    vertbins: usize, // Number of vertical bins for the histogram
) -> Result<Array1<f64>, Box<dyn Error>> { // Returns the averaged response curve (Array1<f64>)
    if stacked_responses.is_empty() || weights.is_empty() || stacked_responses.shape()[0] != weights.len() || response_time.len() != stacked_responses.shape()[1] {
        return Err("Input data mismatch for weighted_mode_avr".into());
    }

    let response_len = stacked_responses.shape()[1];

    // 1. Create the weighted 2D histogram.
    let hist_result = create_weighted_hist2d(
        stacked_responses,
        weights,
        response_time,
        vertrange,
        vertbins,
    )?;

    let hist2d_norm = hist_result.hist2d_norm;
    let yscale = hist_result.yscale; // Y-scale represents the center of the vertical bins

    // 2. Find the mode (peak) in each time slice (column) of the normalized histogram.
    let mut mode_response = Array1::<f64>::zeros(response_len);

    for j in 0..response_len { // Iterate over time slices (columns)
        let column = hist2d_norm.column(j);

        // Find the index of the maximum weight in the current time slice
        if let Some((max_idx, &max_weight)) = column.indexed_iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)) {
             if max_weight > 1e-9 { // Only consider non-zero peaks
                 // The mode response value is the center of the y-bin with the maximum weight
                 mode_response[j] = yscale[max_idx]; // Directly assign
             } else {
                 mode_response[j] = 0.0; // Assign default if no significant peak
             }
        } else {
             mode_response[j] = 0.0; // Assign default if column is empty
        }
    }

    Ok(mode_response)
}


/// Calculates the system's step response using windowing, deconvolution, and averaging.
/// Input: Raw time, setpoint and gyro data, sample rate.
/// Output: Tuple of (response_time, low_setpoint_response, high_setpoint_response, combined_response) or an error.
pub fn calculate_step_response_python_style(
    time: &Array1<f64>, // Need time for relative time and windowing
    setpoint: &Array1<f32>,
    gyro_filtered: &Array1<f32>,
    sample_rate: f64,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> { // Updated return type
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

        let mut step_response = cumulative_sum(&impulse_response_truncated);

        // Truncate the step response to the desired response length *before* normalization/QC check
        if step_response.len() < response_length_samples {
             eprintln!("Warning: Step response shorter than response_length_samples for window {}. Skipping.", i);
             continue;
        }
        let truncated_step_response = step_response.slice(s![0..response_length_samples]).to_owned();


        // --- Individual Response Normalization (Y Correction) & Quality Control ---
        let mut passes_qc = false;
        if ss_start_sample < ss_end_sample { // Only perform QC if steady-state window is valid
            let steady_state_segment = truncated_step_response.slice(s![ss_start_sample..ss_end_sample]);

            // Use .mean() from ndarray, not ndarray-stats
            if let Some(steady_state_mean) = steady_state_segment.mean() {
                 if steady_state_mean.is_finite() && steady_state_mean >= STEADY_STATE_MIN_VAL as f32 && steady_state_mean <= STEADY_STATE_MAX_VAL as f32 {
                     // Normalize the entire truncated step response by the steady-state mean
                     if steady_state_mean.abs() > 1e-9 {
                         step_response = truncated_step_response.mapv(|v| v / steady_state_mean);
                         passes_qc = true;
                     } else {
                          eprintln!("Warning: Steady-state mean near zero for window {}. Skipping normalization and QC.", i);
                     }
                 }
            } else {
                 eprintln!("Warning: Could not calculate steady-state mean for window {}. Skipping QC.", i);
            }
        }

        if passes_qc {
             stacked_step_responses_qc.push(step_response);
             window_max_setpoints_qc.push(max_setpoint_in_window);
        }
    }

    let num_qc_windows = stacked_step_responses_qc.len();
    if num_qc_windows == 0 {
        return Err("No windows passed filtering and quality control.".into());
    }

    // Convert Vec<Array1> to Array2 for weighted_mode_avr
    let valid_stacked_responses = Array2::from_shape_fn((num_qc_windows, response_length_samples), |(i, j)| {
        stacked_step_responses_qc[i][j]
    });
    let valid_window_max_setpoints = Array1::from(window_max_setpoints_qc);


    // 3. Implement Setpoint Masking for QC Windows.
    let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v < SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });
    let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v >= SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });
    // Create a mask for all QC windows (all 1.0s)
    let combined_mask: Array1<f32> = Array1::ones(num_qc_windows);


    // 4. Implement Weighted Mode Averaging.
    // Time vector for the response plot (0 to RESPONSE_LENGTH_S).
    let response_time = Array1::linspace(0.0, RESPONSE_LENGTH_S, response_length_samples);

    // Calculate weighted mode average for low setpoint windows.
    let low_response_mode_avg = weighted_mode_avr(
        &valid_stacked_responses,
        &low_mask,
        &response_time.mapv(|t| t as f64),
        [-2.0, 2.0], // Example vertical range for histogram
        100, // Example vertical bins
    )?;

    // Calculate weighted mode average for high setpoint windows.
    let high_response_mode_avg = weighted_mode_avr(
        &valid_stacked_responses,
        &high_mask,
        &response_time.mapv(|t| t as f64),
        [-2.0, 2.0], // Example vertical range
        100, // Example vertical bins
    )?;

    // Calculate weighted mode average for ALL QC windows.
     let combined_response_mode_avg = weighted_mode_avr(
        &valid_stacked_responses,
        &combined_mask, // Use the combined mask
        &response_time.mapv(|t| t as f64),
        [-2.0, 2.0], // Example vertical range
        100, // Example vertical bins
    )?;


    Ok((response_time.mapv(|t| t as f64), low_response_mode_avg, high_response_mode_avg, combined_response_mode_avg)) // Updated return tuple
}