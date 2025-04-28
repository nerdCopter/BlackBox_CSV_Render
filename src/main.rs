use csv::ReaderBuilder; // For reading CSV files efficiently.
use plotters::prelude::*; // For creating plots and charts.
use plotters::prelude::full_palette::ORANGE; // Import the ORANGE color constant from the correct palette
use std::error::Error; // Standard trait for error handling.
use std::env; // For accessing command-line arguments.
use std::path::Path; // For working with file paths.
use std::fs::File; // For file operations (opening files).
use std::io::BufReader; // For buffered reading, improving file I/O performance.
use std::collections::VecDeque; // For efficient moving average window.

// --- Dependencies for Step Response Calculation ---
use realfft::num_complex::Complex32; // Complex number type for FFT results.
use realfft::RealFftPlanner; // FFT planner for real-valued signals.
use ndarray::{Array1, Array2, Axis}; // For multi-dimensional arrays (useful for window stacking and 2D histograms)
use ndarray_stats::QuantileExt; // For finding min/max in ndarrays
use ndarray::s; // Import the slicing macro

/// Structure to hold data parsed from a single row of the CSV log.
/// Uses `Option<f64>` to handle potentially missing or unparseable values.
#[derive(Debug, Default, Clone)]
struct LogRowData {
    time_sec: Option<f64>,        // Timestamp (in seconds).
    p_term: [Option<f64>; 3],     // Proportional term [Roll, Pitch, Yaw]. Header example: "axisP[0]".
    i_term: [Option<f64>; 3],     // Integral term [Roll, Pitch, Yaw]. Header example: "axisI[0]".
    d_term: [Option<f64>; 3],     // Derivative term [Roll, Pitch, Yaw]. Header example: "axisD[0]".
    setpoint: [Option<f64>; 3],   // Target setpoint value [Roll, Pitch, Yaw]. Header example: "setpoint[0]".
    gyro: [Option<f64>; 3],       // Gyroscope readings (filtered) [Roll, Pitch, Yaw]. Header example: "gyroADC[0]".
    gyro_unfilt: [Option<f64>; 3], // Unfiltered Gyroscope readings [Roll, Pitch, Yaw]. Header example: "gyroUnfilt[0]". Fallback: debug[0..2]. Default: 0.0
    debug: [Option<f64>; 4],      // Debug values [0..3]. Header example: "debug[0]".
}

// Define constants for plot dimensions.
const PLOT_WIDTH: u32 = 1920;
const PLOT_HEIGHT: u32 = 1080;

// Define constant for the step response plot duration in seconds.
const STEP_RESPONSE_PLOT_DURATION_S: f64 = 0.5; // Example: 0.5 seconds

// Constants for the new step response calculation method (inspired by Python)
const FRAME_LENGTH_S: f64 = 1.0; // Length of each window in seconds
const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep from each window
const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows within a frame length
const CUTOFF_FREQUENCY_HZ: f64 = 25.0; // Cutoff frequency for Wiener filter noise spectrum
const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)
const SETPOINT_THRESHOLD: f64 = 500.0; // Threshold for low/high setpoint masking

// Helper function to calculate plot range with 15% padding on each side.
// Uses a fixed padding if the range is very small to avoid excessive zoom.
fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    // Add padding to the range. Use larger padding for very small ranges.
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

// Helper function to draw a "Data Unavailable" message on a plot area.
fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
) -> Result<(), Box<dyn Error>> {
    let message = format!("Axis {} {} Data Unavailable", axis_index, plot_type);
    area.draw(&Text::new(
        message,
        (50, 50), // Position within the subplot.
        ("sans-serif", 20).into_font().color(&RED), // Text style.
    ))?;
    Ok(())
}


// --- START: Step Response Calculation Functions (Python-inspired) ---

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
        window[i] = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / (n_alpha as f64)).cos()) as f32; // Fix: Changed f664 to f64
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
             // This case should ideally not be reached with the num_windows calculation, but as a safeguard.
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
    cutoff_frequency_hz: f64,
) -> Array1<f32> {
    let n = input_window.len();
    if n == 0 || sample_rate <= 0.0 {
        return Array1::zeros(n);
    }

    // Pad to next power of 2 for efficient FFT (optional but common practice)
    let padded_n = n.next_power_of_two();
    let input_padded = {
        let mut padded = Array1::<f32>::zeros(padded_n);
        padded.slice_mut(s![0..n]).assign(input_window);
        padded
    };
    let output_padded = {
        let mut padded = Array1::<f32>::zeros(padded_n);
        padded.slice_mut(s![0..n]).assign(output_window);
        padded
    };


    let h_spec = fft_forward(&input_padded);
    let g_spec = fft_forward(&output_padded);

    if h_spec.is_empty() || g_spec.is_empty() || h_spec.len() != g_spec.len() {
         eprintln!("Warning: FFT output empty or length mismatch in Wiener deconvolution.");
         return Array1::zeros(n); // Return zeros if FFT failed
    }

    let freq = fft_rfftfreq(padded_n, 1.0 / sample_rate as f32); // Frequencies for real FFT output

    // Calculate noise spectrum (simplified based on Python logic)
    let mut sn = Array1::<f32>::ones(freq.len());
    // Attenuate frequencies above cutoff
    for i in 0..freq.len() {
        if freq[i] > cutoff_frequency_hz as f32 {
            // Simple rolloff above cutoff
            let factor = (freq[i] - cutoff_frequency_hz as f32) / (sample_rate as f32 / 2.0 - cutoff_frequency_hz as f32).max(1.0);
            sn[i] = 1.0 / (10.0 * factor.max(0.1)); // Increase noise power for higher frequencies
        }
    }
     // Apply Gaussian smoothing to the noise spectrum (simplified)
     // This requires a 1D Gaussian filter implementation or a library.
     // For now, we'll skip the Gaussian filter for simplicity and focus on the core Wiener formula.
     // A proper implementation would use a library like `scipy.ndimage.gaussian_filter1d` equivalent.
     // sn = gaussian_filter1d_rust(&sn, filter_width); // Placeholder for smoothing

    let epsilon = 1e-9; // Small value to prevent division by zero
    let mut deconvolved_spec = Array1::<Complex32>::zeros(h_spec.len());

    for i in 0..h_spec.len() {
        let h = h_spec[i];
        let g = g_spec[i];
        let h_conj = h.conj();
        let sn_val = sn[i]; // Use the calculated noise spectrum value

        // Wiener filter formula: G * H* / (|H|^2 + Sn / Ss)
        // Assuming signal power spectrum Ss is proportional to |H|^2, the term becomes Sn / |H|^2
        // The Python code's `1. / sn` term in the denominator suggests it's using 1 / (Sn/Ss) where Ss is implicitly 1.
        // Let's try to match the Python formula: G * H* / (|H|^2 + 1. / sn)
        let denominator = (h * h_conj).re + (1.0 / (sn_val + epsilon)); // Add epsilon to sn_val too

        if denominator.abs() > epsilon {
            deconvolved_spec[i] = (g * h_conj) / denominator;
        } else {
            deconvolved_spec[i] = Complex32::new(0.0, 0.0); // Avoid division by zero
        }
    }

    // Perform inverse FFT and truncate to original length
    let deconvolved_impulse = fft_inverse(&deconvolved_spec, padded_n);

    // Return the impulse response truncated to the original window length
    deconvolved_impulse.slice(s![0..n]).to_owned()
}

/// Calculates the frequencies for the real FFT output.
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

/// Applies a moving average filter to smooth the input data.
#[allow(dead_code)] // Allow dead code for this function as it's not used in the new calculation
fn moving_average_smooth_array(data: &Array1<f32>, window_size: usize) -> Array1<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_owned(); // No smoothing needed or possible.
    }

    let mut smoothed_data = Array1::<f32>::zeros(data.len());
    let mut current_sum: f32 = 0.0;
    let mut history: VecDeque<f32> = VecDeque::with_capacity(window_size); // Window buffer.

    for (i, &val) in data.iter().enumerate() {
        history.push_back(val); // Add new value to window.
        current_sum += val; // Update sum.

        // If window is full, remove the oldest element.
        if history.len() > window_size {
            if let Some(old_val) = history.pop_front() { // Safely remove oldest value.
                 current_sum -= old_val; // Update sum.
            }
        }

        // Calculate average over the current window size (handles initial partial windows).
        let current_window_len = history.len() as f32;
        if current_window_len > 0.0 {
            smoothed_data[i] = current_sum / current_window_len;
        } else {
            smoothed_data[i] = 0.0; // Should not happen if data is not empty.
        }
    }
    smoothed_data
}

/// Creates a 2D histogram from 1D arrays of x, y, and weights.
/// Returns a struct containing the histogram data and scales.
#[allow(dead_code)] // Allow dead code for this struct as it's not fully implemented/used yet
struct Hist2D {
    hist2d_norm: Array2<f64>,
    hist2d: Array2<f64>,
    xhist: Array1<f64>,
    xscale: Array1<f64>,
    yscale: Array1<f64>,
}

#[allow(dead_code)] // Allow dead code for this function as it's not fully implemented/used yet
fn create_hist2d(
    x: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array2<f64>, // Weights are 2D, corresponding to the stacked data
    bins: (usize, usize), // (nx, ny)
    range: ([f64; 2], [f64; 2]), // ([xmin, xmax], [ymin, ymax])
) -> Result<Hist2D, Box<dyn Error>> {
    if x.len() != weights.shape()[0] || y.len() != weights.shape()[1] {
         return Err("Input lengths mismatch for create_hist2d".into());
    }

    let nx = bins.0;
    let ny = bins.1;
    let xmin = range.0[0];
    let xmax = range.0[1];
    let ymin = range.1[0];
    let ymax = range.1[1];

    let mut hist2d = Array2::<f64>::zeros((ny, nx)); // Note: histogram2d returns shape (ny, nx)
    let mut xhist = Array1::<f64>::zeros(nx);

    // Simple manual binning for demonstration. A proper implementation would use a more efficient algorithm.
    let dx = (xmax - xmin) / nx as f64;
    let dy = (ymax - ymin) / ny as f64;

    for i in 0..weights.shape()[0] { // Iterate over windows
        let current_x = x[i];
        let x_bin = ((current_x - xmin) / dx).floor() as usize;

        if x_bin < nx {
            xhist[x_bin] += 1.0; // Count occurrences in x bin
            for j in 0..weights.shape()[1] { // Iterate over time points in window
                let current_y = y[j];
                let current_weight = weights[[i, j]];
                let y_bin = ((current_y - ymin) / dy).floor() as usize;

                if y_bin < ny {
                    hist2d[[y_bin, x_bin]] += current_weight; // Add weighted value to 2D bin
                }
            }
        }
    }

    // Calculate normalized histogram
    let mut hist2d_norm = Array2::<f64>::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            if xhist[i] > 1e-9 { // Avoid division by zero
                hist2d_norm[[j, i]] = hist2d[[j, i]] / xhist[i];
            }
        }
    }

    // Generate scales
    let xscale = Array1::linspace(xmin, xmax, nx + 1);
    let yscale = Array1::linspace(ymin, ymax, ny + 1);


    Ok(Hist2D {
        hist2d_norm,
        hist2d,
        xhist,
        xscale,
        yscale,
    })
}


/// Calculates the weighted mode average of a stack of step responses.
/// This is a simplified version of the Python logic, focusing on averaging based on weights.
/// A true "mode" average would require finding peaks in a histogram.
fn weighted_mode_avr(
    stacked_responses: &Array2<f32>, // Stack of step responses (windows x time)
    weights: &Array1<f32>, // Weight for each window
    response_time: &Array1<f64>, // Time vector for the response (0 to RESPONSE_LENGTH_S)
    _vertrange: [f64; 2], // [ymin, ymax] for potential histogramming (not used in this simplified version) // Fix: Added underscore
    _vertbins: usize, // Number of vertical bins (not used in this simplified version) // Fix: Added underscore
) -> Result<(Array1<f64>, Array1<f64>, Option<Hist2D>), Box<dyn Error>> {
    if stacked_responses.is_empty() || weights.is_empty() || stacked_responses.shape()[0] != weights.len() {
        return Err("Input data mismatch for weighted_mode_avr".into());
    }

    let num_windows = stacked_responses.shape()[0];
    let response_len = stacked_responses.shape()[1];

    if response_len != response_time.len() {
         return Err("Response time length mismatch".into());
    }

    // Calculate weighted average at each time point
    let mut average_response = Array1::<f64>::zeros(response_len);
    let mut variance_response = Array1::<f64>::zeros(response_len); // For standard deviation

    let total_weight: f64 = weights.iter().map(|&w| w as f64).sum();
    if total_weight < 1e-9 {
         // If total weight is near zero, return zeros and indicate no meaningful average.
         return Ok((Array1::zeros(response_len), Array1::zeros(response_len), None));
    }


    for j in 0..response_len { // Iterate over time points in the response
        let mut weighted_sum = 0.0;
        let mut sum_of_weights = 0.0;
        let mut values_at_time_j = Vec::new(); // Collect values for variance

        for i in 0..num_windows { // Iterate over windows
            let value = stacked_responses[[i, j]] as f64;
            let weight = weights[i] as f64;

            if value.is_finite() && weight.is_finite() && weight > 0.0 {
                weighted_sum += value * weight;
                sum_of_weights += weight;
                values_at_time_j.push(value);
            }
        }

        if sum_of_weights > 1e-9 {
            average_response[j] = weighted_sum / sum_of_weights;

            // Calculate weighted variance
            let mut weighted_variance_sum = 0.0;
            for i in 0..num_windows {
                 let value = stacked_responses[[i, j]] as f64;
                 let weight = weights[i] as f64;
                 if value.is_finite() && weight.is_finite() && weight > 0.0 {
                     weighted_variance_sum += weight * (value - average_response[j]).powi(2);
                 }
            }
            // Use sample variance formula (N-1 in denominator, but weighted)
            // A common weighted variance formula: sum(w * (x - mean)^2) / (sum(w) * (N - 1) / N)
            // Or simpler: sum(w * (x - mean)^2) / (sum(w) - sum(w^2)/sum(w))
            // Let's use a simpler approximation for now: sum(w * (x - mean)^2) / sum(w)
            // This is the population variance. For sample variance, it's more complex.
            // Let's use the sum(w * (x - mean)^2) / sum(w) for simplicity, as in some weighted variance definitions.
            if sum_of_weights > 1e-9 {
                 variance_response[j] = weighted_variance_sum / sum_of_weights;
            } else {
                 variance_response[j] = 0.0; // Avoid division by zero
            }


        } else {
            average_response[j] = 0.0; // No valid data for this time point
            variance_response[j] = 0.0;
        }
    }

    // Standard deviation is the square root of variance
    let std_dev_response = variance_response.mapv(|v| v.sqrt());

    // Note: This simplified version does not return the histogram data, so the third element is None.
    // Implementing the full histogram-based mode finding is more complex.
    Ok((average_response, std_dev_response, None))
}


/// Calculates the system's step response using windowing, deconvolution, and averaging.
/// Input: Raw time, setpoint and gyro data, sample rate.
/// Output: Tuple of (response_time, low_setpoint_response, high_setpoint_response) or an error.
pub fn calculate_step_response_python_style(
    time: &Array1<f64>, // Need time for relative time and windowing // Fix: Changed times to time
    setpoint: &Array1<f32>,
    gyro_filtered: &Array1<f32>,
    sample_rate: f64,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> { // Returns (response_time, low_resp, high_resp)
    // Basic input validation.
    if time.is_empty() || setpoint.is_empty() || gyro_filtered.is_empty() || setpoint.len() != gyro_filtered.len() || time.len() != setpoint.len() || sample_rate <= 0.0 { // Fix: Changed times to time
        return Err("Invalid input to calculate_step_response_python_style: Empty data, length mismatch, or invalid sample rate.".into());
    }

    let _total_len = time.len(); // Fix: Changed times to time, Added underscore
    let _dt = 1.0 / sample_rate; // Added underscore

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
    let stacked_input_windowed = &stacked_input * &window_func; // Fix: Element-wise multiplication
    let stacked_output_windowed = &stacked_output * &window_func; // Fix: Element-wise multiplication


    // Prepare storage for step responses from each window.
    let mut stacked_step_responses = Array2::<f32>::zeros((num_windows, response_length_samples));
    let mut window_max_setpoints = Array1::<f32>::zeros(num_windows); // To store max setpoint per window

    // 2. Perform Wiener deconvolution and cumulative sum for each window.
    for i in 0..num_windows {
        let input_window = stacked_input_windowed.row(i).to_owned();
        let output_window = stacked_output_windowed.row(i).to_owned();

        // Calculate max setpoint in the original input window for masking later.
        // Use the original (non-windowed) input for the setpoint check.
        let original_input_window = stacked_input.row(i).to_owned();
        window_max_setpoints[i] = original_input_window.iter().fold(0.0, |max_val, &v| max_val.max(v.abs()));


        let impulse_response = wiener_deconvolution_window(
            &input_window,
            &output_window,
            sample_rate,
            CUTOFF_FREQUENCY_HZ,
        );

        // The impulse response length from Wiener deconvolution is padded_n, not frame_length_samples.
        // We need to truncate it first before cumulative sum and final truncation.
        let impulse_response_truncated = impulse_response.slice(s![0..frame_length_samples]).to_owned();


        if impulse_response_truncated.is_empty() || impulse_response_truncated.len() != frame_length_samples {
             eprintln!("Warning: Truncated impulse response length mismatch for window {}. Skipping.", i);
             continue; // Skip this window if deconvolution failed or length is wrong
        }

        let step_response = cumulative_sum(&impulse_response_truncated);

        // Truncate the step response to the desired response length.
        if step_response.len() >= response_length_samples {
            stacked_step_responses.row_mut(i).assign(&step_response.slice(s![0..response_length_samples]));
        } else {
             eprintln!("Warning: Step response shorter than response_length_samples for window {}. Skipping.", i);
             continue; // Skip if step response is too short
        }
    }

    // Filter out windows where deconvolution/processing failed (rows of zeros in stacked_step_responses)
    let valid_window_indices: Vec<usize> = (0..num_windows)
        .filter(|&i| stacked_step_responses.row(i).iter().any(|&v| v != 0.0)) // Keep windows that are not all zeros
        .collect();

    if valid_window_indices.is_empty() {
         return Err("No valid step responses calculated from windows.".into());
    }

    let valid_stacked_responses = stacked_step_responses.select(Axis(0), &valid_window_indices);
    let valid_window_max_setpoints = window_max_setpoints.select(Axis(0), &valid_window_indices);


    // 3. Implement Setpoint Masking for Windows.
    let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v < SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });
    let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v >= SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });

    // Optional: Implement `toolow_mask` similar to Python if needed for filtering very low input windows.
    // For now, we'll use simple low/high masks.

    // 4. Implement Weighted Mode Averaging.
    // We need a time vector for the response plot (0 to RESPONSE_LENGTH_S).
    let response_time = Array1::linspace(0.0, RESPONSE_LENGTH_S, response_length_samples);

    // Calculate weighted average for low setpoint windows.
    let (low_response_avg, _low_response_std, _low_hist) = weighted_mode_avr(
        &valid_stacked_responses,
        &low_mask, // Use low_mask as weights
        &response_time.mapv(|t| t as f64), // Convert time to f64 for weighted_mode_avr
        [-2.0, 2.0], // Example vertical range for potential histogramming
        1000, // Example vertical bins
    )?;

    // Calculate weighted average for high setpoint windows.
    let (high_response_avg, _high_response_std, _high_hist) = weighted_mode_avr(
        &valid_stacked_responses,
        &high_mask, // Use high_mask as weights
        &response_time.mapv(|t| t as f64), // Convert time to f64 for weighted_mode_avr
        [-2.0, 2.0], // Example vertical range
        1000, // Example vertical bins
    )?;

    // 5. Return the calculated step responses and the response time vector.
    Ok((response_time.mapv(|t| t as f64), low_response_avg, high_response_avg))
}

// --- END: Step Response Calculation Functions ---


fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file.csv>", args[0]);
        std::process::exit(1);
    }
    let input_file = &args[1];
    let input_path = Path::new(input_file);
    // Extract the base name of the input file for naming output plots.
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    // Define the exact CSV headers needed for the analysis.
    let target_headers = [
        // Time (0)
        "time (us)",
        // P Term (1-3)
        "axisP[0]", "axisP[1]", "axisP[2]",
        // I Term (4-6)
        "axisI[0]", "axisI[1]", "axisI[2]",
        // D Term (7-9)
        "axisD[0]", "axisD[1]", "axisD[2]",
        // Setpoint (10-12)
        "setpoint[0]", "setpoint[1]", "setpoint[2]",
        // Gyroscope Data (Filtered) (13-15)
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        // Gyroscope Data (Unfiltered) (16-18)
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]",
        // Debug Data (19-22)
        "debug[0]", "debug[1]", "debug[2]", "debug[3]",
    ];

    // Flags to track if specific optional or plot-dependent headers are found.
    let mut axis_d2_header_found = false; // Tracks if "axisD[2]" is present.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[axis]" is present.
    let mut gyro_header_found = [false; 3]; // Tracks if "gyroADC[axis]" is present (filtered gyro).
    let mut gyro_unfilt_header_found = [false; 3]; // Tracks if "gyroUnfilt[axis]" is present.
    let mut debug_header_found = [false; 4]; // Tracks if "debug[idx]" is present.


    // Read the CSV header row and map target headers to their column indices.
    let header_indices: Vec<Option<usize>> = {
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // First row is the header.
            .trim(csv::Trim::All) // Trim whitespace from headers and fields.
            .from_reader(BufReader::new(file)); // Use buffered reader for efficiency.

        let header_record = reader.headers()?.clone(); // Get the header record.
        println!("Headers found in CSV: {:?}", header_record);

        // Find the index of each target header in the actual CSV header record.
        let indices: Vec<Option<usize>> = target_headers
            .iter()
            .map(|&target_header| {
                // `.position` finds the first occurrence. Trim headers from file for robust matching.
                header_record.iter().position(|h| h.trim() == target_header)
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut essential_pid_headers_found = true; // Flag for essential headers for PIDsum plot.

        // Check essential PID headers (Indices 0-8).
        for i in 0..=8 {
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_pid_headers_found = false; // Mark as missing if any essential PID header is not found.
                    format!("Not Found (Essential for PIDsum Plot!)")
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }

        // Check optional 'axisD[2]' header (Target index 9).
        let axis_d2_name = target_headers[9];
        let axis_d2_status = match indices[9] {
            Some(idx) => {
                axis_d2_header_found = true; // Mark as found if present.
                format!("Found at index {}", idx)
            }
            None => format!("Not Found (Optional for PIDsum plot, will default to 0.0)"),
        };
        println!("  '{}' (Target Index {}): {}", axis_d2_name, 9, axis_d2_status);

        // Check setpoint headers (Target indices 10, 11, 12).
        for axis in 0..3 {
            let target_idx = 10 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    setpoint_header_found[axis] = true; // Mark axis-specific flag.
                    format!("Found at index {}", idx)
                 }
                 None => {
                    // Essential for both Setpoint vs PIDsum and Step Response plots for this axis.
                    format!("Not Found (Essential for Setpoint vs PIDsum Plot Axis {} AND Step Response Plot Axis {})", axis, axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }

        // Check gyro (filtered) headers (Target indices 13, 14, 15).
         for axis in 0..3 {
            let target_idx = 13 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    gyro_header_found[axis] = true; // Mark axis-specific flag.
                    format!("Found at index {}", idx)
                 }
                 None => {
                    // Essential for Step Response plot for this axis. Also needed for Gyro vs Unfilt plot.
                    format!("Not Found (Essential for Step Response Plot Axis {} AND Gyro vs Unfilt Plot Axis {})", axis, axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }

        // Check gyroUnfilt headers (Target indices 16, 17, 18).
        for axis in 0..3 {
            let target_idx = 16 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    gyro_unfilt_header_found[axis] = true; // Mark axis-specific flag.
                    format!("Found at index {}", idx)
                 }
                 None => {
                    format!("Not Found (Will try to use debug[{}] as fallback for Gyro vs Unfilt plot)", axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }

        // Check debug headers (Target indices 19, 20, 21, 22).
        for idx_offset in 0..4 {
            let target_idx = 19 + idx_offset;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    debug_header_found[idx_offset] = true; // Mark flag.
                    format!("Found at index {}", idx)
                 }
                 None => {
                    format!("Not Found (Optional, used as fallback for gyroUnfilt[0-2])")
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }


        // Exit if any essential header for the basic PIDsum plot is missing.
        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8) // Check indices 0 through 8.
                 .filter(|&i| indices[i].is_none()) // Find which ones are None.
                 .map(|i| format!("'{}'", target_headers[i])) // Format their names.
                 .collect();
             // Return an error indicating which essential headers are missing.
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
        indices // Return the vector of Option<usize> indices.
    };

    // --- Data Reading and Storage ---
    // Vector to store parsed data from each valid row.
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading P/I/D term, Setpoint, Gyro, and Debug data from CSV...");
    { // Inner scope to ensure the file reader is dropped after reading.
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // Skip the header row during data reading.
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        // Iterate through each row of the CSV file.
        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => { // Successfully read a record (row).
                    let mut current_row_data = LogRowData::default(); // Initialize struct for this row.

                    // Helper closure to parse a value from the record using the target header index.
                    // Handles cases where the header wasn't found (index is None) or parsing fails.
                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx) // Get the Option<usize> for the target header.
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref()) // Convert Option<&Option<usize>> to Option<&usize>.
                            .and_then(|&csv_idx| record.get(csv_idx)) // Get the string value from the record using the csv_idx.
                            .and_then(|val_str| val_str.parse::<f64>().ok()) // Try parsing the string to f64.
                    };

                    // --- Parse Time ---
                    // Time (us) is target index 0.
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         // Convert microseconds to seconds.
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         // Skip row if time is missing or invalid, as it's essential.
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue; // Move to the next row.
                    }

                    // --- Parse P, I, D, Setpoint, Gyro (filtered) for each axis (0=Roll, 1=Pitch, 2=Yaw) ---
                    for axis in 0..3 {
                        // P term (target indices 1, 2, 3)
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        // I term (target indices 4, 5, 6)
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // D term (target indices 7, 8, 9)
                        let d_target_idx = 7 + axis;
                        // Special handling for optional axisD[2].
                        if axis == 2 && !axis_d2_header_found {
                             current_row_data.d_term[axis] = Some(0.0); // Default to 0.0 if header is missing.
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        // Setpoint (target indices 10, 11, 12) - Only parse if the header was found.
                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        } // Otherwise, it remains None (default).

                        // Gyro (filtered) (target indices 13, 14, 15) - Only parse if the header was found.
                        if gyro_header_found[axis] {
                             current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
                        } // Otherwise, it remains None (default).
                    }

                    // --- Parse Gyro Unfiltered and Debug ---
                    let mut parsed_gyro_unfilt = [None; 3];
                    let mut parsed_debug = [None; 4];

                    // Parse gyroUnfilt (target indices 16, 17, 18) - Only if header was found.
                    for axis in 0..3 {
                        if gyro_unfilt_header_found[axis] {
                            parsed_gyro_unfilt[axis] = parse_f64_by_target_idx(16 + axis);
                        }
                    }

                    // Parse debug (target indices 19, 20, 21, 22) - Only if header was found.
                    for idx_offset in 0..4 {
                        if debug_header_found[idx_offset] {
                            parsed_debug[idx_offset] = parse_f64_by_target_idx(19 + idx_offset);
                        }
                        // Store the parsed debug value directly into the row data.
                        current_row_data.debug[idx_offset] = parsed_debug[idx_offset];
                    }

                    // --- Apply Fallback Logic for gyro_unfilt ---
                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val), // Use gyroUnfilt if available
                            None => match parsed_debug[axis] { // Try debug[axis] as fallback (indices 0, 1, 2 match)
                                Some(val) => Some(val), // Use debug[axis]
                                None => Some(0.0),      // Default to 0.0 if both are missing/invalid
                            }
                        };
                    }

                    // Store the parsed data for this row.
                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    // Report error if a row cannot be read/parsed by the CSV reader.
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // Reader is dropped here.

    println!("Finished reading {} data rows.", all_log_data.len());
    // Report status of optional/conditional headers.
    if !axis_d2_header_found {
        println!("INFO: 'axisD[2]' header was not found. Used 0.0 for Axis 2 D-term calculation in PIDsum plot.");
    }
    for axis in 0..3 {
        if !setpoint_header_found[axis] {
             println!("INFO: 'setpoint[{}]' header was not found. Setpoint vs PIDsum and Step Response plots for Axis {} cannot be generated.", axis, axis);
        }
         if !gyro_header_found[axis] {
             println!("INFO: 'gyroADC[{}]' header was not found. Step Response and Gyro vs Unfilt plots for Axis {} cannot be generated.", axis, axis);
        }
    }
    // Report status of gyroUnfilt and debug headers and fallback logic.
    for axis in 0..3 {
        if !gyro_unfilt_header_found[axis] {
            if debug_header_found[axis] {
                println!("INFO: 'gyroUnfilt[{}]' header was not found. Used 'debug[{}]' as fallback for Gyro vs Unfilt plot.", axis, axis);
            } else {
                println!("INFO: Neither 'gyroUnfilt[{}]' nor 'debug[{}]' headers were found. Used 0.0 for gyro_unfilt[{}] in Gyro vs Unfilt plot.", axis, axis, axis);
            }
        }
    }
    for idx_offset in 0..4 {
        if !debug_header_found[idx_offset] {
            println!("INFO: 'debug[{}]' header was not found. Data will be None (unless used as fallback for gyroUnfilt).", idx_offset);
        }
    }


    // Exit if no valid data rows were read.
    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    if all_log_data.len() > 1 {
        let mut total_delta = 0.0; // Sum of time differences between consecutive valid samples.
        let mut count = 0; // Number of valid time differences calculated.
        for i in 1..all_log_data.len() {
            // Ensure both current and previous timestamps are valid.
            if let (Some(t1), Some(t0)) = (all_log_data[i].time_sec, all_log_data[i-1].time_sec) {
                let delta = t1 - t0;
                // Only consider positive time differences to avoid issues with duplicate timestamps or out-of-order data.
                if delta > 1e-9 { // Use a small epsilon to compare floating points.
                    total_delta += delta;
                    count += 1;
                }
            }
        }
        // Calculate average delta time if valid differences were found.
        if count > 0 {
            let avg_delta = total_delta / count as f64;
            sample_rate = Some(1.0 / avg_delta); // Sample rate is the inverse of average delta time.
            println!("Estimated Sample Rate: {:.2} Hz", sample_rate.unwrap());
        }
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might fail or be inaccurate.");
         // Proceed, but step response calculation will likely return empty or fail later.
    }


    // --- Data Preparation for Plots ---
    // Data structure for PIDsum plot: (time, P+I+D) for each axis.
    let mut pid_output_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Data structure for Setpoint vs PIDsum plot: (time, setpoint, P+I+D) for each axis.
    let mut setpoint_vs_pidsum_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Data structure for Gyro vs Unfiltered Gyro plot: (time, gyro_filtered, gyro_unfiltered) for each axis.
    let mut gyro_vs_unfilt_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Temporary storage for inputs needed by `calculate_step_response`: (times, setpoints, gyros) for each axis.
    // Note: Step response uses the *filtered* gyro (`gyroADC`).
    let mut step_response_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];
    // Flags indicating if sufficient data exists for each plot type per axis.
    let mut pid_data_available = [false; 3]; // Tracks if P, I, and D are available for PIDsum.
    let mut setpoint_data_available = [false; 3]; // Tracks if Setpoint, P, I, D are available for Setpoint vs PIDsum.
    let mut gyro_vs_unfilt_data_available = [false; 3]; // Tracks if filtered and unfiltered gyro data are available.
    // Tracks if the *input* data (time, setpoint, gyro) required for step response calculation is present.
    let mut step_response_input_available = [false; 3];


    // First pass: Iterate through parsed `all_log_data` to populate plot data structures.
    for row in &all_log_data {
        if let Some(time) = row.time_sec { // Only process rows with valid time.
            for axis_index in 0..3 {
                // Prepare PIDsum data: Requires P, I, D terms for the axis.
                if let (Some(p), Some(i), Some(d)) =
                    (row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    pid_output_data[axis_index].push((time, p + i + d));
                    pid_data_available[axis_index] = true; // Mark PIDsum as available for this axis.
                }

                // Prepare Setpoint vs PIDsum data: Requires Setpoint, P, I, D terms.
                // Only attempt if the setpoint header for this axis was found earlier.
                if setpoint_header_found[axis_index] {
                    if let (Some(setpoint), Some(p), Some(i), Some(d)) =
                        (row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                    {
                        setpoint_vs_pidsum_data[axis_index].push((time, setpoint, p + i + d));
                        setpoint_data_available[axis_index] = true; // Mark SetpointVsPidsum available.
                    }
                }

                 // Collect Step Response Input Data: Requires Time, Setpoint, Gyro (filtered).
                 // Only attempt if both setpoint and *filtered* gyro headers for this axis were found.
                 if setpoint_header_found[axis_index] && gyro_header_found[axis_index] {
                     // Use row.gyro (filtered gyro) for step response calculation.
                     if let (Some(setpoint), Some(gyro_filt)) = (row.setpoint[axis_index], row.gyro[axis_index]) {
                        // Append data to the respective vectors within the tuple for this axis.
                        step_response_input_data[axis_index].0.push(time); // Time (f64)
                        step_response_input_data[axis_index].1.push(setpoint as f32); // Setpoint (f32)
                        step_response_input_data[axis_index].2.push(gyro_filt as f32); // Filtered Gyro (f32)
                        step_response_input_available[axis_index] = true; // Mark that input data exists for this axis.
                     }
                 }

                 // Collect Gyro vs Unfiltered Gyro Data: Requires Time, Gyro (filtered), Gyro (unfiltered).
                 // Filtered gyro header must be found. Unfiltered gyro uses fallback logic, so row.gyro_unfilt should always be Some.
                 if gyro_header_found[axis_index] { // Check if filtered gyro header exists
                     if let (Some(gyro_filt), Some(gyro_unfilt)) = (row.gyro[axis_index], row.gyro_unfilt[axis_index]) {
                         gyro_vs_unfilt_data[axis_index].push((time, gyro_filt, gyro_unfilt));
                         gyro_vs_unfilt_data_available[axis_index] = true; // Mark GyroVsUnfilt available.
                     }
                 }
            }
        }
    }

    // --- Calculate Step Response Data (using Python-inspired method) ---
    println!("\n--- Calculating Step Response (Python-inspired method) ---");
    // Stores the calculated step response data: (response_time, low_resp, high_resp) for each axis.
    let mut step_response_results: [Option<(Array1<f64>, Array1<f64>, Array1<f64>)>; 3] = [None, None, None];

     if let Some(sr) = sample_rate { // Only proceed if sample rate was determined.
        for axis_index in 0..3 {
            // Only calculate if the required input data (time, setpoint, gyro_filtered) was collected earlier.
            if step_response_input_available[axis_index] {
                println!("  Calculating step response for Axis {}...", axis_index);
                 // Get references to the input data vectors for this axis.
                let time_arr = Array1::from(step_response_input_data[axis_index].0.clone()); // Clone to convert to Array1 // Fix: Renamed times_arr to time_arr
                let setpoints_arr = Array1::from(step_response_input_data[axis_index].1.clone());
                let gyros_filtered_arr = Array1::from(step_response_input_data[axis_index].2.clone());

                // Check if there are enough data points for windowing.
                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                if time_arr.len() >= min_required_samples { // Fix: Changed times_arr to time_arr
                    match calculate_step_response_python_style(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) { // Fix: Changed ×_arr to ×_arr
                        Ok(result) => {
                            // Check if the calculated responses are non-empty.
                            if !result.0.is_empty() && (!result.1.is_empty() || !result.2.is_empty()) {
                                step_response_results[axis_index] = Some(result); // Store the calculated data.
                                println!("    ... Calculation successful for Axis {}.", axis_index);
                            } else {
                                println!("    ... Calculation returned empty responses for Axis {}.", axis_index);
                            }
                        }
                        Err(e) => {
                            eprintln!("    ... Calculation failed for Axis {}: {}", axis_index, e);
                        }
                    }
                } else {
                     // Not enough data points for windowing.
                     println!("    ... Skipping Axis {}: Not enough data points ({}) for windowing (need at least {}).", axis_index, time_arr.len(), min_required_samples); // Fix: Changed times_arr to time_arr
                }
            } else {
                 // Input data was missing (Setpoint or Gyro (filtered) header likely not found).
                 println!("  Skipping Axis {}: Missing required input data (Setpoint or Filtered Gyro 'gyroADC').", axis_index);
            }
        }
    } else {
         // Sample rate is unknown, cannot perform calculation.
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Generate Stacked PIDsum Plot ---
    println!("\n--- Generating Stacked PIDsum Plot (All Axes) ---");
    // Check if PIDsum data is available for at least one axis.
    if pid_data_available.iter().any(|&x| x) {
        let output_file_pidsum = format!("{}_PIDsum_stacked.png", root_name);
        // Setup the drawing backend and main drawing area.
        let root_area_pidsum = BitMapBackend::new(&output_file_pidsum, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_pidsum.fill(&WHITE)?; // White background.
        // Split the main area into 3 vertical subplots (one for each axis).
        let sub_plot_areas = root_area_pidsum.split_evenly((3, 1));
        let pidsum_plot_color = Palette99::pick(1); // Consistent color for PIDsum.

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index]; // Get the subplot area for this axis.
            // Check if data exists and is non-empty for this specific axis.
            if pid_data_available[axis_index] && !pid_output_data[axis_index].is_empty() {
                // Find min/max time and PIDsum value for axis-specific plot ranges.
                let (time_min, time_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
                let (output_min, output_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));

                 // If min/max are still infinite, data was likely invalid (e.g., all NaN). Draw unavailable message.
                 if time_min.is_infinite() || output_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "PIDsum")?;
                     continue; // Move to the next axis subplot.
                 }

                // Final ranges with padding.
                let (final_time_min, final_time_max) = (time_min, time_max); // No padding needed for time usually.
                let (final_pidsum_min, final_pidsum_max) = calculate_range(output_min, output_max);

                // Build the chart within the subplot area.
                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} PIDsum (P+I+D)", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_pidsum_min..final_pidsum_max)?;
                // Configure mesh lines and labels.
                chart.configure_mesh().x_desc("Time (s)").y_desc("PIDsum").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;
                // Draw the PIDsum data as a line series.
                chart.draw_series(LineSeries::new(
                    pid_output_data[axis_index].iter().cloned(), // Clone data points for the series.
                    &pidsum_plot_color,
                ))?;
            } else {
                // Data not available for this axis, draw placeholder message.
                println!("  INFO: No PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                draw_unavailable_message(area, axis_index, "PIDsum")?;
            }
        }
        root_area_pidsum.present()?; // Save the plot to the file.
        println!("  Stacked PIDsum plot saved as '{}'.", output_file_pidsum);
    } else {
        // No PIDsum data available for any axis.
        println!("  Skipping Stacked PIDsum Plot: No PIDsum data available for any axis.");
    }


    // --- Generate Stacked Setpoint vs PIDsum Plot ---
    println!("\n--- Generating Stacked Setpoint vs PIDsum Plot (All Axes) ---");
    // Check if Setpoint vs PIDsum data is available for at least one axis.
    if setpoint_data_available.iter().any(|&x| x) {
        let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
        let root_area_setpoint = BitMapBackend::new(&output_file_setpoint, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint.fill(&WHITE)?;
        let sub_plot_areas = root_area_setpoint.split_evenly((3, 1));
        let setpoint_plot_color = Palette99::pick(2); // Consistent color for Setpoint.
        let pidsum_vs_setpoint_color = Palette99::pick(0); // Consistent color for PIDsum (different from above).

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            // Check if data exists and is non-empty for this specific axis.
            if setpoint_data_available[axis_index] && !setpoint_vs_pidsum_data[axis_index].is_empty() {
                // Find min/max time.
                let (time_min, time_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                // Find min/max value across *both* setpoint and PIDsum for Y-axis range.
                let (val_min, val_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| {
                        (min_y.min(*s).min(*p), max_y.max(*s).max(*p))
                    });

                 // Check for invalid range data.
                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
                     continue;
                 }

                // Final ranges with padding.
                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                // Build the chart.
                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs PIDsum", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_value_min..final_value_max)?;
                // Configure mesh and labels.
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                // Draw Setpoint series.
                let sp_color_ref = &setpoint_plot_color; // Need reference for legend closure.
                chart.draw_series(LineSeries::new(
                    // Map data to (time, setpoint) tuples.
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, s, _p)| (*t, *s)),
                    sp_color_ref,
                ))?
                .label("Setpoint") // Add label for the legend.
                // Define how the legend entry looks.
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color_ref.stroke_width(2)));

                // Draw PIDsum series.
                let pid_color_ref = &pidsum_vs_setpoint_color; // Need reference for legend closure.
                chart.draw_series(LineSeries::new(
                    // Map data to (time, pidsum) tuples.
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, _s, p)| (*t, *p)),
                    pid_color_ref,
                ))?
                .label("PIDsum") // Add label for the legend.
                // Define how the legend entry looks.
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_color_ref.stroke_width(2)));

                // Configure and draw the legend.
                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                // Data not available for this axis, draw placeholder.
                println!("  INFO: No Setpoint vs PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                 draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
            }
        }
        root_area_setpoint.present()?; // Save the plot to file.
        println!("  Stacked Setpoint vs PIDsum plot saved as '{}'.", output_file_setpoint);
    } else {
        // No Setpoint vs PIDsum data available for any axis.
        println!("  Skipping Stacked Setpoint vs PIDsum Plot: No Setpoint vs PIDsum data available for any axis.");
    }


    // --- Generate Stacked Step Response Plot ---
    println!("\n--- Generating Stacked Step Response Plot (All Axes) ---");
    const SETPOINT_THRESHOLD_PLOT: f64 = 500.0; // Define the threshold for coloring in the plot

    // Check if step response calculation was successful for at least one axis.
    if step_response_results.iter().any(|x| x.is_some()) { // Fix: Removed & from closure pattern
        // Dynamic filename based on plot duration.
        let output_file_step = format!("{}_step_response_stacked_plot_{}s.png", root_name, STEP_RESPONSE_PLOT_DURATION_S);
        let root_area_step = BitMapBackend::new(&output_file_step, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_step.fill(&WHITE)?;
        let sub_plot_areas = root_area_step.split_evenly((3, 1));
        let step_response_color_low_sp = Palette99::pick(3); // Color for low setpoint response
        let step_response_color_high_sp = &ORANGE; // Color for high setpoint response

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];
            // Check if calculation succeeded and returned data for this axis.
            if let Some((response_time, low_response_avg, high_response_avg)) = &step_response_results[axis_index] {

                // We are plotting the calculated low and high step response curves directly.
                // The time vector is `response_time`.

                // Find plot ranges specifically for the calculated step response data.
                // Time range is from 0 to RESPONSE_LENGTH_S (which is STEP_RESPONSE_PLOT_DURATION_S).
                let _time_min_plot = response_time.min().cloned().unwrap_or(0.0); // Fix: Added underscore
                let _time_max_plot = response_time.max().cloned().unwrap_or(STEP_RESPONSE_PLOT_DURATION_S); // Fix: Added underscore


                 // Find min/max value across *both* low and high responses for Y-axis range.
                 let mut resp_min = f64::INFINITY;
                 let mut resp_max = f64::NEG_INFINITY;

                 if !low_response_avg.is_empty() {
                     resp_min = resp_min.min(low_response_avg.min().cloned().unwrap_or(f64::INFINITY));
                     resp_max = resp_max.max(low_response_avg.max().cloned().unwrap_or(f64::NEG_INFINITY));
                 }
                 if !high_response_avg.is_empty() {
                     resp_min = resp_min.min(high_response_avg.min().cloned().unwrap_or(f64::INFINITY));
                     resp_max = resp_max.max(high_response_avg.max().cloned().unwrap_or(f64::NEG_INFINITY));
                 }


                // If no valid data points were found in either response, min/max will still be infinite.
                if resp_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Step Response (No Plottable Data)")?;
                     continue;
                 }

                // Apply padding to the response range.
                let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
                // Use the target duration for the X-axis max, plus padding.
                let final_time_max = STEP_RESPONSE_PLOT_DURATION_S * 1.05; // Apply padding to the target duration

                // Build the chart. Time axis starts at 0 relative to the start of the response.
                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Step Response (~{}s)", axis_index, STEP_RESPONSE_PLOT_DURATION_S), ("sans-serif", 20)) // Updated caption
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    // Use 0.0 as the start time for the plot, and the calculated max plot time.
                    .build_cartesian_2d(0f64..final_time_max.max(1e-9), final_resp_min..final_resp_max)?; // Ensure time range is at least non-zero


                // Configure mesh and labels, using fewer X labels suitable for the shorter time range.
                chart.configure_mesh()
                    .x_desc("Time (s) relative to response start") // Updated label
                    .y_desc("Normalized Response")
                    .x_labels(8) // Adjusted labels for fixed duration
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7))
                    .label_style(("sans-serif", 12))
                    .draw()?;

                // Draw the low setpoint (< 500 deg/s) step response line
                if !low_response_avg.is_empty() {
                    let low_sp_color_ref = &step_response_color_low_sp; // Reference for closure
                    chart.draw_series(LineSeries::new(
                        response_time.iter().zip(low_response_avg.iter()).map(|(&t, &v)| (t, v)),
                        low_sp_color_ref,
                    ))?
                    .label(format!("< {} deg/s", SETPOINT_THRESHOLD_PLOT)) // Add label for the legend.
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], low_sp_color_ref.stroke_width(2)));
                }


                // Draw the high setpoint (>= 500 deg/s) step response line
                if !high_response_avg.is_empty() {
                    let high_sp_color_ref = step_response_color_high_sp; // Reference for closure
                    chart.draw_series(LineSeries::new(
                        response_time.iter().zip(high_response_avg.iter()).map(|(&t, &v)| (t, v)),
                        high_sp_color_ref,
                    ))?
                    .label(format!("\u{2265} {} deg/s", SETPOINT_THRESHOLD_PLOT)) // Add label for the legend (using >= symbol)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], high_sp_color_ref.stroke_width(2)));
                }


                // Configure and draw the legend.
                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;

            } else {
                // Step response data is unavailable for this axis. Determine the reason.
                let reason = if !setpoint_header_found[axis_index] || !gyro_header_found[axis_index] {
                    "Setpoint/gyroADC Header Missing" // Missing essential headers for step response.
                 } else if sample_rate.is_none() { // Check using is_none() method
                    "Sample Rate Unknown" // Sample rate couldn't be estimated.
                 } else if !step_response_input_available[axis_index] {
                     "Input Data Missing/Invalid" // Headers found, but no valid data rows.
                 } else { // Headers present, sample rate known, input data collected, but calculation failed/returned empty.
                     "Calculation Failed/No Data"
                 };
                println!("  INFO: No Step Response data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Step Response ({})", reason))?;
            }
        }
        root_area_step.present()?; // Save the plot file.
        println!("  Stacked Step Response plot saved as '{}'. (Duration: {}s)", output_file_step, STEP_RESPONSE_PLOT_DURATION_S);

    } else {
        // Step response calculation did not succeed for any axis.
        println!("  Skipping Stacked Step Response Plot: No step response data could be calculated for any axis.");
    }

    // --- Generate Stacked Gyro vs Unfiltered Gyro Plot ---
    println!("\n--- Generating Stacked Gyro vs Unfiltered Gyro Plot (All Axes) ---");
    // Check if Gyro vs Unfiltered Gyro data is available for at least one axis.
    if gyro_vs_unfilt_data_available.iter().any(|&x| x) {
        let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
        let root_area_gyro = BitMapBackend::new(&output_file_gyro, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_gyro.fill(&WHITE)?;
        let sub_plot_areas = root_area_gyro.split_evenly((3, 1));
        let gyro_unfilt_color = Palette99::pick(4).mix(0.6); // Lighter/desaturated color for unfiltered
        let gyro_filt_color = Palette99::pick(5).filled(); // More prominent color for filtered

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            // Check if data exists and is non-empty for this specific axis.
            if gyro_vs_unfilt_data_available[axis_index] && !gyro_vs_unfilt_data[axis_index].is_empty() {
                // Find min/max time.
                let (time_min, time_max) = gyro_vs_unfilt_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                // Find min/max value across *both* filtered and unfiltered gyro for Y-axis range.
                let (val_min, val_max) = gyro_vs_unfilt_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, gf, gu)| {
                        (min_y.min(*gf).min(*gu), max_y.max(*gf).max(*gu))
                    });

                 // Check for invalid range data.
                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Gyro/UnfiltGyro")?;
                     continue;
                 }

                // Final ranges with padding.
                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                // Build the chart.
                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Filtered vs Unfiltered Gyro", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_value_min..final_value_max)?;
                // Configure mesh and labels.
                chart.configure_mesh().x_desc("Time (s)").y_desc("Gyro Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                // Draw Unfiltered Gyro series (drawn first, potentially less prominent).
                let unfilt_color_ref = &gyro_unfilt_color;
                chart.draw_series(LineSeries::new(
                    // Map data to (time, gyro_unfiltered) tuples.
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, _gf, gu)| (*t, *gu)),
                    unfilt_color_ref,
                ))?
                .label("Unfiltered Gyro (gyroUnfilt/debug)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], unfilt_color_ref.stroke_width(2)));

                // Draw Filtered Gyro series (drawn second, more prominent).
                let filt_color_ref = &gyro_filt_color;
                chart.draw_series(LineSeries::new(
                    // Map data to (time, gyro_filtered) tuples.
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, gf, _gu)| (*t, *gf)),
                    filt_color_ref.stroke_width(2), // Make filtered line slightly thicker
                ))?
                .label("Filtered Gyro (gyroADC)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], filt_color_ref.stroke_width(3))); // Thicker legend line too

                // Configure and draw the legend.
                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                // Data not available for this axis, draw placeholder.
                let reason = if !gyro_header_found[axis_index] {
                    "gyroADC Header Missing" // Filtered gyro is essential
                 } else {
                    "No Valid Data Rows" // Header found, but no rows had both values
                 };
                println!("  INFO: No Gyro vs Unfiltered Gyro data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Gyro/UnfiltGyro ({})", reason))?;
            }
        }
        root_area_gyro.present()?; // Save the plot to file.
        println!("  Stacked Gyro vs Unfiltered Gyro plot saved as '{}'.", output_file_gyro);
    } else {
        // No Gyro vs Unfiltered Gyro data available for any axis.
        println!("  Skipping Stacked Gyro vs Unfiltered Gyro Plot: No data available for any axis.");
    }


    Ok(()) // Indicate successful execution.
}