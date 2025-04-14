use csv::ReaderBuilder; // For reading CSV files efficiently.
use plotters::prelude::*; // For creating plots and charts.
use std::error::Error; // Standard trait for error handling.
use std::env; // For accessing command-line arguments.
use std::path::Path; // For working with file paths.
use std::fs::File; // For file operations (opening files).
use std::io::BufReader; // For buffered reading, improving file I/O performance.
use std::collections::VecDeque; // For moving average in step response

// --- Dependencies for Step Response ---
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

/// Structure to hold the relevant data extracted from a single row of the CSV log.
/// Uses `Option<f64>` to gracefully handle missing or unparseable values in the CSV.
#[derive(Debug, Default, Clone)]
struct LogRowData {
    time_sec: Option<f64>,        // Timestamp of the log entry, converted to seconds.
    p_term: [Option<f64>; 3],     // Proportional term for each axis (Roll, Pitch, Yaw). Assumes CSV header like "axisP[0]".
    i_term: [Option<f64>; 3],     // Integral term for each axis. Assumes CSV header like "axisI[0]".
    d_term: [Option<f64>; 3],     // Derivative term for each axis. Assumes CSV header like "axisD[0]".
    setpoint: [Option<f64>; 3],   // Target setpoint value for each axis. Assumes CSV header like "setpoint[0]".
    gyro: [Option<f64>; 3],       // Gyroscope readings (filtered assumed) for each axis. Assumes CSV header like "gyroADC[0]". Needed for step response.
}

// Define constants for plot dimensions
const PLOT_WIDTH: u32 = 1920;
const PLOT_HEIGHT: u32 = 1080;

// Helper function to calculate plot range with padding
fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    // Add a slightly larger padding for potentially smaller subplot value ranges
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

// Helper function to draw "Data Unavailable" message
fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
) -> Result<(), Box<dyn Error>> {
    let message = format!("Axis {} {} Data Unavailable", axis_index, plot_type);
    area.draw(&Text::new(
        message,
        (50, 50), // Position of the text within the subplot
        ("sans-serif", 20).into_font().color(&RED), // Style
    ))?;
    Ok(())
}


// --- START: Step Response Calculation Functions ---

fn fft_forward(data: &[f32]) -> Vec<Complex32> {
    // Ensure input is not empty
    if data.is_empty() {
        return Vec::new();
    }
    let mut input = data.to_vec();
    let n = input.len();
    if n == 0 { return Vec::new(); } // Handle empty input after potential filtering

    let planner = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut output = planner.make_output_vec();
    if planner.process(&mut input, &mut output).is_err() {
         eprintln!("Warning: FFT forward processing failed.");
         // Return empty or handle error appropriately
         let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
         return vec![Complex32::new(0.0, 0.0); expected_complex_len];
    }
    output
}

fn fft_inverse(data: &[Complex32], original_length_n: usize) -> Vec<f32> {
    // Ensure input is not empty and N is valid
    if data.is_empty() || original_length_n == 0 {
        return vec![0.0; original_length_n];
    }
    let mut input = data.to_vec();
    // The inverse planner needs the length of the original real signal (N)
    let planner = RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec(); // Output will have length N

    // Check if planner input length matches provided data length
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
        return vec![0.0; original_length_n];
    }

    if planner.process(&mut input, &mut output).is_ok() {
        // Normalize the IFFT output (realfft doesn't normalize by default)
        let scale = 1.0 / original_length_n as f32;
        output.iter_mut().for_each(|x| *x *= scale);
        output
    } else {
        // Error during processing
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        vec![0.0; original_length_n]
    }
}

// Helper function for moving average smoothing
fn moving_average_smooth(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_vec(); // No smoothing needed or possible
    }

    let mut smoothed_data = Vec::with_capacity(data.len());
    let mut current_sum: f32 = 0.0;
    // Using VecDeque to efficiently manage the sliding window sum
    let mut history: VecDeque<f32> = VecDeque::with_capacity(window_size);

    for &val in data.iter() {
        history.push_back(val);
        current_sum += val;

        // If the window is full, remove the oldest element from the sum and the deque
        if history.len() > window_size {
            // Use if let for safer pop
            if let Some(old_val) = history.pop_front() {
                 current_sum -= old_val;
            }
        }

        // Calculate the average over the current window contents
        // The effective window size grows until it reaches `window_size`
        let current_window_len = history.len() as f32;
        if current_window_len > 0.0 {
            smoothed_data.push(current_sum / current_window_len);
        } else {
            smoothed_data.push(0.0); // Should not happen if data is not empty
        }
    }

    smoothed_data
}


/// Calculates the step response from setpoint (input) and gyro (output) data.
///
/// Args:
/// * `times`: Slice of timestamps (f64).
/// * `setpoint`: Slice of setpoint values (f32).
/// * `gyro_filtered`: Slice of filtered gyroscope values (f32).
/// * `sample_rate`: Estimated sample rate of the data (f64).
///
/// Returns:
/// * `Vec<(f64, f64)>`: Vector of tuples (time_from_start, normalized_smoothed_step_response).
///   Returns an empty vector if inputs are invalid or calculation fails.
pub fn calculate_step_response(
    times: &[f64],
    setpoint: &[f32],
    gyro_filtered: &[f32],
    sample_rate: f64,
) -> Vec<(f64, f64)> {
    // Basic validation
    if times.is_empty() || setpoint.is_empty() || gyro_filtered.is_empty() || setpoint.len() != gyro_filtered.len() || times.len() != setpoint.len() || sample_rate <= 0.0 {
        eprintln!("Warning: Invalid input to calculate_step_response. Empty data, length mismatch, or invalid sample rate.");
        return Vec::new(); // Return empty if inputs are invalid
    }

    let n_samples = setpoint.len(); // Original number of samples (N)

    let input_spectrum = fft_forward(setpoint);
    let output_spectrum = fft_forward(gyro_filtered);

    // Ensure FFT outputs are compatible
    if input_spectrum.len() != output_spectrum.len() || input_spectrum.is_empty() {
         eprintln!("Warning: FFT outputs have different lengths or are empty. Cannot calculate frequency response.");
        return Vec::new();
    }

    let input_spec_conj: Vec<_> = input_spectrum.iter().map(|c| c.conj()).collect();

    // Calculate Frequency Response H(f) = (Input*(f) * Output(f)) / (Input*(f) * Input(f))
    // Add a small epsilon to denominator to avoid division by zero/very small numbers
    let epsilon = 1e-9;
    let frequency_response: Vec<_> = input_spectrum
        .iter()
        .zip(output_spectrum.iter())
        .zip(input_spec_conj.iter())
        .map(|((i, o), i_conj)| {
            let denominator = (i_conj * i).re.max(epsilon); // Use real part, ensure positive and non-zero
            (i_conj * o) / denominator
        })
        .collect();

    // Calculate Impulse Response (Inverse FFT of Frequency Response)
    // Pass the original signal length `n_samples`
    let impulse_response = fft_inverse(&frequency_response, n_samples);
    if impulse_response.is_empty() || impulse_response.len() != n_samples {
        eprintln!("Warning: Impulse response calculation failed or length mismatch.");
        return Vec::new();
    }


    // Calculate Step Response (Cumulative Sum of Impulse Response)
    let mut cumulative_sum = 0.0;
    let step_response: Vec<f32> = impulse_response
        .iter()
        .enumerate() // *** CORRECTED: Use enumerate to get index ***
        .map(|(index, &x)| { // *** CORRECTED: Get index here ***
            // Check for NaN/Inf in impulse response before summing
            if x.is_finite() {
                 cumulative_sum += x;
            } else {
                // *** CORRECTED: Use index from enumerate ***
                eprintln!("Warning: Non-finite value ({}) detected in impulse response at index {}. Skipping.", x, index);
                // Keep cumulative_sum as is, effectively skipping this value's contribution
            }
            cumulative_sum
        })
        .collect();


    // --- START: Smoothing Step ---
    // Define the desired smoothing duration in seconds (e.g., 10ms)
    let smoothing_duration_s = 0.01; // 10 ms
    // Calculate dynamic window size based on sample rate
    // Ensure window size is at least 1
    let window_size = ((smoothing_duration_s * sample_rate).round() as usize).max(1);

    // Apply moving average smoothing
    let smoothed_step_response = moving_average_smooth(&step_response, window_size);
    // --- END: Smoothing Step ---


    // Determine the number of samples corresponding to 500ms for truncation
    let num_points_500ms = (sample_rate * 0.5).ceil() as usize;
    // Ensure we don't take more points than available
    let truncated_len = num_points_500ms.min(smoothed_step_response.len());
    if truncated_len == 0 {
        eprintln!("Warning: Truncated step response length is zero.");
        return Vec::new(); // Nothing to normalize or return
    }

    // Calculate the average of the *first* `truncated_len` points of the SMOOTHED response
    // Avoid including potentially unstable later parts in the normalization average.
    let avg_sum: f32 = smoothed_step_response.iter().take(truncated_len).sum();
    // Ensure divisor is not zero
    let divisor = truncated_len as f32;
    let avg = if divisor > 0.0 { avg_sum / divisor } else { 1.0 }; // Default average to 1 if no data

    // Avoid division by zero or near-zero average for normalization
    let normalization_factor = if avg.abs() < 1e-7 {
        eprintln!("Warning: Near-zero average ({}) detected for step response normalization. Normalization might be inaccurate.", avg);
        1.0 // Avoid division by zero, but acknowledge potential issue
    } else {
        avg
    };


    let normalized_smoothed: Vec<f32> = smoothed_step_response
        .iter()
        .take(truncated_len) // limit to first 500ms (or available) after smoothing
        .map(|x| x / normalization_factor)
        .collect();

    let start_time = times.first().cloned().unwrap_or(0.0);

    // Combine time data with the smoothed, normalized, truncated step response
    times
        .iter()
        .zip(normalized_smoothed.into_iter()) // Use smoothed, normalized, truncated data
        .map(|(&t, s)| (t - start_time, s as f64)) // Adjust time to start from 0
        .collect()
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
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    // Define the specific CSV headers we are interested in extracting data from.
    let target_headers = [
        // Time (Essential for X-axis)
        "time (us)",    // 0: Base time unit, converted to seconds later.
        // P Term Components (Essential for PIDsum plot)
        "axisP[0]",     // 1: P term for Axis 0 (e.g., Roll)
        "axisP[1]",     // 2: P term for Axis 1 (e.g., Pitch)
        "axisP[2]",     // 3: P term for Axis 2 (e.g., Yaw)
        // I Term Components (Essential for PIDsum plot)
        "axisI[0]",     // 4: I term for Axis 0
        "axisI[1]",     // 5: I term for Axis 1
        "axisI[2]",     // 6: I term for Axis 2
        // D Term Components (Essential for PIDsum plot, except AxisD[2] which is optional)
        "axisD[0]",     // 7: D term for Axis 0
        "axisD[1]",     // 8: D term for Axis 1
        "axisD[2]",     // 9: D term for Axis 2 - Considered optional; defaults to 0.0 if header is missing.
        // Setpoint Components (Essential for Setpoint vs PIDsum plot and Step Response)
        "setpoint[0]",  // 10: Setpoint for Axis 0
        "setpoint[1]",  // 11: Setpoint for Axis 1
        "setpoint[2]",  // 12: Setpoint for Axis 2
        // Gyro Components (Essential for Step Response)
        "gyroADC[0]",   // 13: Gyro for Axis 0
        "gyroADC[1]",   // 14: Gyro for Axis 1
        "gyroADC[2]",   // 15: Gyro for Axis 2
    ];

    // Flags to track if specific optional or plot-specific headers are found in the CSV.
    let mut axis_d2_header_found = false; // Tracks if "axisD[2]" is present.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[x]" is present for each axis.
    let mut gyro_header_found = [false; 3]; // Tracks if "gyroADC[x]" is present for each axis.

    // Read the CSV header row and map the target headers to their actual column indices.
    let header_indices: Vec<Option<usize>> = {
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        let indices: Vec<Option<usize>> = target_headers
            .iter()
            .map(|&target_header| {
                header_record.iter().position(|h| h.trim() == target_header) // Trim headers from file just in case
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut essential_pid_headers_found = true; // For PIDsum plot

        // Check essential PID headers (0-8)
        for i in 0..=8 {
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_pid_headers_found = false;
                    format!("Not Found (Essential for PIDsum Plot!)")
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }

        // Check optional 'axisD[2]' (target index 9)
        let axis_d2_name = target_headers[9];
        let axis_d2_status = match indices[9] {
            Some(idx) => {
                axis_d2_header_found = true;
                format!("Found at index {}", idx)
            }
            None => format!("Not Found (Optional for PIDsum plot, will default to 0.0)"),
        };
        println!("  '{}' (Target Index {}): {}", axis_d2_name, 9, axis_d2_status);

        // Check setpoint headers (target indices 10, 11, 12)
        for axis in 0..3 {
            let target_idx = 10 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    setpoint_header_found[axis] = true;
                    format!("Found at index {}", idx)
                 }
                 None => {
                    format!("Not Found (Essential for Setpoint vs PIDsum Plot Axis {} AND Step Response Plot Axis {})", axis, axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }

        // Check gyro headers (target indices 13, 14, 15)
         for axis in 0..3 {
            let target_idx = 13 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    gyro_header_found[axis] = true;
                    format!("Found at index {}", idx)
                 }
                 None => {
                    format!("Not Found (Essential for Step Response Plot Axis {})", axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }


        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8)
                 .filter(|&i| indices[i].is_none())
                 .map(|i| format!("'{}'", target_headers[i]))
                 .collect();
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
        indices
    };

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading P/I/D term, Setpoint, and Gyro data from CSV...");
    {
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // --- Parse Time ---
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    // --- Parse P, I, D, Setpoint, Gyro for each axis ---
                    for axis in 0..3 {
                        // P term (target indices 1, 2, 3)
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        // I term (target indices 4, 5, 6)
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // D term (target indices 7, 8, 9)
                        let d_target_idx = 7 + axis;
                        if axis == 2 && !axis_d2_header_found {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        // Setpoint (target indices 10, 11, 12) - Parse only if header found
                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        }
                        // Gyro (target indices 13, 14, 15) - Parse only if header found
                        if gyro_header_found[axis] {
                             current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
                        }
                    }
                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    }

    println!("Finished reading {} data rows.", all_log_data.len());
    if !axis_d2_header_found {
        println!("INFO: 'axisD[2]' header was not found. Used 0.0 for Axis 2 D-term calculation in PIDsum plot.");
    }
    for axis in 0..3 {
        if !setpoint_header_found[axis] {
             println!("INFO: 'setpoint[{}]' header was not found. Setpoint vs PIDsum and Step Response plots for Axis {} cannot be generated.", axis, axis);
        }
         if !gyro_header_found[axis] {
             println!("INFO: 'gyroADC[{}]' header was not found. Step Response plot for Axis {} cannot be generated.", axis, axis);
        }
    }

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        for i in 1..all_log_data.len() {
            if let (Some(t1), Some(t0)) = (all_log_data[i].time_sec, all_log_data[i-1].time_sec) {
                let delta = t1 - t0;
                if delta > 1e-9 { // Avoid division by zero or issues with duplicate timestamps
                    total_delta += delta;
                    count += 1;
                }
            }
        }
        if count > 0 {
            let avg_delta = total_delta / count as f64;
            sample_rate = Some(1.0 / avg_delta);
            println!("Estimated Sample Rate: {:.2} Hz", sample_rate.unwrap());
        }
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might fail or be inaccurate.");
         // Proceed, but step response calculation will likely return empty.
    }


    // --- Data Preparation for Plots ---
    let mut pid_output_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_vs_pidsum_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // *** CORRECTED: Initialize the array of tuples correctly ***
    let mut step_response_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];
    let mut pid_data_available = [false; 3];
    let mut setpoint_data_available = [false; 3]; // For Setpoint vs PIDsum plot
    // Track if *input* data for step response is available (time, setpoint, gyro)
    let mut step_response_input_available = [false; 3];


    // First pass: Prepare data for PIDsum, SetpointVsPidsum, and collect inputs for Step Response
    for row in &all_log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                // PIDsum data
                if let (Some(p), Some(i), Some(d)) =
                    (row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    pid_output_data[axis_index].push((time, p + i + d));
                    pid_data_available[axis_index] = true;
                }

                // Setpoint vs PIDsum data
                if setpoint_header_found[axis_index] { // Only process if setpoint header exists
                    if let (Some(setpoint), Some(p), Some(i), Some(d)) =
                        (row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                    {
                        setpoint_vs_pidsum_data[axis_index].push((time, setpoint, p + i + d));
                        setpoint_data_available[axis_index] = true;
                    }
                }

                 // Step Response Input Data Collection
                 // Requires time, setpoint, and gyro for this specific row and axis
                 if setpoint_header_found[axis_index] && gyro_header_found[axis_index] {
                     if let (Some(setpoint), Some(gyro)) = (row.setpoint[axis_index], row.gyro[axis_index]) {
                        // *** CORRECTED: Access array element first, then tuple field ***
                        step_response_input_data[axis_index].0.push(time);
                        step_response_input_data[axis_index].1.push(setpoint as f32);
                        step_response_input_data[axis_index].2.push(gyro as f32);
                        step_response_input_available[axis_index] = true; // Mark that we have *some* input data
                     }
                 }
            }
        }
    }

    // --- Calculate Step Response Data ---
    println!("\n--- Calculating Step Response ---");
    let mut step_response_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut step_response_data_available = [false; 3]; // Track if calculation succeeded

    if let Some(sr) = sample_rate { // Only attempt calculation if sample rate is known
        for axis_index in 0..3 {
            if step_response_input_available[axis_index] {
                println!("  Calculating step response for Axis {}...", axis_index);
                 // *** CORRECTED: Access array element first, then tuple field ***
                let times = &step_response_input_data[axis_index].0;
                let setpoints = &step_response_input_data[axis_index].1;
                let gyros = &step_response_input_data[axis_index].2;

                // Ensure we have enough data points for meaningful FFT
                if times.len() > 10 { // Arbitrary minimum length, adjust if needed
                    let result = calculate_step_response(times, setpoints, gyros, sr);
                    if !result.is_empty() {
                        step_response_data[axis_index] = result;
                        step_response_data_available[axis_index] = true;
                        println!("    ... Calculation successful for Axis {}.", axis_index);
                    } else {
                        println!("    ... Calculation failed or returned empty for Axis {}.", axis_index);
                    }
                } else {
                     println!("    ... Skipping Axis {}: Not enough data points ({}) for step response calculation.", axis_index, times.len());
                }
            } else {
                 println!("  Skipping Axis {}: Missing required input data (Setpoint or Gyro).", axis_index);
            }
        }
    } else {
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Generate Stacked PIDsum Plot ---
    println!("\n--- Generating Stacked PIDsum Plot (All Axes) ---");
    if pid_data_available.iter().any(|&x| x) {
        let output_file_pidsum = format!("{}_PIDsum_stacked.png", root_name);
        let root_area_pidsum = BitMapBackend::new(&output_file_pidsum, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_pidsum.fill(&WHITE)?;
        let sub_plot_areas = root_area_pidsum.split_evenly((3, 1));
        let pidsum_plot_color = Palette99::pick(1); // Green

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];
            if pid_data_available[axis_index] && !pid_output_data[axis_index].is_empty() {
                let (time_min, time_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
                let (output_min, output_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));

                 if time_min.is_infinite() || output_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "PIDsum")?;
                     continue;
                 }

                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_pidsum_min, final_pidsum_max) = calculate_range(output_min, output_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} PIDsum (P+I+D)", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_pidsum_min..final_pidsum_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("PIDsum").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;
                chart.draw_series(LineSeries::new(
                    pid_output_data[axis_index].iter().cloned(),
                    &pidsum_plot_color,
                ))?;
            } else {
                println!("  INFO: No PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                draw_unavailable_message(area, axis_index, "PIDsum")?;
            }
        }
        root_area_pidsum.present()?;
        println!("  Stacked PIDsum plot saved as '{}'.", output_file_pidsum);
    } else {
        println!("  Skipping Stacked PIDsum Plot: No PIDsum data available for any axis.");
    }


    // --- Generate Stacked Setpoint vs PIDsum Plot ---
    println!("\n--- Generating Stacked Setpoint vs PIDsum Plot (All Axes) ---");
    if setpoint_data_available.iter().any(|&x| x) {
        let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
        let root_area_setpoint = BitMapBackend::new(&output_file_setpoint, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint.fill(&WHITE)?;
        let sub_plot_areas = root_area_setpoint.split_evenly((3, 1));
        let setpoint_plot_color = Palette99::pick(2); // Blue
        let pidsum_vs_setpoint_color = Palette99::pick(0); // Red

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if setpoint_data_available[axis_index] && !setpoint_vs_pidsum_data[axis_index].is_empty() {
                let (time_min, time_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| {
                        (min_y.min(*s).min(*p), max_y.max(*s).max(*p))
                    });

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
                     continue;
                 }

                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs PIDsum", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let sp_color_ref = &setpoint_plot_color;
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, s, _p)| (*t, *s)),
                    sp_color_ref,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color_ref));

                let pid_color_ref = &pidsum_vs_setpoint_color;
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, _s, p)| (*t, *p)),
                    pid_color_ref,
                ))?
                .label("PIDsum")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_color_ref));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                println!("  INFO: No Setpoint vs PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                 draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
            }
        }
        root_area_setpoint.present()?;
        println!("  Stacked Setpoint vs PIDsum plot saved as '{}'.", output_file_setpoint);
    } else {
        println!("  Skipping Stacked Setpoint vs PIDsum Plot: No Setpoint vs PIDsum data available for any axis.");
    }


    // --- Generate Stacked Step Response Plot ---
    println!("\n--- Generating Stacked Step Response Plot (All Axes) ---");
    if step_response_data_available.iter().any(|&x| x) { // Check if calculation succeeded for at least one axis
        let output_file_step = format!("{}_step_response_stacked.png", root_name);
        let root_area_step = BitMapBackend::new(&output_file_step, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_step.fill(&WHITE)?;
        let sub_plot_areas = root_area_step.split_evenly((3, 1));
        let step_response_color = Palette99::pick(3); // Choose a color (e.g., purple)

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];
            if step_response_data_available[axis_index] && !step_response_data[axis_index].is_empty() { // Add check for empty data
                // Determine ranges *specifically for this axis's step response*
                // Time range is already 0 to ~0.5s from calculation
                let (_time_min, time_max) = step_response_data[axis_index].iter()
                    // *** CORRECTED: Specify f64 type for initial fold value ***
                    .fold((0.0f64, 0.0f64), |(_min_t, max_t), (t, _)| (0.0, max_t.max(*t))); // Min time is always 0
                 // Value range (normalized response)
                let (resp_min, resp_max) = step_response_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));

                 if time_max <= 1e-9 || resp_min.is_infinite() { // Check if data range is valid (use small epsilon for time_max)
                     draw_unavailable_message(area, axis_index, "Step Response (Invalid Range)")?;
                     continue;
                 }

                // Apply padding to response range
                let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
                // Time range usually 0 to 0.5s, slight padding might be useful
                let final_time_max = time_max * 1.05;


                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Step Response", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(0f64..final_time_max, final_resp_min..final_resp_max)?; // Time starts at 0

                chart.configure_mesh()
                    .x_desc("Time (s)")
                    .y_desc("Normalized Response")
                    .x_labels(6) // Fewer labels suitable for 0-0.5s range
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7))
                    .label_style(("sans-serif", 12))
                    .draw()?;

                chart.draw_series(LineSeries::new(
                    step_response_data[axis_index].iter().cloned(),
                    &step_response_color,
                ))?;
                // No legend needed as title indicates the content

            } else {
                // Determine the reason for unavailability
                let reason = if !setpoint_header_found[axis_index] || !gyro_header_found[axis_index] {
                    "Setpoint/Gyro Header Missing"
                 } else if sample_rate.is_none() {
                    "Sample Rate Unknown"
                 } else if !step_response_input_available[axis_index] {
                     "Input Data Missing/Invalid"
                 } else { // This case means calculation was attempted but failed or returned empty
                     "Calculation Failed/No Data"
                 };
                println!("  INFO: No Step Response data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Step Response ({})", reason))?;
            }
        }
        root_area_step.present()?;
        println!("  Stacked Step Response plot saved as '{}'.", output_file_step);

    } else {
        println!("  Skipping Stacked Step Response Plot: No step response data could be calculated for any axis.");
    }

    Ok(())
}