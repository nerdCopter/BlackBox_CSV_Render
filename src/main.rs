use csv::ReaderBuilder; // For reading CSV files efficiently.
use plotters::prelude::*; // For creating plots and charts.
use std::error::Error; // Standard trait for error handling.
use std::env; // For accessing command-line arguments.
use std::path::Path; // For working with file paths.
use std::fs::File; // For file operations (opening files).
use std::io::BufReader; // For buffered reading, improving file I/O performance.
use std::collections::VecDeque; // For efficient moving average window.

// --- Dependencies for Step Response Calculation ---
use realfft::num_complex::Complex32; // Complex number type for FFT results.
use realfft::RealFftPlanner; // FFT planner for real-valued signals.

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


// --- START: Step Response Calculation Functions ---

/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
/// Returns the complex frequency spectrum. Handles empty input.
fn fft_forward(data: &[f32]) -> Vec<Complex32> {
    // Ensure input is not empty.
    if data.is_empty() {
        return Vec::new();
    }
    let mut input = data.to_vec(); // FFT library requires mutable buffer.
    let n = input.len();
    if n == 0 { return Vec::new(); } // Should be caught by first check, but safety.

    // Plan and execute the forward FFT.
    let planner = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut output = planner.make_output_vec(); // Output buffer for complex spectrum.
    if planner.process(&mut input, &mut output).is_err() {
         eprintln!("Warning: FFT forward processing failed.");
         // Return a vector of zeros with the expected complex length on failure.
         let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
         return vec![Complex32::new(0.0, 0.0); expected_complex_len];
    }
    output // Return the complex spectrum.
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of a complex spectrum.
/// Returns the reconstructed real-valued signal. Requires the original signal length N.
/// Normalizes the output. Handles empty input or length mismatches.
fn fft_inverse(data: &[Complex32], original_length_n: usize) -> Vec<f32> {
    // Ensure input is not empty and original length N is valid.
    if data.is_empty() || original_length_n == 0 {
        return vec![0.0; original_length_n]; // Return zeros matching original length.
    }
    let mut input = data.to_vec(); // IFFT library requires mutable buffer.
    // The inverse planner needs the length of the *original* real signal (N).
    let planner = RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec(); // Output buffer for real signal (length N).

    // Verify the complex input length matches expectations based on N.
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
        return vec![0.0; original_length_n]; // Return zeros matching original length.
    }

    // Execute the inverse FFT.
    if planner.process(&mut input, &mut output).is_ok() {
        // Normalize the IFFT output (realfft crate doesn't normalize automatically).
        let scale = 1.0 / original_length_n as f32;
        output.iter_mut().for_each(|x| *x *= scale);
        output // Return the reconstructed real signal.
    } else {
        // Error during IFFT processing.
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        vec![0.0; original_length_n] // Return zeros matching original length.
    }
}

/// Applies a moving average filter to smooth the input data.
/// Uses a `VecDeque` for efficient window management.
fn moving_average_smooth(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || data.is_empty() {
        return data.to_vec(); // No smoothing needed or possible.
    }

    let mut smoothed_data = Vec::with_capacity(data.len());
    let mut current_sum: f32 = 0.0;
    let mut history: VecDeque<f32> = VecDeque::with_capacity(window_size); // Window buffer.

    for &val in data.iter() {
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
            smoothed_data.push(current_sum / current_window_len);
        } else {
            smoothed_data.push(0.0); // Should not happen if data is not empty.
        }
    }
    smoothed_data
}


/// Calculates the system's step response using FFT-based deconvolution.
/// Input: System setpoint (input), gyro readings (output), timestamps, sample rate.
/// Output: Vector of (time_since_start, normalized_smoothed_step_response).
/// Steps:
/// 1. Calculate FFT of input (setpoint) and output (gyro).
/// 2. Calculate Frequency Response H(f) = (Input*(f) * Output(f)) / (|Input(f)|^2 + epsilon).
/// 3. Calculate Impulse Response = IFFT(H(f)).
/// 4. Calculate Step Response = Cumulative Sum(Impulse Response).
/// 5. Smooth the step response using a moving average.
/// 6. Truncate the response to 500ms.
/// 7. Normalize the response by the average value over the truncated duration.
pub fn calculate_step_response(
    times: &[f64],
    setpoint: &[f32],
    gyro_filtered: &[f32], // Assuming gyro data might be pre-filtered if needed.
    sample_rate: f64,
) -> Vec<(f64, f64)> {
    // Basic input validation.
    if times.is_empty() || setpoint.is_empty() || gyro_filtered.is_empty() || setpoint.len() != gyro_filtered.len() || times.len() != setpoint.len() || sample_rate <= 0.0 {
        eprintln!("Warning: Invalid input to calculate_step_response. Empty data, length mismatch, or invalid sample rate.");
        return Vec::new(); // Return empty vector on invalid input.
    }

    let n_samples = setpoint.len(); // Original number of samples (N).

    // Step 1: Calculate FFTs.
    let input_spectrum = fft_forward(setpoint);
    let output_spectrum = fft_forward(gyro_filtered);

    // Ensure FFT outputs are valid and compatible for frequency response calculation.
    if input_spectrum.len() != output_spectrum.len() || input_spectrum.is_empty() {
         eprintln!("Warning: FFT outputs have different lengths or are empty. Cannot calculate frequency response.");
        return Vec::new();
    }

    // Calculate complex conjugate of the input spectrum.
    let input_spec_conj: Vec<_> = input_spectrum.iter().map(|c| c.conj()).collect();

    // Step 2: Calculate Frequency Response H(f).
    // Add epsilon to denominator to prevent division by zero or very small numbers.
    let epsilon = 1e-9;
    let frequency_response: Vec<_> = input_spectrum
        .iter()
        .zip(output_spectrum.iter())
        .zip(input_spec_conj.iter())
        .map(|((i, o), i_conj)| {
            // Denominator is |Input(f)|^2 = Input*(f) * Input(f). Use real part, ensure positive.
            let denominator = (i_conj * i).re.max(epsilon);
            (i_conj * o) / denominator // H(f) = (Input*(f) * Output(f)) / Denominator
        })
        .collect();

    // Step 3: Calculate Impulse Response (IFFT of Frequency Response).
    // Provide the original signal length `n_samples` for correct IFFT.
    let impulse_response = fft_inverse(&frequency_response, n_samples);
    if impulse_response.is_empty() || impulse_response.len() != n_samples {
        eprintln!("Warning: Impulse response calculation failed or length mismatch.");
        return Vec::new();
    }

    // Step 4: Calculate Step Response (Cumulative Sum of Impulse Response).
    let mut cumulative_sum = 0.0;
    let step_response: Vec<f32> = impulse_response
        .iter()
        .enumerate() // Get index for warning messages.
        .map(|(index, &x)| {
            // Check for NaN/Inf in impulse response before summing to avoid propagating errors.
            if x.is_finite() {
                 cumulative_sum += x;
            } else {
                // Report non-finite values found during summation.
                eprintln!("Warning: Non-finite value ({}) detected in impulse response at index {}. Skipping.", x, index);
                // Keep cumulative_sum as is, effectively skipping this non-finite value.
            }
            cumulative_sum // Current value of the cumulative sum.
        })
        .collect();


    // Step 5: Smooth the Step Response.
    // Define smoothing window duration (e.g., 10ms).
    let smoothing_duration_s = 0.01; // 10 ms.
    // Calculate moving average window size based on sample rate. Ensure window size is at least 1.
    let window_size = ((smoothing_duration_s * sample_rate).round() as usize).max(1);
    let smoothed_step_response = moving_average_smooth(&step_response, window_size);


    // Step 6: Truncate the response to the first 500ms.
    // Determine the number of samples corresponding to 500ms.
    let num_points_500ms = (sample_rate * 0.5).ceil() as usize;
    // Ensure truncation length doesn't exceed available data length.
    let truncated_len = num_points_500ms.min(smoothed_step_response.len());
    if truncated_len == 0 {
        eprintln!("Warning: Truncated step response length is zero.");
        return Vec::new(); // Cannot proceed if truncation results in zero length.
    }

    // Step 7: Normalize the smoothed, truncated response.
    // Calculate average value of the *smoothed* response over the *truncated* duration.
    // This average represents the approximate steady-state value for normalization.
    let avg_sum: f32 = smoothed_step_response.iter().take(truncated_len).sum();
    // Ensure divisor is not zero.
    let divisor = truncated_len as f32;
    let avg = if divisor > 0.0 { avg_sum / divisor } else { 1.0 }; // Default average to 1 if no data points.

    // Avoid division by zero or near-zero average during normalization.
    let normalization_factor = if avg.abs() < 1e-7 {
        eprintln!("Warning: Near-zero average ({}) detected for step response normalization. Normalization might be inaccurate.", avg);
        1.0 // Use 1.0 to avoid division by zero, results might be unnormalized.
    } else {
        avg
    };

    // Apply normalization to the smoothed, truncated data.
    let normalized_smoothed: Vec<f32> = smoothed_step_response
        .iter()
        .take(truncated_len) // Use data truncated to 500ms (or available).
        .map(|x| x / normalization_factor) // Normalize each point.
        .collect();

    // Get the starting timestamp from the original time data.
    let start_time = times.first().cloned().unwrap_or(0.0);

    // Combine time data (adjusted to start from 0) with the processed step response values.
    times
        .iter()
        .zip(normalized_smoothed.into_iter()) // Pair original times with processed response data.
        .map(|(&t, s)| (t - start_time, s as f64)) // Adjust time to start from 0, cast response to f64.
        .take(truncated_len) // Ensure output length matches truncated response length.
        .collect() // Return the final (time, value) pairs.
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

    // --- Calculate Step Response Data ---
    println!("\n--- Calculating Step Response ---");
    // Stores the final calculated step response data: (time_from_start, normalized_response).
    let mut step_response_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Tracks if the step response *calculation* was successful and produced data for each axis.
    let mut step_response_data_available = [false; 3];

    if let Some(sr) = sample_rate { // Only proceed if sample rate was determined.
        for axis_index in 0..3 {
            // Only calculate if the required input data (time, setpoint, gyro_filtered) was collected earlier.
            if step_response_input_available[axis_index] {
                println!("  Calculating step response for Axis {}...", axis_index);
                 // Get references to the input data for this axis.
                let times = &step_response_input_data[axis_index].0;
                let setpoints = &step_response_input_data[axis_index].1;
                let gyros_filtered = &step_response_input_data[axis_index].2; // Use filtered gyro data

                // Check if there are enough data points for a meaningful FFT analysis.
                if times.len() > 10 { // Use an arbitrary minimum length (e.g., 10 points). Adjust if needed.
                    let result = calculate_step_response(times, setpoints, gyros_filtered, sr);
                    // Check if the calculation succeeded and returned non-empty results.
                    if !result.is_empty() {
                        step_response_data[axis_index] = result; // Store the calculated data.
                        step_response_data_available[axis_index] = true; // Mark calculation as successful for this axis.
                        println!("    ... Calculation successful for Axis {}.", axis_index);
                    } else {
                        // Calculation function might return empty on internal errors or invalid intermediate results.
                        println!("    ... Calculation failed or returned empty for Axis {}.", axis_index);
                    }
                } else {
                     // Not enough data points for FFT.
                     println!("    ... Skipping Axis {}: Not enough data points ({}) for step response calculation.", axis_index, times.len());
                }
            } else {
                 // Input data was missing (Setpoint or Gyro (filtered) header likely not found).
                 println!("  Skipping Axis {}: Missing required input data (Setpoint or Filtered Gyro 'gyroADC').", axis_index);
            }
        }
    } else {
         // Sample rate is unknown, cannot perform FFT-based calculation.
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
    // Check if step response *calculation was successful* for at least one axis.
    if step_response_data_available.iter().any(|&x| x) {
        let output_file_step = format!("{}_step_response_stacked.png", root_name);
        let root_area_step = BitMapBackend::new(&output_file_step, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_step.fill(&WHITE)?;
        let sub_plot_areas = root_area_step.split_evenly((3, 1));
        let step_response_color = Palette99::pick(3); // Consistent color for step response.

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];
            // Check if calculation succeeded AND resulted in non-empty data for this axis.
            if step_response_data_available[axis_index] && !step_response_data[axis_index].is_empty() {
                // Find plot ranges specifically for the calculated step response data.
                // Time range starts at 0 and goes up to the maximum time in the truncated data (around 0.5s).
                let (_time_min, time_max) = step_response_data[axis_index].iter()
                    // Fold requires explicit type annotation for initial accumulator value.
                    .fold((0.0f64, 0.0f64), |(_min_t, max_t), (t, _)| (0.0, max_t.max(*t))); // Min time is always 0.
                 // Value range is the min/max of the normalized response.
                let (resp_min, resp_max) = step_response_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));

                 // Check if the calculated ranges are valid before plotting.
                 if time_max <= 1e-9 || resp_min.is_infinite() { // Check time > 0 and finite value range.
                     draw_unavailable_message(area, axis_index, "Step Response (Invalid Range)")?;
                     continue;
                 }

                // Apply padding to the response range.
                let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
                // Add slight padding to the maximum time.
                let final_time_max = time_max * 1.05;

                // Build the chart. Time axis starts at 0.
                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Step Response", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(0f64..final_time_max, final_resp_min..final_resp_max)?;

                // Configure mesh and labels, using fewer X labels suitable for the short time range (0-0.5s).
                chart.configure_mesh()
                    .x_desc("Time (s)")
                    .y_desc("Normalized Response")
                    .x_labels(6) // Fewer labels for 0-0.5s range.
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7))
                    .label_style(("sans-serif", 12))
                    .draw()?;

                // Draw the step response data.
                chart.draw_series(LineSeries::new(
                    step_response_data[axis_index].iter().cloned(),
                    &step_response_color,
                ))?;
                // No legend needed as the title identifies the single series.

            } else {
                // Step response data is unavailable for this axis. Determine the reason.
                let reason = if !setpoint_header_found[axis_index] || !gyro_header_found[axis_index] {
                    "Setpoint/gyroADC Header Missing" // Missing essential headers for step response.
                 } else if sample_rate.is_none() {
                    "Sample Rate Unknown" // Sample rate couldn't be estimated.
                 } else if !step_response_input_available[axis_index] {
                     "Input Data Missing/Invalid" // Headers found, but no valid data rows.
                 } else { // Headers present, sample rate known, input data collected, but calculation failed/returned empty.
                     "Calculation Failed/No Data"
                 };
                println!("  INFO: No Step Response data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 // Draw placeholder message including the reason.
                 draw_unavailable_message(area, axis_index, &format!("Step Response ({})", reason))?;
            }
        }
        root_area_step.present()?; // Save the plot file.
        println!("  Stacked Step Response plot saved as '{}'.", output_file_step);

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