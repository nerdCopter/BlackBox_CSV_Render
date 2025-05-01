// src/main.rs

mod log_data;
mod constants;
mod plotting_utils;
mod step_response;

use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

// Removed: use ndarray::s; // Import the s! macro for slicing - It's not used in main.rs
use ndarray::{Array1, Array2}; // Import Array2
use ndarray_stats::QuantileExt; // Import QuantileExt for .min() and .max() on Array1

use log_data::LogRowData;
use constants::*;
use plotting_utils::*;
use step_response::*;

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file.csv>", args[0]);
        std::process::exit(1);
    }
    let input_file = &args[1];
    let input_path = Path::new(input_file);
    println!("Reading {}", input_file);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    let target_headers = [
        "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]",
        "axisI[0]", "axisI[1]", "axisI[2]",
        "axisD[0]", "axisD[1]", "axisD[2]",
        "setpoint[0]", "setpoint[1]", "setpoint[2]",
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]",
        "debug[0]", "debug[1]", "debug[2]", "debug[3]",
        "throttle",
    ];

    // Flags to track if specific optional or plot-dependent headers are found.
    // These are used to decide which plots/calculations are possible.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[axis]" is present.
    let mut gyro_header_found = [false; 3]; // Tracks if "gyroADC[axis]" is present (filtered gyro).
    let mut gyro_unfilt_header_found = [false; 3]; // Tracks if "gyroUnfilt[axis]" is present.
    let mut debug_header_found = [false; 4]; // Tracks if "debug[idx]" is present.

    // Declare header_indices here so it's accessible outside the block
    let header_indices: Vec<Option<usize>>;

    // Read CSV header and map target headers to indices.
    { // Block to limit the scope of the file reader
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));
        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        header_indices = target_headers.iter().map(|&target_header| {
            header_record.iter().position(|h| h.trim() == target_header)
        }).collect();

        println!("Header mapping status:");
        let mut essential_pid_headers_found = true;

        // Check essential PID headers (Time, P, I, D[0], D[1]).
        for i in 0..=8 { // Indices 0 through 8
            let name = target_headers[i];
             let found = header_indices[i].is_some(); // Use header_indices here
             println!("  '{}': {}", name, if found { "Found" } else { "Not Found" });
             if !found { // Time (0), P (1-3), I (4-6), D[0-1] (7-8) are essential for PIDsum
                 essential_pid_headers_found = false;
             }
        }

        // Check optional 'axisD[2]' header (Index 9).
        let axis_d2_found_in_csv = header_indices[9].is_some(); // Use header_indices here
        println!("  '{}': {} (Optional, defaults to 0.0 if not found)", target_headers[9], if axis_d2_found_in_csv { "Found" } else { "Not Found" });

        // Check setpoint headers (Indices 10-12).
        for axis in 0..3 {
            setpoint_header_found[axis] = header_indices[10 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Setpoint vs PIDsum, Setpoint vs Gyro, and Step Response Axis {})", target_headers[10 + axis], if setpoint_header_found[axis] { "Found" } else { "Not Found" }, axis); // Updated message
        }

        // Check gyro (filtered) headers (Indices 13-15).
         for axis in 0..3 {
            gyro_header_found[axis] = header_indices[13 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Step Response, Setpoint vs Gyro, and Gyro vs Unfilt Axis {})", target_headers[13 + axis], if gyro_header_found[axis] { "Found" } else { "Not Found" }, axis); // Updated message
        }

        // Check gyroUnfilt headers (Indices 16-18).
        for axis in 0..3 {
            gyro_unfilt_header_found[axis] = header_indices[16 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for Gyro vs Unfilt Axis {})", target_headers[16 + axis], if gyro_unfilt_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check debug headers (Indices 19-22).
        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[19 + idx_offset].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for gyroUnfilt[0-2])", target_headers[19 + idx_offset], if debug_header_found[idx_offset] { "Found" } else { "Not Found" });
        }

        // Check throttle header (Index 23).
        let throttle_found_in_csv = header_indices[23].is_some(); // Use header_indices directly
        println!("  '{}': {} (Optional, for filtering)", target_headers[23], if throttle_found_in_csv { "Found" } else { "Not Found" });


        // Check if at least time, P, I, D[0], D[1] are found
        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8).filter(|&i| header_indices[i].is_none()).map(|i| format!("'{}'", target_headers[i])).collect(); // Use header_indices here
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
    } // File reader is dropped here

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading data rows...");
    { // Block to limit the scope of the file reader
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    // Helper closure to parse a value from the record using the target header index.
                    // Uses the header_indices vector which is now in scope.
                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // Parse Time (us)
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    // Parse P, I, D, Setpoint, Gyro (filtered)
                    for axis in 0..3 {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // D term with optional axisD[2] fallback
                        let d_target_idx = 7 + axis;
                        // Use header_indices directly
                        if axis == 2 && header_indices[d_target_idx].is_none() {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        }
                        if gyro_header_found[axis] {
                             current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
                        }
                    }

                    // Parse gyroUnfilt and debug
                    let mut parsed_gyro_unfilt = [None; 3];
                    let mut parsed_debug = [None; 4];

                    for axis in 0..3 {
                        if gyro_unfilt_header_found[axis] {
                            parsed_gyro_unfilt[axis] = parse_f64_by_target_idx(16 + axis);
                        }
                    }

                    for idx_offset in 0..4 {
                        if debug_header_found[idx_offset] {
                            parsed_debug[idx_offset] = parse_f64_by_target_idx(19 + idx_offset);
                        }
                        current_row_data.debug[idx_offset] = parsed_debug[idx_offset];
                    }

                    // Apply Fallback Logic for gyro_unfilt
                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => match parsed_debug[axis] {
                                Some(val) => Some(val),
                                None => Some(0.0), // Default to 0.0 if both missing
                            }
                        };
                    }

                    // Parse Throttle
                    // Use header_indices directly
                    if header_indices[23].is_some() {
                        current_row_data.throttle = parse_f64_by_target_idx(23);
                    }

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // File reader is dropped here


    println!("Finished reading {} data rows.", all_log_data.len());

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    let mut first_time: Option<f64> = None;
    let mut last_time: Option<f64> = None;

    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        for i in 1..all_log_data.len() {
            if let (Some(t1), Some(t0)) = (all_log_data[i].time_sec, all_log_data[i-1].time_sec) {
                let delta = t1 - t0;
                if delta > 1e-9 {
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

        first_time = all_log_data.first().and_then(|row| row.time_sec);
        last_time = all_log_data.last().and_then(|row| row.time_sec);
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might fail.");
    }

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Data Preparation for Plots ---
    let mut pid_output_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_vs_pidsum_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut gyro_vs_unfilt_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_vs_gyro_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()]; // This one is used for PID Error source
    let mut step_response_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    let mut pid_data_available = [false; 3]; // Tracks if P, I, D data exists for PIDsum
    let mut setpoint_data_available = [false; 3]; // Tracks if Setpoint data exists (used for Setpoint vs PIDsum plot availability)
    let mut gyro_vs_unfilt_data_available = [false; 3]; // Tracks if data exists for Gyro vs Unfilt plot
    let mut setpoint_vs_gyro_data_available = [false; 3]; // Tracks if data exists for Setpoint vs Gyro plot AND PID Error source
    let mut step_response_input_available = [false; 3]; // Tracks if data exists for Step Response calc


    // Populate plot data structures and step response input data.
    for row in &all_log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                // PIDsum data
                if let (Some(p), Some(i), Some(d)) = (row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index]) {
                    pid_output_data[axis_index].push((time, p + i + d));
                    pid_data_available[axis_index] = true;
                }

                // Setpoint vs PIDsum data (requires Setpoint, P, I, D)
                if setpoint_header_found[axis_index] && pid_data_available[axis_index] { // Ensure PID data is also available
                    if let (Some(setpoint), Some(p), Some(i), Some(d)) = (row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index]) {
                        setpoint_vs_pidsum_data[axis_index].push((time, setpoint, p + i + d));
                        setpoint_data_available[axis_index] = true;
                    }
                }

                 // Setpoint vs Gyro data (requires Setpoint and GyroADC). Used for SetpointVsGyro plot AND PID Error.
                 if setpoint_header_found[axis_index] && gyro_header_found[axis_index] {
                    if let (Some(setpoint), Some(gyro_filt)) = (row.setpoint[axis_index], row.gyro[axis_index]) {
                        setpoint_vs_gyro_data[axis_index].push((time, setpoint, gyro_filt));
                        setpoint_vs_gyro_data_available[axis_index] = true;
                    }
                 }

                 // Step Response Input Data (filtered by time and movement, requires Setpoint and GyroADC)
                 if setpoint_header_found[axis_index] && gyro_header_found[axis_index] {
                     if let (Some(setpoint), Some(gyro_filt)) = (row.setpoint[axis_index], row.gyro[axis_index]) {
                         let mut include_point = false;
                         if let (Some(first), Some(last)) = (first_time, last_time) {
                             if time >= first + EXCLUDE_START_S && time <= last - EXCLUDE_END_S {
                                 if setpoint.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_filt.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                     include_point = true;
                                 }
                             }
                         } else { // Include if time range unknown, based on movement only
                             if setpoint.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_filt.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                  include_point = true;
                             }
                         }

                         if include_point {
                             step_response_input_data[axis_index].0.push(time);
                             step_response_input_data[axis_index].1.push(setpoint as f32);
                             step_response_input_data[axis_index].2.push(gyro_filt as f32);
                             step_response_input_available[axis_index] = true;
                         }
                     }
                 }

                 // Gyro vs Unfiltered Gyro Data (requires GyroADC and GyroUnfilt/Debug)
                 if gyro_header_found[axis_index] && row.gyro_unfilt[axis_index].is_some() { // Ensure gyroUnfilt/debug fallback was successful
                     if let (Some(gyro_filt), Some(gyro_unfilt)) = (row.gyro[axis_index], row.gyro_unfilt[axis_index]) {
                         gyro_vs_unfilt_data[axis_index].push((time, gyro_filt, gyro_unfilt));
                         gyro_vs_unfilt_data_available[axis_index] = true;
                     }
                 }
            }
        }
    }


    // --- Calculate Step Response Data ---
    println!("\n--- Calculating Step Response ---");
    // Store the raw QC'd stacked responses and setpoints for later averaging in main
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];


     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            if step_response_input_available[axis_index] {
                println!("  Calculating step response for Axis {}...", axis_index);
                let time_arr = Array1::from(step_response_input_data[axis_index].0.clone());
                let setpoints_arr = Array1::from(step_response_input_data[axis_index].1.clone());
                let gyros_filtered_arr = Array1::from(step_response_input_data[axis_index].2.clone());

                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                if time_arr.len() >= min_required_samples {
                    // Call the calculation function which now returns stacked QC'd responses
                    match calculate_step_response_python_style(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) {
                        Ok(result) => {
                             let num_qc_windows = result.1.shape()[0]; // Check number of QC windows
                             if num_qc_windows > 0 {
                                 step_response_calculation_results[axis_index] = Some(result); // Store the calculated data.
                                 println!("    ... Calculation successful for Axis {}. {} windows passed QC.", axis_index, num_qc_windows);
                             } else {
                                println!("    ... Calculation returned no valid windows for Axis {}. Skipping.", axis_index);
                             }
                        }
                        Err(e) => {
                            eprintln!("    ... Calculation failed for Axis {}: {}", axis_index, e);
                        }
                    }
                } else {
                     println!("    ... Skipping Axis {}: Not enough movement data points ({}) for windowing (need at least {}).", axis_index, time_arr.len(), min_required_samples);
                }
            } else {
                 println!("  Skipping Axis {}: Missing required input data (Setpoint or Filtered Gyro 'gyroADC').", axis_index);
            }
        }
    } else {
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Generate Stacked PIDsum & PID Error Plot --- //
    println!("\n--- Generating Stacked PIDsum & PID Error Plot ---");
    // Check if *any* axis has PIDsum data OR Setpoint/Gyro data (for PID Error)
    if pid_data_available.iter().any(|&x| x) || setpoint_vs_gyro_data_available.iter().any(|&x| x) {
        let output_file_pidsum_error = format!("{}_PIDsum_and_Error_stacked.png", root_name); //
        // Create the main drawing area for the entire plot
        let root_area_pidsum_error = BitMapBackend::new(&output_file_pidsum_error, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_pidsum_error.fill(&WHITE)?;

        // Add main title on the full drawing area
        root_area_pidsum_error.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;

        // Create a margined area below the title for the subplots
        let margined_root_area_pidsum_error = root_area_pidsum_error.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_pidsum_error.split_evenly((3, 1));

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];

            let has_pidsum_data_initial = pid_data_available[axis_index] && !pid_output_data[axis_index].is_empty();
            // Check if setpoint_vs_gyro_data was originally available for calculating error
            let setpoint_gyro_headers_found = setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            let has_setpoint_gyro_data_initial = setpoint_vs_gyro_data_available[axis_index] && !setpoint_vs_gyro_data[axis_index].is_empty();

            // Flags will be updated based on whether filtered data is empty
            let mut has_pidsum_data = has_pidsum_data_initial;
            let mut has_setpoint_gyro_data = has_setpoint_gyro_data_initial;

            // Only attempt to proceed if at least one data source was initially available
            if has_pidsum_data_initial || has_setpoint_gyro_data_initial {

                // Determine time range from available data (using initial checks)
                 let time_min = if has_pidsum_data_initial && !pid_output_data[axis_index].is_empty() {
                                    pid_output_data[axis_index][0].0
                                } else if has_setpoint_gyro_data_initial && !setpoint_vs_gyro_data[axis_index].is_empty() {
                                    setpoint_vs_gyro_data[axis_index][0].0
                                } else {
                                    // This case should theoretically not be reached due to the outer if,
                                    // but provide a safe default if somehow data flags are true but vecs are empty.
                                    0.0
                                };
                 let time_max = if has_pidsum_data_initial && !pid_output_data[axis_index].is_empty() {
                                    pid_output_data[axis_index].last().map(|p| p.0).unwrap_or(time_min)
                                } else if has_setpoint_gyro_data_initial && !setpoint_vs_gyro_data[axis_index].is_empty() {
                                    setpoint_vs_gyro_data[axis_index].last().map(|p| p.0).unwrap_or(time_min)
                                } else {
                                     // Fallback
                                     time_min + 1.0
                                };


                let mut pid_error_data: Vec<(f64, f64)> = Vec::new();
                let mut error_min = f64::INFINITY;
                let mut error_max = f64::NEG_INFINITY;

                // Calculate PID Error data if Setpoint/Gyro headers were found AND data is available
                if setpoint_gyro_headers_found && has_setpoint_gyro_data_initial { // Use the header check here too
                    pid_error_data = setpoint_vs_gyro_data[axis_index].iter()
                        .filter_map(|(t, sp, gyro)| {
                             // Ensure both setpoint and gyro are valid numbers
                             if sp.is_finite() && gyro.is_finite() {
                                 let error = sp - gyro;
                                 error_min = error_min.min(error);
                                 error_max = error_max.max(error);
                                 Some((*t, error))
                             } else {
                                 None
                             }
                        })
                        .collect();
                     // Update flag based on whether valid error data was collected
                     has_setpoint_gyro_data = !pid_error_data.is_empty();
                } else {
                     // If headers weren't found or data wasn't initially available, ensure the flag is false
                     has_setpoint_gyro_data = false;
                }


                let mut output_min = f64::INFINITY;
                let mut output_max = f64::NEG_INFINITY;

                // Calculate min/max for PIDsum data if available
                if has_pidsum_data_initial { // Check if pid_output_data was initially available
                     for &(_, val) in &pid_output_data[axis_index] {
                         if val.is_finite() {
                             output_min = output_min.min(val);
                             output_max = output_max.max(val);
                         }
                     }
                     // Update flag based on whether valid pidsum data exists after filtering
                     has_pidsum_data = output_min.is_finite();
                } else {
                     // If data wasn't initially available, ensure the flag is false
                     has_pidsum_data = false;
                }

                // Recalculate combined min/max only if at least one dataset is still valid
                 if has_pidsum_data || has_setpoint_gyro_data {
                    // Start min/max from infinity/neg_infinity to ensure any value sets it
                    let mut combined_min = f64::INFINITY;
                    let mut combined_max = f64::NEG_INFINITY;

                    if has_pidsum_data {
                         combined_min = combined_min.min(output_min);
                         combined_max = combined_max.max(output_max);
                    }
                    if has_setpoint_gyro_data {
                         combined_min = combined_min.min(error_min);
                         combined_max = combined_max.max(error_max);
                    }

                    // Only proceed to draw the chart if combined min/max are not infinite
                    if combined_min.is_finite() && combined_max.is_finite() {
                         let (final_y_min, final_y_max) = calculate_range(combined_min, combined_max);

                         let mut chart = ChartBuilder::on(area)
                             .caption(format!("Axis {} PID (PIDsum & Error)", axis_index), ("sans-serif", 20)) // Use updated caption
                             .margin(5).x_label_area_size(30).y_label_area_size(50)
                             .build_cartesian_2d(time_min..time_max, final_y_min..final_y_max)?; // Use combined range
                         chart.configure_mesh().x_desc("Time (s)").y_desc("Value (deg/s)").x_labels(10).y_labels(5) // Use updated y-desc
                             .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                         let mut legend_drawn = false;

                         // Draw PIDsum series if available
                         if has_pidsum_data {
                             let pidsum_color = Palette99::pick(COLOR_PIDSUM);
                             chart.draw_series(LineSeries::new(
                                 pid_output_data[axis_index].iter().cloned(),
                                 &pidsum_color,
                             ))?
                             .label("PIDsum (P+I+D)")
                             .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pidsum_color.stroke_width(2)));
                             legend_drawn = true;
                         }

                         // Draw PID Error series if available
                         if has_setpoint_gyro_data {
                              let pid_error_color = Palette99::pick(COLOR_PIDERROR); // Use the new color
                              chart.draw_series(LineSeries::new(
                                  pid_error_data.into_iter(), // Use the calculated error data
                                  &pid_error_color,
                              ))?
                              .label("PID Error (Setpoint - Gyro)") // Label for the new series
                              .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_error_color.stroke_width(2)));
                              legend_drawn = true;
                         }

                         // Configure and draw legend if any series was drawn
                         if legend_drawn {
                              chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                                  .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
                         }
                    } else {
                         // After filtering, no valid data remains for this axis plot
                         println!("  INFO: No valid PIDsum or Setpoint/Gyro data found for Axis {}. Drawing placeholder.", axis_index);
                         draw_unavailable_message(area, axis_index, "PIDsum/PIDerror (No Valid Data)")?;
                    }


                 } else {
                     // After filtering, no valid data remains for this axis plot
                     println!("  INFO: No valid PIDsum or Setpoint/Gyro data found for Axis {}. Drawing placeholder.", axis_index);
                     draw_unavailable_message(area, axis_index, "PIDsum/PIDerror (No Valid Data)")?;
                 }

            } else {
                // Neither PIDsum nor Setpoint/Gyro data is available for this axis
                println!("  INFO: No PIDsum or Setpoint/Gyro header/data available for Axis {}. Drawing placeholder.", axis_index);
                let reason = if !pid_data_available[axis_index] && !(setpoint_header_found[axis_index] && gyro_header_found[axis_index]) {
                    "Headers Missing" // Refined missing header check
                 } else {
                    "No Valid Data Rows"
                 };
                draw_unavailable_message(area, axis_index, &format!("PIDsum/PIDerror ({})", reason))?;
            }
        }
        root_area_pidsum_error.present()?; // Use the correct root area variable
        println!("  Stacked PIDsum & PID Error plot saved as '{}'.", output_file_pidsum_error); //

    } else {
        println!("  Skipping Stacked PIDsum & PID Error Plot: No PIDsum or Setpoint/Gyro data available for any axis."); //
    }

    // --- Generate Stacked Setpoint vs PIDsum Plot ---
    println!("\n--- Generating Stacked Setpoint vs PIDsum Plot ---");
    if setpoint_data_available.iter().any(|&x| x) {
        let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_setpoint = BitMapBackend::new(&output_file_setpoint, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_setpoint.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_setpoint = root_area_setpoint.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_setpoint.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if setpoint_data_available[axis_index] && !setpoint_vs_pidsum_data[axis_index].is_empty() {
                let (time_min, time_max) = setpoint_vs_pidsum_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_pidsum_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| (min_y.min(*s).min(*p), max_y.max(*s).max(*p)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max); // Corrected typo here

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs PIDsum", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let sp_color = Palette99::pick(COLOR_SETPOINT);
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, s, _p)| (*t, *s)),
                    &sp_color,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color.stroke_width(2)));

                let pid_color = Palette99::pick(COLOR_PIDSUM_VS_SETPOINT);
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, _s, p)| (*t, *p)),
                    &pid_color,
                ))?
                .label("PIDsum")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_color.stroke_width(2)));

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

    // --- Generate Stacked Setpoint vs Gyro Plot ---
    println!("\n--- Generating Stacked Setpoint vs Gyro Plot ---");
    if setpoint_vs_gyro_data_available.iter().any(|&x| x) {
        let output_file_setpoint_gyro = format!("{}_SetpointVsGyro_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_setpoint_gyro = BitMapBackend::new(&output_file_setpoint_gyro, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint_gyro.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_setpoint_gyro.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_setpoint_gyro = root_area_setpoint_gyro.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_setpoint_gyro.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if setpoint_vs_gyro_data_available[axis_index] && !setpoint_vs_gyro_data[axis_index].is_empty() {
                let (time_min, time_max) = setpoint_vs_gyro_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_gyro_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, g)| (min_y.min(*s).min(*g), max_y.max(*s).max(*g)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/Gyro")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs Gyro", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                    let gyro_color = Palette99::pick(COLOR_STEP_RESPONSE_LOW_SP); // Use the same blue/green as low setpoint step response
                    chart.draw_series(LineSeries::new(
                        setpoint_vs_gyro_data[axis_index].iter().map(|(t, _s, g)| (*t, *g)),
                        &gyro_color,
                    ))?
                    .label("Gyro (gyroADC)")
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], gyro_color.stroke_width(2)));

                    let sp_color = COLOR_STEP_RESPONSE_HIGH_SP; // Use the same orange as high setpoint step response
                chart.draw_series(LineSeries::new(
                    setpoint_vs_gyro_data[axis_index].iter().map(|(t, s, _g)| (*t, *s)),
                    &sp_color,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color.stroke_width(2)));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                println!("  INFO: No Setpoint vs Gyro data available for Axis {}. Drawing placeholder.", axis_index);
                 draw_unavailable_message(area, axis_index, "Setpoint/Gyro")?;
            }
        }
        root_area_setpoint_gyro.present()?;
        println!("  Stacked Setpoint vs Gyro plot saved as '{}'.", output_file_setpoint_gyro);
    } else {
        println!("  Skipping Stacked Setpoint vs Gyro Plot: No Setpoint vs Gyro data available for any axis.");
    }

    // --- Generate Stacked Gyro vs Unfiltered Gyro Plot ---
    println!("\n--- Generating Stacked Gyro vs Unfiltered Gyro Plot ---");
    if gyro_vs_unfilt_data_available.iter().any(|&x| x) {
        let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_gyro = BitMapBackend::new(&output_file_gyro, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_gyro.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_gyro.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_gyro = root_area_gyro.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_gyro.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if gyro_vs_unfilt_data_available[axis_index] && !gyro_vs_unfilt_data[axis_index].is_empty() {
                let (time_min, time_max) = gyro_vs_unfilt_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = gyro_vs_unfilt_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, gf, gu)| (min_y.min(*gf).min(*gu), max_y.max(*gf).max(*gu)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Gyro/UnfiltGyro")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Filtered vs Unfiltered Gyro", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Gyro Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let unfilt_color = Palette99::pick(COLOR_GYRO_UNFILT); // Removed .mix(0.6) for full opacity
                chart.draw_series(LineSeries::new(
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, _gf, gu)| (*t, *gu)),
                    &unfilt_color,
                ))?
                .label("Unfiltered Gyro (gyroUnfilt/debug)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], unfilt_color.stroke_width(2)));

                let filt_color = Palette99::pick(COLOR_GYRO_FILT).filled();
                chart.draw_series(LineSeries::new(
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, gf, _gu)| (*t, *gf)),
                    filt_color,
                ))?
                .label("Filtered Gyro (gyroADC)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], filt_color.stroke_width(2)));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                let reason = if !gyro_header_found[axis_index] {
                    "gyroADC Header Missing"
                 } else {
                    "No Valid Data Rows"
                 };
                println!("  INFO: No Gyro vs Unfiltered Gyro data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Gyro/UnfiltGyro ({})", reason))?;
            }
        }
        root_area_gyro.present()?;
        println!("  Stacked Gyro vs Unfiltered Gyro plot saved as '{}'.", output_file_gyro);
    } else {
        println!("  Skipping Stacked Gyro vs Unfiltered Gyro Plot: No data available for any axis.");
    }

    println!();
    Ok(())
}