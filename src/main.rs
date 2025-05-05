// src/main.rs

mod log_data;
mod constants;
mod plotting_utils;
mod step_response;
mod fft_utils; // Declare the new module
mod spectrograph; // Declare the new module

use csv::ReaderBuilder;

use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use ndarray::{Array1, Array2};

use log_data::LogRowData;
use constants::{MAX_RAW_THROTTLE, THROTTLE_THRESHOLD, EXCLUDE_START_S, EXCLUDE_END_S, MOVEMENT_THRESHOLD_DEG_S, FRAME_LENGTH_S}; // Import necessary constants
// Import the specific plot functions
use plotting_utils::{
    plot_pidsum_error_setpoint,
    plot_setpoint_vs_pidsum,
    plot_setpoint_vs_gyro,
    plot_gyro_vs_unfilt,
    plot_step_response,
    plot_spectrographs, // Import the new spectrograph plot function
};
use spectrograph::SpectrographData; // Import the SpectrographData struct

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
    // Keep these for header checking logic.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[axis]" is present.
    let mut gyro_header_found = [false; 3]; // Tracks if "gyroADC[axis]" is present (filtered gyro).
    let mut gyro_unfilt_header_found = [false; 3]; // Tracks if "gyroUnfilt[axis]" is present.
    let mut debug_header_found = [false; 4]; // Tracks if "debug[idx]" is present.
    // Removed mut and initial assignment as it's only used in the is_some() check below
    let throttle_header_found: bool; // Track if throttle header is found


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
             if i == 0 || (i >= 1 && i <= 6) || (i == 7 || i == 8) { // Time, P, I, D[0], D[1] are essential
                 if !found {
                      essential_pid_headers_found = false;
                 }
             }
        }

        // Check optional 'axisD[2]' header (Index 9).
        let axis_d2_found_in_csv = header_indices[9].is_some(); // Use header_indices here
        println!("  '{}': {} (Optional, defaults to 0.0 if not found)", target_headers[9], if axis_d2_found_in_csv { "Found" } else { "Not Found" });

        // Check setpoint headers (Indices 10-12).
        for axis in 0..3 {
            setpoint_header_found[axis] = header_indices[10 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Setpoint plots and Step Response Axis {})", target_headers[10 + axis], if setpoint_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check gyro (filtered) headers (Indices 13-15).
         for axis in 0..3 {
            gyro_header_found[axis] = header_indices[13 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Step Response, Gyro plots, and PID Error Axis {})", target_headers[13 + axis], if gyro_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check gyroUnfilt headers (Indices 16-18).
        for axis in 0..3 {
            gyro_unfilt_header_found[axis] = header_indices[16 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for Gyro vs Unfilt Axis {})", target_headers[16 + axis], if gyro_unfilt_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check debug headers (Indices 19-22).
        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[19 + idx_offset].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for gyroUnfilt[0-2], used by spectrograph if available)", target_headers[19 + idx_offset], if debug_header_found[idx_offset] { "Found" } else { "Not Found" });
        }

        // Check throttle header (Index 23).
        throttle_header_found = header_indices[23].is_some(); // Use header_indices here
        println!("  '{}': {} (Optional, used for spectrograph filtering and Y-axis)", target_headers[23], if throttle_header_found { "Found" } else { "Not Found" });


        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8).filter(|&i| header_indices[i].is_none() && (i==0 || (i>=1 && i<=6) || (i==7 || i==8)) ).map(|i| format!("'{}'", target_headers[i])).collect(); // Use header_indices here
             return Err(format!("Error: Missing essential headers for PIDsum calculation: {}. Aborting.", missing_essentials.join(", ")).into());
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
                             current_row_data.d_term[axis] = Some(0.0); // Default to 0.0 if axisD[2] is missing
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
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

                    // Apply Fallback Logic for gyro_unfilt (debug[0-2] --> gyroUnfilt[0-2])
                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => match parsed_debug[axis] {
                                Some(val) => Some(val),
                                None => None, // Keep None if both missing, plotting code will handle it
                            }
                        };
                    }

                    // Parse Throttle
                    current_row_data.throttle = parse_f64_by_target_idx(23);

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // File reader is dropped here


    println!("Finished reading {} data rows.", all_log_data.len());

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        let mut prev_time: Option<f64> = None;
        for row in &all_log_data {
            if let Some(current_time) = row.time_sec {
                if let Some(pt) = prev_time {
                    let delta = current_time - pt;
                    // Only count valid time deltas
                     if delta > 1e-9 && delta < 1.0 { // Assume deltas > 1s are errors/pauses
                        total_delta += delta;
                        count += 1;
                    }
                }
                prev_time = Some(current_time);
            }
        }
        if count > 0 {
            let avg_delta = total_delta / count as f64;
            sample_rate = Some(1.0 / avg_delta);
            println!("Estimated Sample Rate: {:.2} Hz", sample_rate.unwrap());
        }
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Calculations might be affected.");
    }


    // --- Prepare Filtered Data for Calculations ---
    // Prepare step response input data filtered by time and movement threshold *once*.
    // This needs first_time and last_time which are implicitly available from the first/last row,
    // and requires setpoint and gyro being available in the log data.
    let mut step_response_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    // Prepare spectrograph input data filtered by time range *and throttle*, convert to f32 for signal.
    // Now stores (time_sec, throttle_percent, signal_f32).
    let mut spectrograph_input_data_filt: [Vec<(f64, f64, f32)>; 3] = Default::default();
    let mut spectrograph_input_data_unfilt: [Vec<(f64, f64, f32)>; 3] = Default::default();


    let first_time_val = all_log_data.first().and_then(|row| row.time_sec);
    let last_time_val = all_log_data.last().and_then(|row| row.time_sec);

    // Only attempt to filter input data if time range is available AND required headers were found
    let required_headers_for_sr_input = setpoint_header_found.iter().all(|&f| f) && gyro_header_found.iter().all(|&f| f);
    // Spectrograph needs gyro and/or gyro_unfilt, and optionally throttle for filtering
     let required_headers_for_spectrograph_input = gyro_header_found.iter().any(|&f| f) || gyro_unfilt_header_found.iter().any(|&f| f) || debug_header_found[0] || debug_header_found[1] || debug_header_found[2];


    if let (Some(first_time_val), Some(last_time_val)) = (first_time_val, last_time_val) {
         for row in &all_log_data {
             if let Some(time) = row.time_sec {
                 // Apply time range filter
                 if time >= first_time_val + EXCLUDE_START_S && time <= last_time_val - EXCLUDE_END_S {

                     // Apply throttle filter for spectrograph data if throttle header was found
                     let throttle_ok = if throttle_header_found {
                         row.throttle.map_or(false, |t| t >= THROTTLE_THRESHOLD)
                     } else {
                         true // If no throttle header, don't filter by throttle
                     };

                     if throttle_ok {
                         // Filter for Step Response (requires movement)
                         if required_headers_for_sr_input {
                             if let (Some(setpoint_roll), Some(gyro_roll),
                                     Some(setpoint_pitch), Some(gyro_pitch),
                                     Some(setpoint_yaw), Some(gyro_yaw)) = (
                                 row.setpoint[0], row.gyro[0],
                                 row.setpoint[1], row.gyro[1],
                                 row.setpoint[2], row.gyro[2],
                             ) {
                                 // Apply movement threshold
                                 if setpoint_roll.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_roll.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                     step_response_input_data[0].0.push(time);
                                     step_response_input_data[0].1.push(setpoint_roll as f32);
                                     step_response_input_data[0].2.push(gyro_roll as f32);
                                 }
                                 if setpoint_pitch.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_pitch.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                     step_response_input_data[1].0.push(time);
                                     step_response_input_data[1].1.push(setpoint_pitch as f32);
                                     step_response_input_data[1].2.push(gyro_pitch as f32);
                                 }
                                  if setpoint_yaw.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_yaw.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                     step_response_input_data[2].0.push(time);
                                     step_response_input_data[2].1.push(setpoint_yaw as f32);
                                     step_response_input_data[2].2.push(gyro_yaw as f32);
                                 }
                             }
                         }

                         // Filter for Spectrographs (only requires data existence, no movement threshold, but now throttle)
                         // Store time, throttle percentage, and signal
                         if required_headers_for_spectrograph_input {
                            let throttle_percent = row.throttle.map(|t| t / MAX_RAW_THROTTLE * 100.0).unwrap_or(0.0); // Default to 0% if throttle missing/invalid

                             for axis_index in 0..3 {
                                 if let Some(gyro_val) = row.gyro[axis_index] {
                                     spectrograph_input_data_filt[axis_index].push((time, throttle_percent, gyro_val as f32));
                                 }
                                 // Use the already applied fallback logic in LogRowData
                                 if let Some(gyro_unfilt_val) = row.gyro_unfilt[axis_index] {
                                     spectrograph_input_data_unfilt[axis_index].push((time, throttle_percent, gyro_unfilt_val as f32));
                                 }
                             }
                         }
                     } // End of throttle_ok filter
                 } // End of time range filter
             } // End of time_sec check
         } // End of row iteration

         if !required_headers_for_sr_input {
             println!("\nINFO: Skipping Step Response input data filtering: Setpoint or Gyro headers missing.");
         }
         if !required_headers_for_spectrograph_input {
             println!("\nINFO: Skipping Spectrograph input data filtering: GyroADC and GyroUnfilt/Debug headers missing.");
         }
         if throttle_header_found && spectrograph_input_data_filt.iter().all(|v| v.is_empty()) && spectrograph_input_data_unfilt.iter().all(|v| v.is_empty()) {
              println!("\nINFO: Spectrograph input data is empty after applying throttle threshold ({}). Adjust THROTTLE_THRESHOLD if needed.", THROTTLE_THRESHOLD);
         }


    } else {
         let reason = if first_time_val.is_none() || last_time_val.is_none() {
             "Time range unknown"
         } else {
              // This branch shouldn't be reached if sample_rate is None but time is Some
             "Error in time filtering logic"
         };
         println!("\nINFO: Skipping Time Range filtering for calculations: {}.", reason);
    }


    // --- Calculate Step Response Data ---
    println!("\n--- Calculating Step Response ---");
    // Store the raw QC'd stacked responses and setpoints for later averaging *within the plot function*.
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];

     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            // Check if there's *any* movement-filtered data for this axis AND required headers were found
            let required_headers_found = setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found && !step_response_input_data[axis_index].0.is_empty() {
                println!("  Attempting step response calculation for Axis {}...", axis_index);
                let time_vec = step_response_input_data[axis_index].0.clone();
                let setpoints_vec = step_response_input_data[axis_index].1.clone();
                let gyros_filtered_vec = step_response_input_data[axis_index].2.clone();

                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                 // Check if the time vector length is sufficient *before* converting to Array1
                 if time_vec.len() >= min_required_samples {
                    let time_arr = Array1::from(time_vec);
                    let setpoints_arr = Array1::from(setpoints_vec);
                    let gyros_filtered_arr = Array1::from(gyros_filtered_vec);

                    match step_response::calculate_step_response_python_style(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) {
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
                     println!("    ... Skipping Axis {}: Not enough movement data points ({}) for windowing (need at least {}).", axis_index, time_vec.len(), min_required_samples);
                }
            } else {
                 let reason = if !required_headers_found {
                     "Setpoint or Gyro headers missing"
                 } else {
                     "No movement-filtered input data available"
                 };
                 println!("  Skipping Step Response calculation for Axis {}: {}", axis_index, reason);
            }
        }
    } else {
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Calculate Spectrograph Data ---
    println!("\n--- Calculating Spectrographs ---");
    let mut spectrograph_filtered_results: [Option<SpectrographData>; 3] = [None, None, None];
    let mut spectrograph_unfiltered_results: [Option<SpectrographData>; 3] = [None, None, None];

    if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            // Check if *any* spectrograph data is available for this axis (filtered or unfiltered fallback)
            let has_filtered_data_input = !spectrograph_input_data_filt[axis_index].is_empty();
            let has_unfiltered_data_input = !spectrograph_input_data_unfilt[axis_index].is_empty();

            if !has_filtered_data_input && !has_unfiltered_data_input {
                 println!("  Skipping spectrograph calculations for Axis {}: No filtered or unfiltered data available in the time range with throttle threshold.", axis_index);
                 continue;
            }

            // Extract time, throttle percentage, and signal for this axis
            let (time_vec_filt, throttle_vec_filt, signal_vec_filt): (Vec<f64>, Vec<f64>, Vec<f32>) = spectrograph_input_data_filt[axis_index].iter().map(|(t, thr, sig)| (*t, *thr, *sig)).collect();
             let (time_vec_unfilt, throttle_vec_unfilt, signal_vec_unfilt): (Vec<f64>, Vec<f64>, Vec<f32>) = spectrograph_input_data_unfilt[axis_index].iter().map(|(t, thr, sig)| (*t, *thr, *sig)).collect();

            // Convert to Array1
            let time_arr_filt = Array1::from(time_vec_filt);
            let throttle_arr_filt = Array1::from(throttle_vec_filt);
            let signal_arr_filt = Array1::from(signal_vec_filt);

            let time_arr_unfilt = Array1::from(time_vec_unfilt);
            let throttle_arr_unfilt = Array1::from(throttle_vec_unfilt);
            let signal_arr_unfilt = Array1::from(signal_vec_unfilt);


            // Calculate filtered spectrograph
            if has_filtered_data_input {
                println!("  Attempting filtered spectrograph calculation for Axis {}...", axis_index);
                match spectrograph::calculate_spectrograph(&time_arr_filt, &throttle_arr_filt, &signal_arr_filt, sr) {
                    Ok(data) => {
                        if !data.power.is_empty() {
                             // Clone the data when assigning to the result array
                             println!("    ... Filtered spectrograph calculation successful for Axis {}. Power Shape: {:?}.", axis_index, data.power.shape());
                             spectrograph_filtered_results[axis_index] = Some(data);
                        } else {
                             println!("    ... Filtered spectrograph calculation returned no data for Axis {}.", axis_index);
                         }
                    },
                    Err(e) => eprintln!("    ... Filtered spectrograph calculation failed for Axis {}: {}", axis_index, e),
                }
            } else {
                 println!("  Skipping filtered spectrograph for Axis {}: No data available in the time range with throttle threshold.", axis_index);
            }

            // Calculate unfiltered spectrograph
            if has_unfiltered_data_input {
                 println!("  Attempting unfiltered spectrograph calculation for Axis {}...", axis_index);
                match spectrograph::calculate_spectrograph(&time_arr_unfilt, &throttle_arr_unfilt, &signal_arr_unfilt, sr) {
                    Ok(data) => {
                         if !data.power.is_empty() {
                            // Clone the data when assigning to the result array
                            println!("    ... Unfiltered spectrograph calculation successful for Axis {}. Power Shape: {:?}.", axis_index, data.power.shape());
                            spectrograph_unfiltered_results[axis_index] = Some(data);
                         } else {
                             println!("    ... Unfiltered spectrograph calculation returned no data for Axis {}.", axis_index);
                         }
                    },
                    Err(e) => eprintln!("    ... Unfiltered spectrograph calculation failed for Axis {}: {}", axis_index, e),
                }
            } else {
                println!("  Skipping unfiltered spectrograph for Axis {}: No data available in the time range with throttle threshold.", axis_index);
            }
        }
    } else {
         println!("  Skipping Spectrograph Calculations: Sample rate could not be determined.");
    }


    // --- Generate Plots ---
    // Pass the full log_data and root_name to each plotting function
    plot_pidsum_error_setpoint(&all_log_data, &root_name)?;
    plot_setpoint_vs_pidsum(&all_log_data, &root_name)?;
    plot_setpoint_vs_gyro(&all_log_data, &root_name)?;
    plot_gyro_vs_unfilt(&all_log_data, &root_name)?;
    // Pass step response results and sample rate to the step response plot function
    plot_step_response(&step_response_calculation_results, &root_name, sample_rate)?;
    // Pass spectrograph results to the spectrograph plot function
    plot_spectrographs(&spectrograph_filtered_results, &spectrograph_unfiltered_results, &root_name)?;


    println!("\nProcessing complete.");
    Ok(())
}

// src/main.rs