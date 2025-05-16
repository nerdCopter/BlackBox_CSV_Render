// src/main.rs

mod log_data;
mod constants;
mod plotting_utils;
mod step_response;
mod fft_utils;

use csv::ReaderBuilder;

use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use ndarray::{Array1, Array2};

use log_data::LogRowData;
use constants::*;
// Import the specific plot functions
use plotting_utils::{
    plot_pidsum_error_setpoint,
    plot_setpoint_vs_gyro,
    plot_gyro_vs_unfilt,
    plot_step_response,
    plot_throttle_spectrograms,
};

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
        "time (us)",                        // 0
        "axisP[0]", "axisP[1]", "axisP[2]", // 1, 2, 3
        "axisI[0]", "axisI[1]", "axisI[2]", // 4, 5, 6
        "axisD[0]", "axisD[1]", "axisD[2]", // 7, 8, 9
        "setpoint[0]", "setpoint[1]", "setpoint[2]", // 10, 11, 12
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",    // 13, 14, 15
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]", // 16, 17, 18
        "debug[0]", "debug[1]", "debug[2]", "debug[3]",    // 19, 20, 21, 22
        "setpoint[3]", // Index 23, used for throttle
    ];

    // Flags to track if specific optional or plot-dependent headers are found.
    let mut setpoint_header_found = [false; 3];
    let mut gyro_header_found = [false; 3];
    let mut gyro_unfilt_header_found = [false; 3];
    let mut debug_header_found = [false; 4];
    let setpoint3_header_found: bool; // Tracks if "setpoint[3]" (throttle) is present.


    let header_indices: Vec<Option<usize>>;

    {
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));
        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        header_indices = target_headers.iter().map(|&target_header| {
            header_record.iter().position(|h| h.trim() == target_header)
        }).collect();

        println!("Header mapping status:");
        let mut essential_pid_headers_found = true;

        for i in 0..=8 {
            let name = target_headers[i];
             let found = header_indices[i].is_some();
             println!("  '{}': {}", name, if found { "Found" } else { "Not Found" });
             if !found {
                 essential_pid_headers_found = false;
             }
        }

        let axis_d2_found_in_csv = header_indices[9].is_some();
        println!("  '{}': {} (Optional, defaults to 0.0 if not found)", target_headers[9], if axis_d2_found_in_csv { "Found" } else { "Not Found" });

        for axis in 0..3 {
            setpoint_header_found[axis] = header_indices[10 + axis].is_some();
            println!("  '{}': {} (Essential for Setpoint plots and Step Response Axis {})", target_headers[10 + axis], if setpoint_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

         for axis in 0..3 {
            gyro_header_found[axis] = header_indices[13 + axis].is_some();
            println!("  '{}': {} (Essential for Step Response, Gyro plots, and PID Error Axis {})", target_headers[13 + axis], if gyro_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        for axis in 0..3 {
            gyro_unfilt_header_found[axis] = header_indices[16 + axis].is_some();
            println!("  '{}': {} (Fallback for Gyro vs Unfilt Axis {})", target_headers[16 + axis], if gyro_unfilt_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[19 + idx_offset].is_some();
            println!("  '{}': {} (Fallback for gyroUnfilt[0-2])", target_headers[19 + idx_offset], if debug_header_found[idx_offset] { "Found" } else { "Not Found" });
        }

        setpoint3_header_found = header_indices[23].is_some();
        println!("  '{}': {} (Essential for Throttle Spectrograms)", target_headers[23], if setpoint3_header_found { "Found" } else { "Not Found" });


        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8).filter(|&i| header_indices[i].is_none()).map(|i| format!("'{}'", target_headers[i])).collect();
             return Err(format!("Error: Missing essential headers for PIDsum calculation: {}. Aborting.", missing_essentials.join(", ")).into());
        }
    }


    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading data rows...");
    {
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));

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

                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    for axis in 0..3 {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        let d_target_idx = 7 + axis;
                        if axis == 2 && header_indices[d_target_idx].is_none() {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
                    }

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

                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => match parsed_debug[axis] {
                                Some(val) => Some(val),
                                None => None,
                            }
                        };
                    }

                    current_row_data.throttle = parse_f64_by_target_idx(23);

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    }


    println!("Finished reading {} data rows.", all_log_data.len());

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    let mut sample_rate: Option<f64> = None;
    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        let mut prev_time: Option<f64> = None;
        for row in &all_log_data {
            if let Some(current_time) = row.time_sec {
                if let Some(pt) = prev_time {
                    let delta = current_time - pt;
                    if delta > 1e-9 {
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
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might be affected.");
    }

    let mut contiguous_sr_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    let first_time = all_log_data.first().and_then(|row| row.time_sec);
    let last_time = all_log_data.last().and_then(|row| row.time_sec);

    let required_headers_for_sr_input = setpoint_header_found.iter().all(|&f| f) && gyro_header_found.iter().all(|&f| f);

    if let (Some(first_time_val), Some(last_time_val), Some(_sr)) = (first_time, last_time, sample_rate) {
         if required_headers_for_sr_input {
             for row in &all_log_data {
                 if let (Some(time), Some(setpoint_roll), Some(gyro_roll),
                         Some(setpoint_pitch), Some(gyro_pitch),
                         Some(setpoint_yaw), Some(gyro_yaw)) = (
                     row.time_sec, row.setpoint[0], row.gyro[0],
                     row.setpoint[1], row.gyro[1],
                     row.setpoint[2], row.gyro[2],
                 ) {
                     if time >= first_time_val + EXCLUDE_START_S && time <= last_time_val - EXCLUDE_END_S {
                         contiguous_sr_input_data[0].0.push(time); contiguous_sr_input_data[0].1.push(setpoint_roll as f32); contiguous_sr_input_data[0].2.push(gyro_roll as f32);
                         contiguous_sr_input_data[1].0.push(time); contiguous_sr_input_data[1].1.push(setpoint_pitch as f32); contiguous_sr_input_data[1].2.push(gyro_pitch as f32);
                         contiguous_sr_input_data[2].0.push(time); contiguous_sr_input_data[2].1.push(setpoint_yaw as f32); contiguous_sr_input_data[2].2.push(gyro_yaw as f32);
                     }
                 }
             }
         } else {
              println!("\nINFO: Skipping Step Response data collection: Setpoint or Gyro headers missing.");
         }
    } else {
         let reason = if first_time.is_none() || last_time.is_none() {
             "Time range unknown"
         } else {
             "Sample Rate unknown"
         };
         println!("\nINFO: Skipping Step Response input data filtering: {}.", reason);
    }


    println!("\n--- Calculating Step Response ---");
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];

     if let Some(sr_val) = sample_rate { // Renamed sr to sr_val to avoid conflict if _sr above is uncommented
        for axis_index in 0..3 {
            let required_headers_found_sr = setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found_sr && !contiguous_sr_input_data[axis_index].0.is_empty() {
                println!("  Attempting step response calculation for Axis {}...", axis_index);
                let time_arr = Array1::from(contiguous_sr_input_data[axis_index].0.clone());
                let setpoints_arr = Array1::from(contiguous_sr_input_data[axis_index].1.clone());
                let gyros_filtered_arr = Array1::from(contiguous_sr_input_data[axis_index].2.clone());

                let min_required_samples = (FRAME_LENGTH_S * sr_val).ceil() as usize;
                if time_arr.len() >= min_required_samples {
                    match step_response::calculate_step_response(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr_val) {
                        Ok(result) => {
                             let num_qc_windows = result.1.shape()[0];
                             if num_qc_windows > 0 {
                                 step_response_calculation_results[axis_index] = Some(result);
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
                 let reason = if !required_headers_found_sr {
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


    plot_pidsum_error_setpoint(&all_log_data, &root_name)?;
    plot_setpoint_vs_gyro(&all_log_data, &root_name)?;
    plot_gyro_vs_unfilt(&all_log_data, &root_name)?;
    plot_step_response(&step_response_calculation_results, &root_name, sample_rate)?;
    plot_throttle_spectrograms(&all_log_data, &root_name, sample_rate)?;

    println!("\nProcessing complete.");
    Ok(())
}

// src/main.rs
