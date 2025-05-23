mod log_data;
mod constants;
mod plotting_utils;
mod step_response;
mod fft_utils;
mod log_parser; // New import

use std::error::Error;
use std::env;
use std::path::Path;

use ndarray::{Array1, Array2};

use constants::*;
// Import the specific plot functions
use plotting_utils::{
    plot_pidsum_error_setpoint,
    plot_setpoint_vs_gyro,
    plot_gyro_vs_unfilt,
    plot_step_response,
};
use log_parser::parse_log_file; // Import the new parsing function

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

    // --- Data Reading and Header Status ---
    let (
        all_log_data,
        sample_rate,
        setpoint_header_found,
        gyro_header_found,
        _gyro_unfilt_header_found, // Not directly used here, but returned by parser
        _debug_header_found,       // Not directly used here, but returned by parser
    ) = parse_log_file(&input_path)?;

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Calculate Step Response Data ---
    // Prepare step response input data filtered by time and movement threshold *once*.
    // This filtering logic is now moved inside calculate_step_response.
    // This needs first_time and last_time which are implicitly available from the first/last row,
    // and requires setpoint and gyro being available in the log data.
    let mut contiguous_sr_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    let first_time = all_log_data.first().and_then(|row| row.time_sec);
    let last_time = all_log_data.last().and_then(|row| row.time_sec);

    // Only attempt to filter input data if time range and sample rate are available AND required headers were found
    let mut required_headers_for_sr_input = true;
    for axis in 0..3 { // Only check setpoint[0-2] for step response input
        if !setpoint_header_found[axis] || !gyro_header_found[axis] {
            required_headers_for_sr_input = false;
            break;
        }
    }

    if let (Some(first_time_val), Some(last_time_val), Some(_sr)) = (first_time, last_time, sample_rate) { // _sr is unused here
         if required_headers_for_sr_input {
              // Collect all data within the time exclusion range
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
    // Store the raw QC'd stacked responses and setpoints for later averaging *within the plot function*.
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];

     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            // Check if there's *any* movement-filtered data for this axis AND required headers were found
            // The movement filtering is now inside the calculation function
            let required_headers_found = setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found && !contiguous_sr_input_data[axis_index].0.is_empty() {
                println!("  Attempting step response calculation for Axis {}...", axis_index);
                let time_arr = Array1::from(contiguous_sr_input_data[axis_index].0.clone());
                let setpoints_arr = Array1::from(contiguous_sr_input_data[axis_index].1.clone());
                let gyros_filtered_arr = Array1::from(contiguous_sr_input_data[axis_index].2.clone());

                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                if time_arr.len() >= min_required_samples {
                    match step_response::calculate_step_response(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) {
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
                 let reason = if !required_headers_found {
                     "Setpoint or Gyro headers missing"
                 } else { // This case should be caught by the time range check above, but kept for clarity
                     "No movement-filtered input data available"
                 };
                 println!("  Skipping Step Response calculation for Axis {}: {}", axis_index, reason);
            }
        }
    } else {
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Generate Plots ---
    // Pass the full log_data and root_name to each plotting function
    plot_pidsum_error_setpoint(&all_log_data, &root_name)?;
    plot_setpoint_vs_gyro(&all_log_data, &root_name)?;
    plot_gyro_vs_unfilt(&all_log_data, &root_name)?;
    // Pass step response results and sample rate to the step response plot function
    plot_step_response(&step_response_calculation_results, &root_name, sample_rate)?;


    println!("\nProcessing complete.");
    Ok(())
}

// src/main.rs
