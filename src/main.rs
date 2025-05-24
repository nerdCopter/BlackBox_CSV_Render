// src/main.rs

mod log_data;
mod constants;
mod plot_framework;  // Shared plotting logic
mod plot_functions; // Individual plot functions
mod step_response;
mod fft_utils;
mod log_parser;

use std::error::Error;
use std::env;
use std::path::Path;

use ndarray::{Array1, Array2};

use constants::*;
// Import the specific plot functions from their new locations
use plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use plot_functions::plot_setpoint_vs_gyro::plot_setpoint_vs_gyro;
use plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use plot_functions::plot_step_response::plot_step_response;
use plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;

use log_parser::parse_log_file;

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
        _gyro_unfilt_header_found,
        _debug_header_found,
    ) = parse_log_file(&input_path)?;

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Calculate Step Response Data ---
    let mut contiguous_sr_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    let first_time = all_log_data.first().and_then(|row| row.time_sec);
    let last_time = all_log_data.last().and_then(|row| row.time_sec);

    let mut required_headers_for_sr_input = true;
    for axis in 0..3 {
        if !setpoint_header_found[axis] || !gyro_header_found[axis] {
            required_headers_for_sr_input = false;
            break;
        }
    }

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

     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
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

    // --- Generate Plots ---
    plot_pidsum_error_setpoint(&all_log_data, &root_name)?;
    plot_setpoint_vs_gyro(&all_log_data, &root_name)?;
    plot_gyro_vs_unfilt(&all_log_data, &root_name)?;
    plot_step_response(&step_response_calculation_results, &root_name, sample_rate)?;
    plot_gyro_spectrums(&all_log_data, &root_name, sample_rate)?;

    println!("\nProcessing complete.");
    Ok(())
}

// src/main.rs