// src/main.rs

mod log_data;
mod constants;
mod plotting_utils;
mod step_response;
mod fft_utils;
mod log_parser; // Assuming you have this from previous instructions

use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::{File, OpenOptions}; // Added for diagnostic file
use std::io::Write; // Added for diagnostic file

use ndarray::{Array1, Array2};

use constants::*;
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
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy().to_string();

    // --- Setup Diagnostic File ---
    let diag_filename = format!("{}_diag.txt", root_name);
    let mut diag_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&diag_filename)?;
    // --- End Setup Diagnostic File ---

    writeln!(diag_file, "Starting processing for: {}", input_file)?;
    println!("Reading {}...", input_file);

    // --- Data Reading and Header Processing ---
    let (all_log_data, sample_rate, headers) =
        match log_parser::parse_csv(input_file, Some(&mut diag_file)) {
            Ok((data, sr_opt, hdr)) => (data, sr_opt, hdr),
            Err(e) => {
                eprintln!("Fatal error during CSV parsing: {}", e);
                writeln!(diag_file, "Fatal error during CSV parsing: {}", e)?;
                return Err(e);
            }
        };

    // Derive header-presence flags
    let setpoint_header_found = [
        headers.contains(&"setpoint[0]".to_string()),
        headers.contains(&"setpoint[1]".to_string()),
        headers.contains(&"setpoint[2]".to_string()),
    ];
    let gyro_header_found = [
        headers.contains(&"gyroADC[0]".to_string()),
        headers.contains(&"gyroADC[1]".to_string()),
        headers.contains(&"gyroADC[2]".to_string()),
    ];
    let throttle_header_found = headers.contains(&"throttle".to_string());

    println!("Finished reading {} data rows.", all_log_data.len());
    if let Some(sr_val) = sample_rate {
        println!("Estimated Sample Rate: {:.2} Hz", sr_val);
        // Diag file writing for sample rate is now handled in parse_csv
    } else {
        println!("Warning: Sample rate could not be estimated.");
    }

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        writeln!(diag_file, "No valid data rows read, exiting.")?;
        return Ok(());
    }

    // --- Calculate Step Response Data (Restoring logic similar to your copy 20) ---
    let mut contiguous_sr_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = Default::default();
    let first_time = all_log_data.first().and_then(|row| row.time_sec);
    let last_time = all_log_data.last().and_then(|row| row.time_sec);
    let required_headers_for_sr_input = setpoint_header_found.iter().all(|&f| f) && gyro_header_found.iter().all(|&f| f);

    if let (Some(first_time_val), Some(last_time_val), Some(_sr_val)) = (first_time, last_time, sample_rate) {
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
              let msg = "\nINFO: Skipping Step Response data collection: Setpoint or Gyro headers missing.";
              println!("{}", msg);
              writeln!(diag_file, "{}", msg)?;
         }
    } else {
         let reason = if first_time.is_none() || last_time.is_none() { "Time range unknown" } else { "Sample Rate unknown" };
         let msg = format!("\nINFO: Skipping Step Response input data filtering: {}.", reason);
         println!("{}", msg);
         writeln!(diag_file, "{}", msg)?;
    }

    println!("\n--- Calculating Step Response ---");
    writeln!(diag_file, "\n--- Calculating Step Response ---")?;
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];

     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            let required_headers_found = setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found && !contiguous_sr_input_data[axis_index].0.is_empty() {
                println!("  Attempting step response calculation for Axis {}...", axis_index);
                // **NOTE**: Your `step_response::calculate_step_response` might need modification 
                // to accept `Option<&mut File>` if you want its internal details in the diag file.
                // The current signature in `step_response (copy 22).rs` does not take `diag_file`.
                // If it does extensive processing/QC before returning, you might want to add it there.
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
                                 let msg = format!("    ... Calculation successful for Axis {}. {} windows passed QC.", axis_index, num_qc_windows);
                                 println!("{}", msg);
                                 writeln!(diag_file, "{}", msg)?;
                             } else {
                                let msg = format!("    ... Calculation returned no valid windows for Axis {}. Skipping.", axis_index);
                                println!("{}", msg);
                                writeln!(diag_file, "{}", msg)?;
                             }
                        }
                        Err(e) => {
                            let msg = format!("    ... Calculation failed for Axis {}: {}", axis_index, e);
                            eprintln!("{}", msg);
                            writeln!(diag_file, "{}", msg)?;
                        }
                    }
                } else {
                     let msg = format!("    ... Skipping Axis {}: Not enough movement data points ({}) for windowing (need at least {}).", axis_index, time_arr.len(), min_required_samples);
                     println!("{}", msg);
                     writeln!(diag_file, "{}", msg)?;
                }
            } else {
                 let reason = if !required_headers_found {
                     "Setpoint or Gyro headers missing"
                 } else { 
                     "No movement-filtered input data available"
                 };
                 let msg = format!("  Skipping Step Response calculation for Axis {}: {}", axis_index, reason);
                 println!("{}", msg);
                 writeln!(diag_file, "{}", msg)?;
            }
        }
    } else {
         let msg = "  Skipping Step Response Calculation: Sample rate could not be determined.";
         println!("{}", msg);
         writeln!(diag_file, "{}", msg)?;
    }


    // --- Generate Plots ---
    if let Err(e) = plot_pidsum_error_setpoint(&all_log_data, &root_name) {
        eprintln!("Error plotting PIDsum/Error: {}", e);
    }
    if let Err(e) = plot_setpoint_vs_gyro(&all_log_data, &root_name) {
        eprintln!("Error plotting Setpoint/Gyro: {}", e);
    }
    if let Err(e) = plot_gyro_vs_unfilt(&all_log_data, &root_name) {
        eprintln!("Error plotting Gyro/Unfilt: {}", e);
    }
    if let Err(e) = plot_step_response(&step_response_calculation_results, &root_name, sample_rate) {
        eprintln!("Error plotting step response: {}", e);
    }
    if throttle_header_found {
        if let Err(e) = plot_throttle_spectrograms(&all_log_data, &root_name, sample_rate, Some(&mut diag_file)) {
            eprintln!("Error plotting throttle spectrograms: {}", e);
        }
    } else {
        let msg = "\nSkipping Throttle Spectrograms: 'throttle' header not found in CSV.";
        println!("{}", msg);
        writeln!(diag_file, "{}", msg)?;
    }

    println!("\nProcessing complete. Diagnostic log saved to {}", diag_filename);
    Ok(())
}

// src/main.rs