// src/main.rs

mod constants;
mod data_analysis;
mod data_input;
mod plot_framework;
mod plot_functions;

use std::error::Error;
use std::env;
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::constants::{DEFAULT_SETPOINT_THRESHOLD, EXCLUDE_START_S, EXCLUDE_END_S, FRAME_LENGTH_S};

// Specific plot function imports
use crate::plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use crate::plot_functions::plot_setpoint_vs_gyro::plot_setpoint_vs_gyro;
use crate::plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use crate::plot_functions::plot_step_response::plot_step_response;
use crate::plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;
use crate::plot_functions::plot_psd::plot_psd;
use crate::plot_functions::plot_psd_db_heatmap::plot_psd_db_heatmap;
use crate::plot_functions::plot_throttle_freq_heatmap::plot_throttle_freq_heatmap;

// Data input import
use crate::data_input::log_parser::parse_log_file;

// Data analysis imports
use crate::data_analysis::calc_step_response;

fn print_usage_and_exit(program_name: &str) {
    eprintln!("\\nUsage: {} <input_file.csv> [--dps [<value>]]", program_name);
    eprintln!("  <input_file.csv>: Path to the input CSV log file (required).");
    eprintln!("  --dps [<value>]: Optional. Enables detailed step response plots and legend.");
    eprintln!("                   If <value> (deg/s threshold) is provided, it's used.");
    eprintln!("                   If <value> is omitted, defaults to {}.", DEFAULT_SETPOINT_THRESHOLD);
    eprintln!("                   If --dps is omitted, a simplified combined plot is shown with a basic legend entry.");
    eprintln!("\\nArguments can be in any order.");
    std::process::exit(1);
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 2 { // Basic check for at least one argument (input file)
        print_usage_and_exit(program_name);
    }

    let mut input_file_arg: Option<String> = None;
    let mut setpoint_threshold_override: Option<f64> = None;
    let mut dps_flag_present = false;

    let mut i = 1; // Start iterating from the first argument
    while i < args.len() {
        let arg = &args[i];
        if arg == "--dps" {
            if dps_flag_present {
                eprintln!("Error: --dps argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            dps_flag_present = true;
            // Check if next argument is a value for --dps
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                match args[i + 1].parse::<f64>() {
                    Ok(val) => {
                        if val < 0.0 {
                            eprintln!("Error: --dps threshold cannot be negative: {}", val);
                            print_usage_and_exit(program_name);
                        }
                        setpoint_threshold_override = Some(val);
                        i += 1; // Consume the value argument
                    }
                    Err(_) => {
                        eprintln!("Error: Invalid numeric value for --dps: {}", args[i + 1]);
                        print_usage_and_exit(program_name);
                    }
                }
            }
        } else if arg.starts_with("--") {
            eprintln!("Error: Unknown option '{}'", arg);
            print_usage_and_exit(program_name);
        } else {
            // Argument is not an option, assume it's the input file
            if input_file_arg.is_some() {
                eprintln!("Error: Multiple input files specified ('{}' and '{}').", input_file_arg.as_ref().unwrap(), arg);
                print_usage_and_exit(program_name);
            }
            input_file_arg = Some(arg.clone());
        }
        i += 1;
    }

    // Validate input file
    if input_file_arg.is_none() {
        eprintln!("Error: Input file is required.");
        print_usage_and_exit(program_name);
    }
    let input_file_str = input_file_arg.unwrap(); // Safe due to the check above

    // Determine setpoint_threshold and show_legend based on parsed args
    let setpoint_threshold: f64;
    let show_legend: bool;

    if dps_flag_present {
        setpoint_threshold = setpoint_threshold_override.unwrap_or(DEFAULT_SETPOINT_THRESHOLD);
        show_legend = true;
    } else {
        setpoint_threshold = DEFAULT_SETPOINT_THRESHOLD;
        show_legend = false;
    }

    // --- Setup paths and names ---
    let input_path = Path::new(&input_file_str);
    if !input_path.exists() { // Check if file exists after confirming one was provided
        eprintln!("Error: Input file not found: {}", input_file_str);
        std::process::exit(1); // Exit directly as this is a fatal error post-parsing
    }
    println!("Reading {}", input_file_str);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();


    // --- Data Reading and Header Status ---
    let (
        all_log_data,
        sample_rate,
        f_term_header_found,
        setpoint_header_found,
        gyro_header_found,
        _gyro_unfilt_header_found,
        _debug_header_found,
    ) = parse_log_file(&input_path)?;

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    let mut has_nonzero_f_term_data = [false; 3];
    for axis in 0..3 {
        if f_term_header_found[axis] {
            if all_log_data.iter().any(|row| row.f_term[axis].map_or(false, |val| val.abs() > 1e-9)) {
                has_nonzero_f_term_data[axis] = true;
            }
        }
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
                    match calc_step_response::calculate_step_response(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) {
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
    plot_step_response(&step_response_calculation_results, &root_name, sample_rate, &has_nonzero_f_term_data, setpoint_threshold, show_legend)?;
    plot_gyro_spectrums(&all_log_data, &root_name, sample_rate)?;
    plot_psd(&all_log_data, &root_name, sample_rate)?;
    plot_psd_db_heatmap(&all_log_data, &root_name, sample_rate)?;
    plot_throttle_freq_heatmap(&all_log_data, &root_name, sample_rate)?;

    println!("\nProcessing complete.");
    Ok(())
}

// src/main.rs