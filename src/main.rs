// src/main.rs

mod constants;
mod data_analysis;
mod data_input;
mod plot_framework;
mod plot_functions;

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf}; // Added PathBuf // Added HashSet

use ndarray::{Array1, Array2};

use crate::constants::{
    DEFAULT_SETPOINT_THRESHOLD, EXCLUDE_END_S, EXCLUDE_START_S, FRAME_LENGTH_S,
};

// Specific plot function imports
use crate::plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;
use crate::plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use crate::plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use crate::plot_functions::plot_psd::plot_psd;
use crate::plot_functions::plot_psd_db_heatmap::plot_psd_db_heatmap;
use crate::plot_functions::plot_setpoint_vs_gyro::plot_setpoint_vs_gyro;
use crate::plot_functions::plot_step_response::plot_step_response;
use crate::plot_functions::plot_throttle_freq_heatmap::plot_throttle_freq_heatmap;

// Data input import
use crate::data_input::log_parser::parse_log_file;

// Data analysis imports
use crate::data_analysis::calc_step_response;

fn print_usage_and_exit(program_name: &str) {
    eprintln!("
Usage: {} <input_file1.csv> [<input_file2.csv> ...] [--dps <value>] [--output-dir <directory>] [--debug]", program_name);
    eprintln!("  <input_fileX.csv>: Path to one or more input CSV log files (required).");
    eprintln!("  --dps <value>: Optional. Enables detailed step response plots with the specified");
    eprintln!("                 deg/s threshold value. Must be a positive number.");
    eprintln!("                 If --dps is omitted, a general step-response is shown.");
    eprintln!(
        "  --output-dir <directory>: Optional. Specifies the output directory for generated plots."
    );
    eprintln!("                         If omitted, plots are saved in the source folder (input file's directory).");
    eprintln!("  --debug: Optional. Shows detailed metadata information during processing.");
    eprintln!("  --help: Show this help message and exit.");
    eprintln!("  --version: Show version information and exit.");
    eprintln!(
        "
Arguments can be in any order. Wildcards (e.g., *.csv) are supported by the shell."
    );
    std::process::exit(1);
}

fn print_version_and_exit() {
    println!(
        "{} version {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    std::process::exit(0);
}

fn process_file(
    input_file_str: &str,
    setpoint_threshold: f64,
    show_legend: bool,
    use_dir_prefix: bool,
    output_dir: Option<&str>,
    debug_mode: bool,
) -> Result<(), Box<dyn Error>> {
    // --- Setup paths and names ---
    let input_path = Path::new(input_file_str);
    if !input_path.exists() {
        eprintln!("Error: Input file not found: {}", input_file_str);
        return Ok(()); // Continue to next file if this one is not found
    }
    println!("\n--- Processing file: {} ---", input_file_str);

    let file_stem_cow = input_path
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new("unknown_filestem"))
        .to_string_lossy();
    let root_name_string: String;

    if use_dir_prefix {
        let mut dir_prefix_to_add = String::new();
        if let Some(parent_dir) = input_path.parent() {
            if let Some(dir_os_str) = parent_dir.file_name() {
                let dir_name_part = dir_os_str.to_string_lossy();
                // Add prefix only if parent dir name is meaningful (not empty, not current dir indicator like ".")
                if !dir_name_part.is_empty() && dir_name_part != "." {
                    let sanitized_dir = dir_name_part
                        .chars()
                        .map(|c| {
                            if c.is_alphanumeric() || c == '-' || c == '_' {
                                c
                            } else {
                                '_'
                            }
                        })
                        .collect::<String>();
                    dir_prefix_to_add = format!("{}_", sanitized_dir);
                }
            }
        }
        root_name_string = format!("{}{}", dir_prefix_to_add, file_stem_cow);
    } else {
        root_name_string = file_stem_cow.into_owned();
    }

    // --- Data Reading and Header Status ---
    let (
        all_log_data,
        sample_rate,
        f_term_header_found,
        setpoint_header_found,
        gyro_header_found,
        _gyro_unfilt_header_found,
        _debug_header_found,
        _header_metadata,
    ) = match parse_log_file(&input_path, debug_mode) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error parsing log file {}: {}", input_file_str, e);
            return Ok(()); // Continue to next file
        }
    };

    if all_log_data.is_empty() {
        println!(
            "No valid data rows read from {}, cannot generate plots.",
            input_file_str
        );
        return Ok(());
    }

    let mut has_nonzero_f_term_data = [false; 3];
    for axis in 0..3 {
        if f_term_header_found[axis] {
            if all_log_data
                .iter()
                .any(|row| row.f_term[axis].map_or(false, |val| val.abs() > 1e-9))
            {
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

    if let (Some(first_time_val), Some(last_time_val), Some(_sr_val_check)) =
        (first_time, last_time, sample_rate)
    {
        // Renamed _sr to _sr_val_check to avoid conflict
        if required_headers_for_sr_input {
            for row in &all_log_data {
                if let (
                    Some(time),
                    Some(setpoint_roll),
                    Some(gyro_roll),
                    Some(setpoint_pitch),
                    Some(gyro_pitch),
                    Some(setpoint_yaw),
                    Some(gyro_yaw),
                ) = (
                    row.time_sec,
                    row.setpoint[0],
                    row.gyro[0],
                    row.setpoint[1],
                    row.gyro[1],
                    row.setpoint[2],
                    row.gyro[2],
                ) {
                    if time >= first_time_val + EXCLUDE_START_S
                        && time <= last_time_val - EXCLUDE_END_S
                    {
                        contiguous_sr_input_data[0].0.push(time);
                        contiguous_sr_input_data[0].1.push(setpoint_roll as f32);
                        contiguous_sr_input_data[0].2.push(gyro_roll as f32);
                        contiguous_sr_input_data[1].0.push(time);
                        contiguous_sr_input_data[1].1.push(setpoint_pitch as f32);
                        contiguous_sr_input_data[1].2.push(gyro_pitch as f32);
                        contiguous_sr_input_data[2].0.push(time);
                        contiguous_sr_input_data[2].1.push(setpoint_yaw as f32);
                        contiguous_sr_input_data[2].2.push(gyro_yaw as f32);
                    }
                }
            }
        } else {
            println!(
                "
INFO ({}): Skipping Step Response data collection: Setpoint or Gyro headers missing.",
                input_file_str
            );
        }
    } else {
        let reason = if first_time.is_none() || last_time.is_none() {
            "Time range unknown"
        } else {
            "Sample Rate unknown"
        };
        println!(
            "
INFO ({}): Skipping Step Response input data filtering: {}.",
            input_file_str, reason
        );
    }

    println!(
        "
--- Calculating Step Response for {} ---",
        input_file_str
    );
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>;
        3] = [None, None, None];

    if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            let required_headers_found =
                setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found && !contiguous_sr_input_data[axis_index].0.is_empty() {
                println!(
                    "  Attempting step response calculation for Axis {}...",
                    axis_index
                );
                let time_arr = Array1::from(contiguous_sr_input_data[axis_index].0.clone());
                let setpoints_arr = Array1::from(contiguous_sr_input_data[axis_index].1.clone());
                let gyros_filtered_arr =
                    Array1::from(contiguous_sr_input_data[axis_index].2.clone());

                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                if time_arr.len() >= min_required_samples {
                    match calc_step_response::calculate_step_response(
                        &time_arr,
                        &setpoints_arr,
                        &gyros_filtered_arr,
                        sr,
                    ) {
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
                println!(
                    "  Skipping Step Response calculation for Axis {}: {}",
                    axis_index, reason
                );
            }
        }
    } else {
        println!(
            "  Skipping Step Response Calculation for {}: Sample rate could not be determined.",
            input_file_str
        );
    }

    // --- Generate Plots ---
    println!(
        "Generating plots for {} (root name: {})...",
        input_file_str, root_name_string
    );

    // Set the current working directory to the output directory if specified
    let original_dir = std::env::current_dir()?;
    if let Some(output_dir) = output_dir {
        // Ensure output directory exists
        std::fs::create_dir_all(output_dir)?;
        // Change to output directory for plot generation
        std::env::set_current_dir(output_dir)?;
    }

    // Use only the root filename (without path) for PNG output
    plot_pidsum_error_setpoint(&all_log_data, &root_name_string)?;
    plot_setpoint_vs_gyro(&all_log_data, &root_name_string, sample_rate)?;
    plot_gyro_vs_unfilt(&all_log_data, &root_name_string, sample_rate)?;
    plot_step_response(
        &step_response_calculation_results,
        &root_name_string,
        sample_rate,
        &has_nonzero_f_term_data,
        setpoint_threshold,
        show_legend,
    )?;
    plot_gyro_spectrums(&all_log_data, &root_name_string, sample_rate)?;
    plot_psd(&all_log_data, &root_name_string, sample_rate)?;
    plot_psd_db_heatmap(&all_log_data, &root_name_string, sample_rate)?;
    plot_throttle_freq_heatmap(&all_log_data, &root_name_string, sample_rate)?;

    // Restore original working directory
    std::env::set_current_dir(&original_dir)?;

    println!("--- Finished processing file: {} ---", input_file_str);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 2 {
        print_usage_and_exit(program_name);
    }

    let mut input_files: Vec<String> = Vec::new();
    let mut setpoint_threshold_override: Option<f64> = None;
    let mut dps_flag_present = false;
    let mut output_dir: Option<String> = None; // None = not specified (use source folder), Some(dir) = --output-dir with value
    let mut debug_mode = false;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--help" {
            print_usage_and_exit(program_name);
        } else if arg == "--version" {
            print_version_and_exit();
        } else if arg == "--dps" {
            if dps_flag_present {
                eprintln!("Error: --dps argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            if i + 1 >= args.len() || args[i + 1].starts_with("--") {
                eprintln!("Error: --dps requires a numeric value (deg/s threshold).");
                print_usage_and_exit(program_name);
            }
            dps_flag_present = true;
            match args[i + 1].parse::<f64>() {
                Ok(val) => {
                    if val < 0.0 {
                        eprintln!("Error: --dps threshold cannot be negative: {}", val);
                        print_usage_and_exit(program_name);
                    }
                    setpoint_threshold_override = Some(val);
                    i += 1;
                }
                Err(_) => {
                    eprintln!("Error: Invalid numeric value for --dps: {}", args[i + 1]);
                    print_usage_and_exit(program_name);
                }
            }
        } else if arg == "--output-dir" {
            if output_dir.is_some() {
                eprintln!("Error: --output-dir argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            if i + 1 >= args.len() || args[i + 1].starts_with("--") {
                eprintln!("Error: --output-dir requires a directory path.");
                print_usage_and_exit(program_name);
            } else {
                // --output-dir with directory value
                output_dir = Some(args[i + 1].clone());
                i += 1;
            }
        } else if arg == "--debug" {
            debug_mode = true;
        } else if arg.starts_with("--") {
            eprintln!("Error: Unknown option '{}'", arg);
            print_usage_and_exit(program_name);
        } else {
            input_files.push(arg.clone()); // THIS IS THE CORRECT LOGIC FOR VEC
        }
        i += 1;
    }

    if input_files.is_empty() {
        eprintln!("Error: At least one input file is required.");
        print_usage_and_exit(program_name);
    }

    let setpoint_threshold: f64;
    let show_legend: bool;

    if dps_flag_present {
        setpoint_threshold = setpoint_threshold_override.unwrap_or(DEFAULT_SETPOINT_THRESHOLD);
        show_legend = true;
    } else {
        setpoint_threshold = DEFAULT_SETPOINT_THRESHOLD;
        show_legend = false;
    }

    let mut use_dir_prefix_for_root_name = false;
    if input_files.len() > 1 {
        let parent_dirs_set: HashSet<PathBuf> = input_files
            .iter()
            .filter_map(|f_str| Path::new(f_str).parent().map(|p| p.to_path_buf()))
            .collect();
        if parent_dirs_set.len() > 1 {
            use_dir_prefix_for_root_name = true;
        }
    }

    let mut overall_success = true;
    for input_file_str in &input_files {
        // Determine the actual output directory for this file
        let actual_output_dir = match &output_dir {
            None => {
                // No --output-dir specified, use input file's directory (source folder)
                Path::new(input_file_str).parent().and_then(|p| p.to_str())
            }
            Some(dir) => Some(dir.as_str()), // --output-dir with specific directory
        };

        if let Err(e) = process_file(
            input_file_str,
            setpoint_threshold,
            show_legend,
            use_dir_prefix_for_root_name,
            actual_output_dir,
            debug_mode,
        ) {
            eprintln!(
                "An error occurred while processing {}: {}",
                input_file_str, e
            );
            overall_success = false;
        }
    }

    if overall_success {
        println!(
            "
All files processed successfully."
        );
        Ok(())
    } else {
        eprintln!(
            "
Some files could not be processed successfully."
        );
        // Still return Ok(()) here as we've handled errors per file.
        // Or, could return a generic error if preferred for the whole batch.
        // For now, exiting with 0 if any file succeeded, or if all failed but were handled.
        // To signal overall failure to scripts, one might `std::process::exit(1)` here.
        Ok(())
    }
}

// src/main.rs
