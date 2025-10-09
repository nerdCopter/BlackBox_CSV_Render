// src/main.rs

mod axis_names;
mod constants;
mod data_analysis;
mod data_input;
mod pid_context;
mod plot_framework;
mod plot_functions;
mod types;

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array1;

use crate::types::StepResponseResults;

use crate::constants::{
    DEFAULT_SETPOINT_THRESHOLD, EXCLUDE_END_S, EXCLUDE_START_S, FRAME_LENGTH_S,
};

// Specific plot function imports
use crate::plot_functions::plot_d_term_heatmap::plot_d_term_heatmap;
use crate::plot_functions::plot_d_term_psd::plot_d_term_psd;
use crate::plot_functions::plot_d_term_spectrums::plot_d_term_spectrums;
use crate::plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;
use crate::plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use crate::plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use crate::plot_functions::plot_psd::plot_psd;
use crate::plot_functions::plot_psd_db_heatmap::plot_psd_db_heatmap;
use crate::plot_functions::plot_setpoint_vs_gyro::plot_setpoint_vs_gyro;
use crate::plot_functions::plot_step_response::plot_step_response;
use crate::plot_functions::plot_throttle_freq_heatmap::plot_throttle_freq_heatmap;

/// RAII guard to ensure current working directory is restored
struct CwdGuard {
    original_dir: PathBuf,
}

impl CwdGuard {
    /// Create a new CWD guard, saving the current directory
    fn new() -> Result<Self, std::io::Error> {
        let original_dir = env::current_dir()?;
        Ok(CwdGuard { original_dir })
    }
}

impl Drop for CwdGuard {
    /// Automatically restore the original directory when the guard goes out of scope
    fn drop(&mut self) {
        if let Err(e) = env::set_current_dir(&self.original_dir) {
            eprintln!(
                "Warning: Failed to restore original directory to {}: {}",
                self.original_dir.display(),
                e
            );
        }
    }
}

// Data input import
use crate::data_input::log_parser::parse_log_file;
use crate::data_input::pid_metadata::parse_pid_metadata;

// PID context import
use crate::pid_context::PidContext;

// Data analysis imports
use crate::data_analysis::calc_step_response;

/// Expand input paths to a list of CSV files.
/// If a path is a file, validate CSV extension before adding.
/// If a path is a directory, recursively find all CSV files within it.
fn expand_input_paths(input_paths: &[String]) -> Result<Vec<String>, Box<dyn Error>> {
    let mut csv_files = Vec::new();

    for input_path_str in input_paths {
        let input_path = Path::new(input_path_str);

        if input_path.is_file() {
            // It's a file, validate CSV extension before adding
            if let Some(extension) = input_path.extension() {
                if extension.to_string_lossy().eq_ignore_ascii_case("csv") {
                    csv_files.push(input_path_str.clone());
                } else {
                    eprintln!("Warning: Skipping non-CSV file: {}", input_path_str);
                }
            } else {
                eprintln!(
                    "Warning: Skipping file without extension: {}",
                    input_path_str
                );
            }
        } else if input_path.is_dir() {
            // It's a directory, find all CSV files recursively
            let mut dir_csv_files = find_csv_files_in_dir(input_path)?;
            csv_files.append(&mut dir_csv_files);
        } else {
            // Path doesn't exist or isn't accessible
            eprintln!(
                "Warning: Path not found or not accessible: {}",
                input_path_str
            );
        }
    }

    Ok(csv_files)
}

/// Recursively find all CSV files in a directory
fn find_csv_files_in_dir(dir_path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut visited = HashSet::new();
    find_csv_files_in_dir_impl(dir_path, &mut visited)
}

/// Internal implementation with symlink loop protection
fn find_csv_files_in_dir_impl(
    dir_path: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut csv_files = Vec::new();

    if !dir_path.is_dir() {
        return Ok(csv_files);
    }

    // Canonicalize path to detect symlink loops
    let canonical_path = match dir_path.canonicalize() {
        Ok(path) => path,
        Err(_) => {
            eprintln!(
                "Warning: Cannot canonicalize directory path: {}",
                dir_path.display()
            );
            return Ok(csv_files);
        }
    };

    // Check if we've already visited this directory (symlink loop detection)
    if visited.contains(&canonical_path) {
        eprintln!(
            "Warning: Skipping directory due to symlink loop: {}",
            dir_path.display()
        );
        return Ok(csv_files);
    }
    visited.insert(canonical_path);

    let entries = match fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!(
                "Warning: Cannot read directory '{}': {}",
                dir_path.display(),
                err
            );
            return Ok(csv_files);
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!(
                    "Warning: Error reading directory entry in '{}': {}",
                    dir_path.display(),
                    err
                );
                continue;
            }
        };
        let path = entry.path();

        if path.is_dir() {
            // Recursively search subdirectories
            match find_csv_files_in_dir_impl(&path, visited) {
                Ok(mut sub_csv_files) => csv_files.append(&mut sub_csv_files),
                Err(err) => eprintln!(
                    "Warning: Error processing subdirectory '{}': {}",
                    path.display(),
                    err
                ),
            }
        } else if path.is_file() {
            // Check if it's a CSV file
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().eq_ignore_ascii_case("csv") {
                    match path.to_str() {
                        Some(path_str) => csv_files.push(path_str.to_string()),
                        None => eprintln!(
                            "Warning: Skipping file with non-UTF-8 path: {}",
                            path.display()
                        ),
                    }
                }
            }
        }
    }

    // Sort the files for consistent ordering
    csv_files.sort();
    Ok(csv_files)
}

fn print_usage_and_exit(program_name: &str) {
    eprintln!("
Usage: {program_name} <input1> [<input2> ...] [--dps <value>] [--output-dir <directory>] [--butterworth] [--debug]");
    eprintln!("  <inputX>: Path to one or more input CSV log files or directories containing CSV files (required).");
    eprintln!("            If a directory is specified, all CSV files within it (including subdirectories) will be processed.");
    eprintln!("  --dps <value>: Optional. Enables detailed step response plots with the specified");
    eprintln!("                 deg/s threshold value. Must be a positive number.");
    eprintln!("                 If --dps is omitted, a general step-response is shown.");
    eprintln!(
        "  --output-dir <directory>: Optional. Specifies the output directory for generated plots."
    );
    eprintln!("                         If omitted, plots are saved in the source folder (input file's directory).");
    eprintln!(
        "  --butterworth: Optional. Show Butterworth per-stage PT1 cutoffs for PT2/PT3/PT4 filters"
    );
    eprintln!("                 as gray curves/lines on gyro and D-term spectrum plots.");
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
    output_dir: Option<&Path>,
    debug_mode: bool,
    show_butterworth: bool,
) -> Result<(), Box<dyn Error>> {
    // --- Setup paths and names ---
    let input_path = Path::new(input_file_str);
    if !input_path.exists() {
        eprintln!("Error: Input file not found: {input_file_str}");
        return Ok(()); // Continue to next file if this one is not found
    }
    println!("\n--- Processing file: {input_file_str} ---");

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
                    dir_prefix_to_add = format!("{sanitized_dir}_");
                }
            }
        }
        root_name_string = format!("{dir_prefix_to_add}{file_stem_cow}");
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
        header_metadata,
    ) = match parse_log_file(input_path, debug_mode) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error parsing log file {input_file_str}: {e}");
            return Ok(()); // Continue to next file
        }
    };

    if all_log_data.is_empty() {
        println!("No valid data rows read from {input_file_str}, cannot generate plots.");
        return Ok(());
    }

    // Parse PID metadata from headers
    let pid_metadata = parse_pid_metadata(&header_metadata);

    let mut has_nonzero_f_term_data = [false; 3];
    for axis in 0..crate::axis_names::AXIS_NAMES.len() {
        if f_term_header_found[axis]
            && all_log_data
                .iter()
                .any(|row| row.f_term[axis].is_some_and(|val| val.abs() > 1e-9))
        {
            has_nonzero_f_term_data[axis] = true;
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
    for axis in 0..crate::axis_names::AXIS_NAMES.len() {
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
INFO ({input_file_str}): Skipping Step Response data collection: Setpoint or Gyro headers missing."
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
INFO ({input_file_str}): Skipping Step Response input data filtering: {reason}."
        );
    }

    println!(
        "
--- Calculating Step Response for {input_file_str} ---"
    );
    let mut step_response_calculation_results: StepResponseResults = [None, None, None];

    if let Some(sr) = sample_rate {
        for axis_index in 0..crate::axis_names::AXIS_NAMES.len() {
            let axis_name = crate::axis_names::AXIS_NAMES[axis_index];
            let required_headers_found =
                setpoint_header_found[axis_index] && gyro_header_found[axis_index];
            if required_headers_found && !contiguous_sr_input_data[axis_index].0.is_empty() {
                println!("  Attempting step response calculation for {axis_name}...");
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
                                println!("    ... Calculation successful for {axis_name}. {num_qc_windows} windows passed QC.");
                            } else {
                                println!("    ... Calculation returned no valid windows for {axis_name}. Skipping.");
                            }
                        }
                        Err(e) => {
                            eprintln!("    ... Calculation failed for {axis_name}: {e}");
                        }
                    }
                } else {
                    println!("    ... Insufficient data for {axis_name}. Skipping.");
                }
            } else {
                println!("    ... Required headers not found for {axis_name}. Skipping.");
            }
        }
    } else {
        println!("    ... No sample rate available. Skipping step response calculations.");
    }

    // Create RAII guard BEFORE changing directory if needed
    let _cwd_guard = if let Some(output_dir) = output_dir {
        // Create guard to save current directory BEFORE changing it
        let guard = CwdGuard::new()?;
        // Ensure output directory exists
        std::fs::create_dir_all(output_dir)?;
        // Change to output directory for plot generation
        env::set_current_dir(output_dir)?;
        Some(guard)
    } else {
        None
    };
    // CWD will be automatically restored when _cwd_guard goes out of scope

    // Create PID context for centralized PID metadata and related parameters
    let pid_context = PidContext::new(sample_rate, pid_metadata, root_name_string.clone());

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
        &pid_context.pid_metadata,
    )?;
    plot_gyro_spectrums(
        &all_log_data,
        &root_name_string,
        sample_rate,
        Some(&header_metadata),
        show_butterworth,
    )?;
    plot_d_term_psd(
        &all_log_data,
        &root_name_string,
        sample_rate,
        Some(&header_metadata),
        debug_mode,
    )?;
    plot_d_term_spectrums(
        &all_log_data,
        &root_name_string,
        sample_rate,
        Some(&header_metadata),
        show_butterworth,
    )?;
    plot_psd(&all_log_data, &root_name_string, sample_rate)?;
    plot_psd_db_heatmap(&all_log_data, &root_name_string, sample_rate)?;
    plot_throttle_freq_heatmap(&all_log_data, &root_name_string, sample_rate)?;
    plot_d_term_heatmap(&all_log_data, &root_name_string, sample_rate)?;

    // CWD restoration happens automatically when _cwd_guard goes out of scope
    println!("--- Finished processing file: {input_file_str} ---");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 2 {
        print_usage_and_exit(program_name);
    }

    let mut input_paths: Vec<String> = Vec::new();
    let mut setpoint_threshold_override: Option<f64> = None;
    let mut dps_flag_present = false;
    let mut output_dir: Option<String> = None; // None = not specified (use source folder), Some(dir) = --output-dir with value
    let mut debug_mode = false;
    let mut show_butterworth = false;

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
                    if val <= 0.0 {
                        eprintln!("Error: --dps threshold must be > 0: {val}");
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
                output_dir = Some(args[i + 1].clone());
                i += 1;
            }
        } else if arg == "--debug" {
            debug_mode = true;
        } else if arg == "--butterworth" {
            show_butterworth = true;
        } else if arg.starts_with("--") {
            eprintln!("Error: Unknown option '{arg}'");
            print_usage_and_exit(program_name);
        } else {
            input_paths.push(arg.clone());
        }
        i += 1;
    }

    if input_paths.is_empty() {
        eprintln!("Error: At least one input file or directory is required.");
        print_usage_and_exit(program_name);
    }

    // Expand input paths (files and directories) to a list of CSV files
    let input_files = match expand_input_paths(&input_paths) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("Error expanding input paths: {e}");
            std::process::exit(1);
        }
    };

    if input_files.is_empty() {
        eprintln!("Error: No CSV files found in the specified input paths.");
        std::process::exit(1);
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
                Path::new(input_file_str).parent()
            }
            Some(dir) => Some(Path::new(dir)), // --output-dir with specific directory
        };

        if let Err(e) = process_file(
            input_file_str,
            setpoint_threshold,
            show_legend,
            use_dir_prefix_for_root_name,
            actual_output_dir,
            debug_mode,
            show_butterworth,
        ) {
            eprintln!("An error occurred while processing {input_file_str}: {e}");
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
