// src/main.rs

mod axis_names;
mod constants;
mod data_analysis;
mod data_input;
mod debug_mode_lookup;
mod font_config;
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

// Build version string from git info with fallbacks for builds without vergen metadata
fn get_version_string() -> String {
    let sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    let date = option_env!("VERGEN_GIT_COMMIT_DATE").unwrap_or("unknown");
    format!("{sha} ({date})")
}

// Plot configuration struct
#[derive(Debug, Clone, Copy)]
struct PlotConfig {
    pub step_response: bool,
    pub pidsum_error_setpoint: bool,
    pub setpoint_vs_gyro: bool,
    pub setpoint_derivative: bool,
    pub gyro_vs_unfilt: bool,
    pub gyro_spectrums: bool,
    pub d_term_psd: bool,
    pub d_term_spectrums: bool,
    pub psd: bool,
    pub psd_db_heatmap: bool,
    pub throttle_freq_heatmap: bool,
    pub d_term_heatmap: bool,
    pub motor_spectrums: bool,
    pub bode: bool,
    pub pid_activity: bool,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            step_response: true,
            pidsum_error_setpoint: true,
            setpoint_vs_gyro: true,
            setpoint_derivative: true,
            gyro_vs_unfilt: true,
            gyro_spectrums: true,
            d_term_psd: true,
            d_term_spectrums: true,
            psd: true,
            psd_db_heatmap: true,
            throttle_freq_heatmap: true,
            d_term_heatmap: true,
            motor_spectrums: true,
            bode: false,
            pid_activity: true,
        }
    }
}

impl PlotConfig {
    fn none() -> Self {
        Self {
            step_response: false,
            pidsum_error_setpoint: false,
            setpoint_vs_gyro: false,
            setpoint_derivative: false,
            gyro_vs_unfilt: false,
            gyro_spectrums: false,
            d_term_psd: false,
            d_term_spectrums: false,
            psd: false,
            psd_db_heatmap: false,
            throttle_freq_heatmap: false,
            d_term_heatmap: false,
            motor_spectrums: false,
            bode: false,
            pid_activity: false,
        }
    }
}

// Analysis options struct to group related analysis parameters
#[derive(Debug, Clone, Copy)]
struct AnalysisOptions {
    pub setpoint_threshold: f64,
    pub show_legend: bool,
    pub debug_mode: bool,
    pub show_butterworth: bool,
    pub estimate_optimal_p: bool,
    pub frame_class: crate::data_analysis::optimal_p_estimation::FrameClass,
}

use crate::constants::{
    DEFAULT_SETPOINT_THRESHOLD, EXCLUDE_END_S, EXCLUDE_START_S, FRAME_LENGTH_S,
};

// Specific plot function imports
use crate::plot_functions::plot_bode::plot_bode_analysis;
use crate::plot_functions::plot_d_term_heatmap::plot_d_term_heatmap;
use crate::plot_functions::plot_d_term_psd::plot_d_term_psd;
use crate::plot_functions::plot_d_term_spectrums::plot_d_term_spectrums;
use crate::plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;
use crate::plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use crate::plot_functions::plot_motor_spectrums::plot_motor_spectrums;
use crate::plot_functions::plot_pid_activity::plot_pid_activity;
use crate::plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use crate::plot_functions::plot_psd::plot_psd;
use crate::plot_functions::plot_psd_db_heatmap::plot_psd_db_heatmap;
use crate::plot_functions::plot_setpoint_derivative::plot_setpoint_derivative;
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
/// If a path is a directory, find CSV files (optionally recursing into subdirectories).
/// Returns (csv_files, total_skipped_subdirectories)
fn expand_input_paths(
    input_paths: &[String],
    recursive: bool,
    debug_mode: bool,
) -> (Vec<String>, usize) {
    let mut csv_files = Vec::new();
    let mut total_skipped = 0;

    for input_path_str in input_paths {
        let input_path = Path::new(input_path_str);

        if input_path.is_file() {
            // It's a file, validate CSV extension before adding
            if let Some(extension) = input_path.extension() {
                if extension.to_string_lossy().eq_ignore_ascii_case("csv") {
                    // Skip header files (these are metadata files, not flight logs)
                    let lowercase_path = input_path_str.to_ascii_lowercase();
                    if lowercase_path.ends_with(".header.csv")
                        || lowercase_path.ends_with(".headers.csv")
                    {
                        eprintln!("Warning: Skipping header file: {}", input_path_str);
                    } else {
                        csv_files.push(input_path_str.clone());
                    }
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
            // It's a directory, find CSV files (recursive only if flag is set)
            match find_csv_files_in_dir(input_path, recursive, debug_mode) {
                Ok((mut dir_csv_files, skipped_count)) => {
                    csv_files.append(&mut dir_csv_files);
                    total_skipped += skipped_count;
                }
                Err(err) => eprintln!(
                    "Warning: Error processing directory {}: {}",
                    input_path_str, err
                ),
            }
        } else {
            // Path doesn't exist or isn't accessible
            eprintln!(
                "Warning: Path not found or not accessible: {}",
                input_path_str
            );
        }
    }

    (csv_files, total_skipped)
}

/// Find CSV files in a directory, optionally recursing into subdirectories
/// Returns (csv_files, skipped_subdirectories_count)
fn find_csv_files_in_dir(
    dir_path: &Path,
    recursive: bool,
    debug_mode: bool,
) -> Result<(Vec<String>, usize), Box<dyn Error>> {
    let mut visited = HashSet::new();
    find_csv_files_in_dir_impl(dir_path, &mut visited, recursive, debug_mode)
}

/// Internal implementation with symlink loop protection
/// Returns (csv_files, skipped_subdirectories_count)
fn find_csv_files_in_dir_impl(
    dir_path: &Path,
    visited: &mut HashSet<PathBuf>,
    recursive: bool,
    debug_mode: bool,
) -> Result<(Vec<String>, usize), Box<dyn Error>> {
    let mut csv_files = Vec::new();
    let mut skipped_count = 0;

    if !dir_path.is_dir() {
        return Ok((csv_files, skipped_count));
    }

    // Canonicalize path to detect symlink loops
    let canonical_path = match dir_path.canonicalize() {
        Ok(path) => path,
        Err(_) => {
            eprintln!(
                "Warning: Cannot canonicalize directory path: {}",
                dir_path.display()
            );
            return Ok((csv_files, skipped_count));
        }
    };

    // Check if we've already visited this directory (symlink loop detection)
    if visited.contains(&canonical_path) {
        eprintln!(
            "Warning: Skipping directory due to symlink loop: {}",
            dir_path.display()
        );
        return Ok((csv_files, skipped_count));
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
            return Ok((csv_files, skipped_count));
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
            // Recurse into subdirectories only if recursive flag is set
            if recursive {
                match find_csv_files_in_dir_impl(&path, visited, recursive, debug_mode) {
                    Ok((mut sub_csv_files, sub_skipped)) => {
                        csv_files.append(&mut sub_csv_files);
                        skipped_count += sub_skipped;
                    }
                    Err(err) => eprintln!(
                        "Warning: Error processing subdirectory '{}': {}",
                        path.display(),
                        err
                    ),
                }
            } else {
                // Skip this subdirectory
                skipped_count += 1;
                if debug_mode {
                    // Show skip message in debug mode
                    eprintln!(
                        "Note: Skipping subdirectory '{}' (use --recursive to include subdirectories)",
                        path.display()
                    );
                }
            }
        } else if path.is_file() {
            // Check if it's a CSV file
            if let Some(extension) = path.extension() {
                if extension.to_string_lossy().eq_ignore_ascii_case("csv") {
                    // Skip header files (these are metadata files, not flight logs)
                    if let Some(path_str) = path.to_str() {
                        let lowercase = path_str.to_ascii_lowercase();
                        if lowercase.ends_with(".header.csv") || lowercase.ends_with(".headers.csv")
                        {
                            eprintln!("Warning: Skipping header file: {}", path_str);
                        } else {
                            csv_files.push(path_str.to_string());
                        }
                    } else {
                        eprintln!(
                            "Warning: Skipping file with non-UTF-8 path: {}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    // Sort the files for consistent ordering
    csv_files.sort();
    Ok((csv_files, skipped_count))
}

fn print_usage_and_exit(program_name: &str) {
    eprintln!("Graphically render statistical data from Blackbox CSV.");
    eprintln!("
Usage: {program_name} <input1> [<input2> ...] [-O|--output-dir <directory>] [--bode] [--butterworth] [--debug] [--dps <value>] [--estimate-optimal-p] [--prop-size <size>] [--motor] [--pid] [-R|--recursive] [--setpoint] [--step]");
    eprintln!("  <inputX>: One or more input CSV files, directories, or shell-expanded wildcards (required).");
    eprintln!("            Can mix files and directories in a single command.");
    eprintln!("            - Individual CSV file: path/to/file.csv");
    eprintln!("            - Directory: path/to/dir/ (finds CSV files only in that directory)");
    eprintln!("            - Wildcards: *.csv, *LOG*.csv (shell-expanded; works with mixed file and directory patterns)");
    eprintln!(
        "            Note: Header files (.header.csv, .headers.csv) are automatically excluded."
    );
    eprintln!(
        "  -O, --output-dir <directory>: Optional. Specifies the output directory for generated plots."
    );
    eprintln!("                              If omitted, plots are saved in the source folder (input directory).");
    eprintln!("  --bode: Optional. Generate Bode plot analysis (magnitude, phase, coherence).");
    eprintln!("          NOTE: Requires controlled test flights with system-identification inputs");
    eprintln!("          (chirp/PRBS). Not recommended for normal flight logs.");
    eprintln!(
        "  --butterworth: Optional. Show Butterworth per-stage PT1 cutoffs for PT2/PT3/PT4 filters"
    );
    eprintln!("                 as gray curves/lines on gyro and D-term spectrum plots.");
    eprintln!("  --debug: Optional. Shows detailed metadata information during processing.");
    eprintln!("  --dps <value>: Optional. Enables detailed step response plots with the specified");
    eprintln!("                 deg/s threshold value. Must be a positive number.");
    eprintln!("                 If --dps is omitted, a general step-response is shown.");
    eprintln!(
        "  --estimate-optimal-p: Optional. Enable optimal P estimation with physics-aware recommendations."
    );
    eprintln!(
        "                        Analyzes response time vs. frame-class targets and noise levels."
    );
    eprintln!(
        "  --prop-size <size>: Optional. Specify propeller diameter in inches for optimal P estimation."
    );
    eprintln!("                      Valid options: 1-15 (match your actual PROPELLER size)");
    eprintln!(
        "                      Defaults to 5 if --estimate-optimal-p is used without this flag."
    );
    eprintln!(
        "                      Note: This flag is only applied when --estimate-optimal-p is enabled."
    );
    eprintln!("                      Example: 6-inch frame with 5-inch props → use --prop-size 5");
    eprintln!(
        "                      If --prop-size is provided without --estimate-optimal-p, a warning"
    );
    eprintln!("                      will be shown and the prop size setting will be ignored.");
    eprintln!(
        "  --motor: Optional. Generate only motor spectrum plots, skipping all other graphs."
    );
    eprintln!("  --pid: Optional. Generate only P, I, D activity stacked plot (showing all three PID terms over time).");
    eprintln!("  -R, --recursive: Optional. When processing directories, recursively find CSV files in subdirectories.");
    eprintln!(
        "  --setpoint: Optional. Generate only setpoint-related plots (PIDsum, Setpoint vs Gyro, Setpoint Derivative)."
    );
    eprintln!("  --step: Optional. Generate only step response plots, skipping all other graphs.");
    eprintln!("  -h, --help: Show this help message and exit.");
    eprintln!("  -V, --version: Show version information and exit.");
    eprintln!(
        "
Arguments can be in any order. Wildcards (e.g., *.csv) are shell-expanded and work with mixed file/directory patterns."
    );
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {program_name} flight.csv");
    eprintln!("  {program_name} flight.csv --dps 200");
    eprintln!("  {program_name} flight.csv --step --estimate-optimal-p --prop-size 5");
    eprintln!("  {program_name} input/*.csv -O ./output/");
    eprintln!("  {program_name} logs/ -R --step");
    std::process::exit(1);
}

fn process_file(
    input_file_str: &str,
    use_dir_prefix: bool,
    output_dir: Option<&Path>,
    plot_config: PlotConfig,
    analysis_opts: AnalysisOptions,
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
        gyro_unfilt_header_found,
        debug_header_found,
        header_metadata,
    ) = match parse_log_file(input_path, analysis_opts.debug_mode) {
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

    // Display P:D ratios for tuning analysis (Roll and Pitch only)
    // Note: Each aircraft may have different optimal P:D ratios depending on frame,
    // motors, props, etc. Goal ratio should be determined from step response analysis.
    println!("\n--- PID Tuning Analysis ---");

    // Roll axis
    if let Some(roll_ratio) = pid_metadata.roll.calculate_pd_ratio() {
        println!("Roll P:D Ratio: {:.2}", roll_ratio);
    } else {
        println!("Roll P:D Ratio: N/A (insufficient PID data)");
    }

    // Pitch axis
    if let Some(pitch_ratio) = pid_metadata.pitch.calculate_pd_ratio() {
        println!("Pitch P:D Ratio: {:.2}", pitch_ratio);
    } else {
        println!("Pitch P:D Ratio: N/A (insufficient PID data)");
    }

    // Yaw axis (informational only - different tuning philosophy)
    if let Some(yaw_ratio) = pid_metadata.yaw.calculate_pd_ratio() {
        println!(
            "Yaw P:D Ratio: {:.2} (informational - yaw tuning differs from roll/pitch)",
            yaw_ratio
        );
    } else {
        println!("Yaw P:D Ratio: N/A (yaw often uses minimal or no D-term)");
    }
    println!("Note: Optimal P:D ratio varies per aircraft. Check step response for overshoot/undershoot.");
    println!();

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

    // Store peak values and current state for display
    let mut peak_values: [Option<f64>; 3] = [None, None, None];
    let mut current_pd_ratios: [Option<f64>; 3] = [None, None, None];
    let mut assessments: [Option<&str>; 3] = [None, None, None];

    // Conservative recommendations (based on peak response analysis)
    let mut recommended_pd_conservative: [Option<f64>; 3] = [None, None, None];
    let mut recommended_d_conservative: [Option<u32>; 3] = [None, None, None];
    let mut recommended_d_min_conservative: [Option<u32>; 3] = [None, None, None];
    let mut recommended_d_max_conservative: [Option<u32>; 3] = [None, None, None];

    // Aggressive recommendations (more D for better damping)
    let mut recommended_pd_aggressive: [Option<f64>; 3] = [None, None, None];
    let mut recommended_d_aggressive: [Option<u32>; 3] = [None, None, None];
    let mut recommended_d_min_aggressive: [Option<u32>; 3] = [None, None, None];
    let mut recommended_d_max_aggressive: [Option<u32>; 3] = [None, None, None];

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

    // Analyze step response and provide P:D ratio recommendations based on overshoot/undershoot
    if sample_rate.is_some() {
        println!("\n--- Step Response Analysis & P:D Ratio Recommendations ---");
        println!("NOTE: These are STARTING POINTS based on step response analysis.");
        println!("      These recommendations focus on D-term tuning (P:D ratio).");
        if analysis_opts.estimate_optimal_p {
            println!(
                "      See 'Optimal P Estimation' below for P gain magnitude recommendations."
            );
        }
        println!("      Always test in a safe environment. Conservative = safer first step.");
        println!("      Moderate = for experienced pilots (test carefully to avoid hot motors).");
        println!();
        for axis_index in 0..2 {
            // Only Roll (0) and Pitch (1)
            let axis_name = crate::axis_names::AXIS_NAMES[axis_index];

            if let Some((response_time, valid_stacked_responses, _valid_window_max_setpoints)) =
                &step_response_calculation_results[axis_index]
            {
                if valid_stacked_responses.shape()[0] > 0 && !response_time.is_empty() {
                    // Calculate average response to analyze
                    let num_windows = valid_stacked_responses.shape()[0];
                    let response_length = response_time.len();
                    let mask: Array1<f32> = Array1::ones(num_windows);

                    if let Ok(avg_response) = calc_step_response::average_responses(
                        valid_stacked_responses,
                        &mask,
                        response_length,
                    ) {
                        // Find peak value
                        if let Some(peak_value) = calc_step_response::find_peak_value(&avg_response)
                        {
                            let current_ratio = if axis_index == 0 {
                                pid_metadata.roll.calculate_pd_ratio()
                            } else {
                                pid_metadata.pitch.calculate_pd_ratio()
                            };

                            if let Some(current_pd_ratio) = current_ratio {
                                // Analyze overshoot/undershoot based on peak response and calculate recommended ratio
                                // Peak ranges:
                                //   0.95-1.05 = optimal (0-5% overshoot/undershoot)
                                //   1.05-1.10 = acceptable (5-10% overshoot, improvable)
                                //   1.10-1.15 = minor overshoot (11-15%, needs improvement)
                                //   >1.15     = moderate/severe overshoot (needs significant D increase)
                                let (assessment, recommended_ratio) = if peak_value
                                    > crate::constants::PEAK_SIGNIFICANT_MIN
                                {
                                    // Significant overshoot (>20%) - use conservative multiplier
                                    (
                                        "Significant overshoot",
                                        current_pd_ratio
                                            * crate::constants::PD_RATIO_CONSERVATIVE_MULTIPLIER,
                                    )
                                } else if peak_value > crate::constants::PEAK_MODERATE_MIN {
                                    // Moderate overshoot (16-20%) - graduated adjustment
                                    (
                                        "Moderate overshoot",
                                        current_pd_ratio
                                            * crate::constants::PEAK_MODERATE_MULTIPLIER,
                                    )
                                } else if peak_value > crate::constants::PEAK_ACCEPTABLE_MAX {
                                    // Minor overshoot (11-15%) - smaller adjustment
                                    (
                                        "Minor overshoot",
                                        current_pd_ratio * crate::constants::PEAK_MINOR_MULTIPLIER,
                                    )
                                } else if peak_value >= crate::constants::PEAK_ACCEPTABLE_MIN {
                                    // Acceptable (5-10% overshoot) - minimal adjustment
                                    (
                                        "Acceptable response",
                                        current_pd_ratio
                                            * crate::constants::PEAK_ACCEPTABLE_MULTIPLIER,
                                    )
                                } else if peak_value >= crate::constants::PEAK_OPTIMAL_MIN {
                                    // Optimal (0-5% overshoot/undershoot) - no change
                                    ("Optimal response", current_pd_ratio)
                                } else if peak_value >= 0.85 {
                                    // Minor undershoot (6-15%) - small decrease
                                    ("Minor undershoot", current_pd_ratio * 1.05)
                                } else {
                                    // Significant undershoot (>15%) - moderate decrease
                                    ("Significant undershoot", current_pd_ratio * 1.15)
                                };

                                // Store peak value, current P:D ratio, and assessment for plot legends
                                peak_values[axis_index] = Some(peak_value);
                                current_pd_ratios[axis_index] = Some(current_pd_ratio);
                                assessments[axis_index] = Some(assessment);

                                // Only store recommendations if change exceeds threshold
                                // (to avoid showing recommendations for already-good responses)
                                if (recommended_ratio - current_pd_ratio).abs()
                                    > crate::constants::PD_RATIO_MIN_CHANGE_THRESHOLD
                                {
                                    // store conservative recommendation for later use in plots
                                    recommended_pd_conservative[axis_index] =
                                        Some(recommended_ratio);

                                    // Calculate moderate recommendation ONLY for moderate/significant overshoot (>1.15)
                                    // For acceptable/minor overshoot (1.05-1.15), show conservative only
                                    let moderate_ratio =
                                        if peak_value > crate::constants::PEAK_MINOR_MAX {
                                            let ratio = recommended_ratio
                                            * crate::constants::PD_RATIO_MODERATE_MULTIPLIER
                                            / crate::constants::PD_RATIO_CONSERVATIVE_MULTIPLIER;
                                            recommended_pd_aggressive[axis_index] = Some(ratio);
                                            Some(ratio)
                                        } else {
                                            None
                                        };

                                    if let Some(_axis_pid_p) = if axis_index == 0 {
                                        pid_metadata.roll.p
                                    } else {
                                        pid_metadata.pitch.p
                                    } {
                                        // Check if D-Min/D-Max system is enabled
                                        let dmax_enabled = pid_metadata.is_dmax_enabled();

                                        // calculate recommended D (and D-Min/D-Max if applicable) for conservative ratio
                                        let (rec_d, rec_d_min, rec_d_max) = if axis_index == 0 {
                                            pid_metadata.roll.calculate_goal_d_with_range(
                                                recommended_ratio,
                                                dmax_enabled,
                                            )
                                        } else {
                                            pid_metadata.pitch.calculate_goal_d_with_range(
                                                recommended_ratio,
                                                dmax_enabled,
                                            )
                                        };
                                        recommended_d_conservative[axis_index] = rec_d;
                                        recommended_d_min_conservative[axis_index] = rec_d_min;
                                        recommended_d_max_conservative[axis_index] = rec_d_max;

                                        // calculate recommended D (and D-Min/D-Max if applicable) for moderate ratio
                                        // ONLY if moderate ratio was calculated (peak > 1.15)
                                        if let Some(mod_ratio) = moderate_ratio {
                                            let (rec_d_agg, rec_d_min_agg, rec_d_max_agg) =
                                                if axis_index == 0 {
                                                    pid_metadata.roll.calculate_goal_d_with_range(
                                                        mod_ratio,
                                                        dmax_enabled,
                                                    )
                                                } else {
                                                    pid_metadata.pitch.calculate_goal_d_with_range(
                                                        mod_ratio,
                                                        dmax_enabled,
                                                    )
                                                };
                                            recommended_d_aggressive[axis_index] = rec_d_agg;
                                            recommended_d_min_aggressive[axis_index] =
                                                rec_d_min_agg;
                                            recommended_d_max_aggressive[axis_index] =
                                                rec_d_max_agg;
                                        }
                                    }
                                }

                                println!("{axis_name}: Peak={peak_value:.3} → {assessment}");

                                // Always show current P:D ratio with quality assessment
                                let axis_pid = if axis_index == 0 {
                                    &pid_metadata.roll
                                } else {
                                    &pid_metadata.pitch
                                };

                                if let Some(p_val) = axis_pid.p {
                                    println!("  Current P:D={current_pd_ratio:.2}");

                                    // Show recommendations if they were computed (threshold exceeded)
                                    if recommended_pd_conservative[axis_index].is_some() {
                                        // Check for extreme overshoot (may indicate deeper issues)
                                        if peak_value > crate::constants::SEVERE_OVERSHOOT_THRESHOLD
                                        {
                                            println!("  ⚠️  WARNING: Severe overshoot (Peak={peak_value:.2}) may indicate:");
                                            println!(
                                                "      - P value too high, or mechanical issues"
                                            );
                                            println!(
                                                "      - Check for bent props, loose hardware, or damaged motors"
                                            );
                                        }

                                        // Check for unreasonable P:D ratios
                                        let rec_ratio =
                                            recommended_pd_conservative[axis_index].unwrap();
                                        if rec_ratio < crate::constants::MIN_REASONABLE_PD_RATIO {
                                            println!("  ⚠️  WARNING: Recommended P:D ratio ({rec_ratio:.2}) is very low");
                                            println!(
                                                "      Consider increasing P instead of only adding D"
                                            );
                                        } else if rec_ratio
                                            > crate::constants::MAX_REASONABLE_PD_RATIO
                                        {
                                            println!("  ⚠️  WARNING: Recommended P:D ratio ({rec_ratio:.2}) is very high");
                                            println!("      Consider decreasing P or checking for overdamped response");
                                        }

                                        // Check if D-Min/D-Max is enabled
                                        let dmax_enabled = pid_metadata.is_dmax_enabled();

                                        // Show conservative recommendation
                                        if dmax_enabled
                                            && (recommended_d_min_conservative[axis_index]
                                                .is_some()
                                                || recommended_d_max_conservative[axis_index]
                                                    .is_some())
                                        {
                                            // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                                            let d_min_str = recommended_d_min_conservative
                                                [axis_index]
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            let d_max_str = recommended_d_max_conservative
                                                [axis_index]
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            println!("    Conservative recommendation: P:D={:.2} → D-Min≈{}, D-Max≈{} (P={})",
                                                recommended_pd_conservative[axis_index].unwrap(),
                                                d_min_str, d_max_str, p_val);
                                        } else if let Some(recommended_d) =
                                            recommended_d_conservative[axis_index]
                                        {
                                            // D-Min/D-Max disabled: show only base D
                                            println!(
                                                "    Conservative recommendation: P:D={:.2} → D≈{} (P={})",
                                                recommended_pd_conservative[axis_index].unwrap(),
                                                recommended_d,
                                                p_val
                                            );
                                        }

                                        // Show moderate recommendation
                                        if dmax_enabled
                                            && (recommended_d_min_aggressive[axis_index].is_some()
                                                || recommended_d_max_aggressive[axis_index]
                                                    .is_some())
                                        {
                                            // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                                            let d_min_str = recommended_d_min_aggressive
                                                [axis_index]
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            let d_max_str = recommended_d_max_aggressive
                                                [axis_index]
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            println!("    Moderate recommendation:     P:D={:.2} → D-Min≈{}, D-Max≈{} (P={})",
                                                recommended_pd_aggressive[axis_index].unwrap(),
                                                d_min_str, d_max_str, p_val);
                                        } else if let Some(recommended_d_mod) =
                                            recommended_d_aggressive[axis_index]
                                        {
                                            // D-Min/D-Max disabled: show only base D
                                            println!(
                                                "    Moderate recommendation:     P:D={:.2} → D≈{} (P={})",
                                                recommended_pd_aggressive[axis_index].unwrap(),
                                                recommended_d_mod,
                                                p_val
                                            );
                                        }
                                    } else {
                                        // No recommendations needed - response is already good
                                        println!(
                                            "  ({assessment} - no obvious tuning adjustments needed)"
                                        );
                                    }
                                } else {
                                    println!(
                                        "  (P value missing - cannot calculate recommendations)"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        println!();
    }

    // Optimal P Estimation Analysis (if enabled)
    // Store results for both console output and PNG overlay
    let mut optimal_p_analyses: [Option<
        crate::data_analysis::optimal_p_estimation::OptimalPAnalysis,
    >; 3] = [None, None, None];

    if analysis_opts.estimate_optimal_p {
        if let Some(sr) = sample_rate {
            println!("\n--- Optimal P Estimation ---");
            println!(
                "Prop size: {} (use --prop-size to override)",
                analysis_opts.frame_class.name()
            );
            println!();

            for axis_index in 0..crate::axis_names::ROLL_PITCH_AXIS_COUNT {
                // Only Roll (0) and Pitch (1) - Yaw excluded by ROLL_PITCH_AXIS_COUNT
                let axis_name = crate::axis_names::AXIS_NAMES[axis_index];

                if let Some((response_time, valid_stacked_responses, _valid_window_max_setpoints)) =
                    &step_response_calculation_results[axis_index]
                {
                    if valid_stacked_responses.shape()[0] > 0 && !response_time.is_empty() {
                        // Collect individual Td samples from each valid response window
                        let mut td_samples_ms: Vec<f64> = Vec::new();

                        for window_idx in 0..valid_stacked_responses.shape()[0] {
                            let response = valid_stacked_responses.row(window_idx);
                            let response_f64: Vec<f64> =
                                response.iter().map(|&x| x as f64).collect();
                            let response_arr = Array1::from_vec(response_f64);

                            if let Some(td_seconds) =
                                calc_step_response::calculate_delay_time(&response_arr, sr)
                            {
                                td_samples_ms.push(
                                    td_seconds
                                        * crate::constants::OPTIMAL_P_SECONDS_TO_MS_MULTIPLIER,
                                );
                            }
                        }

                        if td_samples_ms.is_empty() {
                            println!("  No valid Td measurements for {axis_name}. Skipping optimal P analysis.");
                            continue;
                        }

                        // Get current P gain
                        let current_p = if axis_index == 0 {
                            pid_metadata.roll.p
                        } else {
                            pid_metadata.pitch.p
                        };

                        // Get current D gain
                        let current_d = if axis_index == 0 {
                            pid_metadata.roll.d
                        } else {
                            pid_metadata.pitch.d
                        };

                        if let Some(p_gain) = current_p {
                            // Calculate HF noise energy from D-term data if available
                            let hf_energy_ratio: Option<f64> = {
                                // Collect D-term data for this axis from the log
                                let d_term_data: Vec<f32> = all_log_data
                                    .iter()
                                    .filter_map(|row| row.d_term[axis_index].map(|v| v as f32))
                                    .collect();

                                // Only analyze if we have sufficient D-term data and sample rate
                                if !d_term_data.is_empty()
                                    && d_term_data.len()
                                        >= crate::constants::OPTIMAL_P_MIN_DTERM_SAMPLES
                                {
                                    crate::data_analysis::spectral_analysis::calculate_hf_energy_ratio(
                                        &d_term_data,
                                        sr,
                                        crate::constants::DTERM_HF_CUTOFF_HZ,
                                    )
                                } else {
                                    None
                                }
                            };

                            // Perform optimal P analysis
                            if let Some(analysis) = crate::data_analysis::optimal_p_estimation::OptimalPAnalysis::analyze(
                            &td_samples_ms,
                            p_gain,
                            current_d,
                            analysis_opts.frame_class,
                            hf_energy_ratio,
                            recommended_pd_conservative[axis_index],
                        ) {
                            // Print console output
                            println!("{}", analysis.format_console_output(axis_name));
                            // Store for PNG overlay (move instead of clone)
                            optimal_p_analyses[axis_index] = Some(analysis);
                        }
                        } else {
                            println!("  P gain not available for {axis_name}. Skipping optimal P analysis.");
                        }
                    }
                }
            }
            println!();
        }
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

    if plot_config.step_response {
        plot_step_response(
            &step_response_calculation_results,
            &root_name_string,
            sample_rate,
            &has_nonzero_f_term_data,
            analysis_opts.setpoint_threshold,
            analysis_opts.show_legend,
            &pid_context.pid_metadata,
            &peak_values,
            &current_pd_ratios,
            &assessments,
            &recommended_pd_conservative,
            &recommended_d_conservative,
            &recommended_d_min_conservative,
            &recommended_d_max_conservative,
            &recommended_pd_aggressive,
            &recommended_d_aggressive,
            &recommended_d_min_aggressive,
            &recommended_d_max_aggressive,
            &optimal_p_analyses,
            analysis_opts.estimate_optimal_p,
        )?;
    }

    if plot_config.pidsum_error_setpoint {
        plot_pidsum_error_setpoint(&all_log_data, &root_name_string)?;
    }

    if plot_config.setpoint_vs_gyro {
        plot_setpoint_vs_gyro(&all_log_data, &root_name_string, sample_rate)?;
    }

    if plot_config.setpoint_derivative {
        plot_setpoint_derivative(&all_log_data, &root_name_string, sample_rate)?;
    }

    // Determine if debug fallback is being used for gyroUnfilt
    let using_debug_fallback = !gyro_unfilt_header_found.iter().any(|&found| found)
        && debug_header_found.iter().take(3).any(|&found| found);

    // Get debug mode name if available
    let debug_mode_label = if using_debug_fallback {
        header_metadata
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("debug_mode"))
            .and_then(|(_, value)| value.parse::<u32>().ok())
            .and_then(|debug_value| {
                header_metadata
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case("Firmware revision"))
                    .and_then(|(_, fw_revision)| {
                        crate::debug_mode_lookup::lookup_debug_mode(fw_revision, debug_value)
                    })
            })
    } else {
        None
    };

    if plot_config.gyro_vs_unfilt {
        plot_gyro_vs_unfilt(
            &all_log_data,
            &root_name_string,
            sample_rate,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.gyro_spectrums {
        plot_gyro_spectrums(
            &all_log_data,
            &root_name_string,
            sample_rate,
            Some(&header_metadata),
            analysis_opts.show_butterworth,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.d_term_psd {
        plot_d_term_psd(
            &all_log_data,
            &root_name_string,
            sample_rate,
            Some(&header_metadata),
            analysis_opts.debug_mode,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.d_term_spectrums {
        plot_d_term_spectrums(
            &all_log_data,
            &root_name_string,
            sample_rate,
            Some(&header_metadata),
            analysis_opts.show_butterworth,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.motor_spectrums {
        plot_motor_spectrums(&all_log_data, &root_name_string, sample_rate)?;
    }

    if plot_config.psd {
        plot_psd(
            &all_log_data,
            &root_name_string,
            sample_rate,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.bode {
        eprintln!();
        eprintln!("⚠️  WARNING: Bode plots are designed for controlled test flights with system-identification inputs.");
        eprintln!(
            "    For normal flight log analysis, use spectrum plots (default behavior) instead."
        );
        eprintln!();
        plot_bode_analysis(
            &all_log_data,
            &root_name_string,
            sample_rate,
            analysis_opts.debug_mode,
        )?;
    }

    if plot_config.psd_db_heatmap {
        plot_psd_db_heatmap(
            &all_log_data,
            &root_name_string,
            sample_rate,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.throttle_freq_heatmap {
        plot_throttle_freq_heatmap(&all_log_data, &root_name_string, sample_rate)?;
    }

    if plot_config.d_term_heatmap {
        plot_d_term_heatmap(
            &all_log_data,
            &root_name_string,
            sample_rate,
            using_debug_fallback,
            debug_mode_label,
        )?;
    }

    if plot_config.pid_activity {
        plot_pid_activity(&all_log_data, &root_name_string, Some(&header_metadata))?;
    }

    // CWD restoration happens automatically when _cwd_guard goes out of scope
    println!("--- Finished processing file: {input_file_str} ---");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Print version at start of every execution
    println!("{} {}", env!("CARGO_PKG_NAME"), get_version_string());
    println!();

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
    let mut plot_config = PlotConfig::default();
    let mut has_only_flags = false;
    let mut step_requested = false;
    let mut motor_requested = false;
    let mut setpoint_requested = false;
    let mut bode_requested = false;
    let mut pid_requested = false;
    let mut recursive = false;
    let mut estimate_optimal_p = false;
    let mut frame_class_override: Option<crate::data_analysis::optimal_p_estimation::FrameClass> =
        None;

    let mut version_flag_set = false;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--help" || arg == "-h" {
            print_usage_and_exit(program_name);
        } else if arg == "--version" || arg == "-V" {
            version_flag_set = true;
        } else if arg == "--recursive" || arg == "-R" {
            recursive = true;
        } else if arg == "--dps" {
            if dps_flag_present {
                eprintln!("Error: --dps argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            if i + 1 >= args.len() {
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
        } else if arg == "--output-dir" || arg == "-O" {
            if output_dir.is_some() {
                eprintln!("Error: --output-dir/-O argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            if i + 1 >= args.len() {
                eprintln!("Error: --output-dir/-O requires a directory path.");
                print_usage_and_exit(program_name);
            } else {
                output_dir = Some(args[i + 1].clone());
                i += 1;
            }
        } else if arg == "--debug" {
            debug_mode = true;
        } else if arg == "--butterworth" {
            show_butterworth = true;
        } else if arg == "--step" {
            has_only_flags = true;
            step_requested = true;
        } else if arg == "--motor" {
            has_only_flags = true;
            motor_requested = true;
        } else if arg == "--setpoint" {
            has_only_flags = true;
            setpoint_requested = true;
        } else if arg == "--bode" {
            has_only_flags = true;
            bode_requested = true;
        } else if arg == "--pid" {
            has_only_flags = true;
            pid_requested = true;
        } else if arg == "--estimate-optimal-p" {
            estimate_optimal_p = true;
        } else if arg == "--prop-size" {
            if frame_class_override.is_some() {
                eprintln!("Error: --prop-size argument specified more than once.");
                print_usage_and_exit(program_name);
            }
            if i + 1 >= args.len() {
                eprintln!(
                    "Error: --prop-size requires a numeric value (propeller diameter in inches: 1-13)."
                );
                print_usage_and_exit(program_name);
            } else {
                let fc_str = args[i + 1].trim();
                match fc_str.parse::<u8>() {
                    Ok(size) => {
                        match crate::data_analysis::optimal_p_estimation::FrameClass::from_inches(
                            size,
                        ) {
                            Some(fc) => frame_class_override = Some(fc),
                            None => {
                                eprintln!("Error: Invalid prop size '{}'. Valid options: 1-15 (propeller diameter in inches)", fc_str);
                                print_usage_and_exit(program_name);
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("Error: Invalid prop size '{}'. Valid options: 1-15 (propeller diameter in inches)", fc_str);
                        print_usage_and_exit(program_name);
                    }
                }
                i += 1;
            }
        } else if arg.starts_with("--") {
            eprintln!("Error: Unknown option '{arg}'");
            print_usage_and_exit(program_name);
        } else {
            input_paths.push(arg.clone());
        }
        i += 1;
    }

    // Apply "only" flags if any were specified (non-mutually exclusive: OR together)
    if has_only_flags {
        plot_config = PlotConfig::none();
        plot_config.step_response = step_requested;
        plot_config.motor_spectrums = motor_requested;
        plot_config.bode = bode_requested;
        plot_config.pid_activity = pid_requested;
        if setpoint_requested {
            plot_config.pidsum_error_setpoint = true;
            plot_config.setpoint_vs_gyro = true;
            plot_config.setpoint_derivative = true;
        }
    }

    // Show debug information when the runtime --debug flag is present
    if debug_mode {
        println!(
            "DEBUG: has_only_flags={}, step_requested={}, motor_requested={}, setpoint_requested={}, plot_config={:?}",
            has_only_flags, step_requested, motor_requested, setpoint_requested, plot_config
        );
    }

    // Exit if only --version flag was set
    if version_flag_set {
        return Ok(());
    }

    // Warn if --prop-size is specified without --estimate-optimal-p
    if frame_class_override.is_some() && !estimate_optimal_p {
        eprintln!("Warning: --prop-size specified without --estimate-optimal-p.");
        eprintln!("         The prop size setting will be ignored.");
        eprintln!("         Use --estimate-optimal-p to enable optimal P estimation.");
        eprintln!();
    }

    if input_paths.is_empty() {
        eprintln!("Error: At least one input file or directory is required.");
        print_usage_and_exit(program_name);
    }

    // Expand input paths (files and directories) to a list of CSV files
    let (input_files, skipped_subdirs) = expand_input_paths(&input_paths, recursive, debug_mode);

    // Print summary of skipped subdirectories when not using recursive mode
    if !recursive && skipped_subdirs > 0 {
        let plural = if skipped_subdirs == 1 {
            "subdirectory"
        } else {
            "subdirectories"
        };
        eprintln!(
            "Note: Skipped {} {} (use --recursive to include subdirectories)",
            skipped_subdirs, plural
        );
    }

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

    // Construct AnalysisOptions once before the loop (Copy type, reusable across all files)
    let analysis_opts = AnalysisOptions {
        setpoint_threshold,
        show_legend,
        debug_mode,
        show_butterworth,
        estimate_optimal_p,
        frame_class: frame_class_override
            .unwrap_or(crate::data_analysis::optimal_p_estimation::FrameClass::FiveInch),
    };

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
            use_dir_prefix_for_root_name,
            actual_output_dir,
            plot_config,
            analysis_opts,
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
