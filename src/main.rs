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
fn expand_input_paths(input_paths: &[String]) -> Vec<String> {
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
            match find_csv_files_in_dir(input_path) {
                Ok(mut dir_csv_files) => csv_files.append(&mut dir_csv_files),
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

    csv_files
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
        println!("      Always test in a safe environment. Conservative = safer first step.");
        println!("      Aggressive = faster tuning for experienced pilots.");
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
                    let mask = Array1::ones(num_windows);

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
                                // Analyze overshoot/undershoot and calculate recommended ratio
                                // Peak ranges: 1.05-1.10 = excellent, 1.11-1.15 = good but improvable, >1.15 = needs more D
                                let (assessment, recommended_ratio) = if peak_value > 1.20 {
                                    // Severe overshoot - conservative increase (don't jump all the way)
                                    ("Significant overshoot", current_pd_ratio * 0.85)
                                // Increase D by ~18%
                                } else if peak_value > 1.15 {
                                    // Moderate overshoot - small increase
                                    ("Moderate overshoot", current_pd_ratio * 0.93)
                                // Increase D by ~8%
                                } else if peak_value > 1.10 {
                                    // Minor overshoot - tiny adjustment
                                    ("Minor overshoot", current_pd_ratio * 0.97)
                                // Increase D by ~3%
                                } else if peak_value >= 1.05 {
                                    // Excellent range - no change needed
                                    ("Excellent response", current_pd_ratio) // Perfect
                                } else if peak_value >= 0.95 {
                                    // Slight underdamping is acceptable
                                    ("Good response", current_pd_ratio) // Still good
                                } else if peak_value >= 0.85 {
                                    // Minor undershoot - small decrease
                                    ("Minor undershoot", current_pd_ratio * 1.05)
                                // Decrease D by ~5%
                                } else {
                                    // Significant undershoot - moderate decrease
                                    ("Significant undershoot", current_pd_ratio * 1.15)
                                    // Decrease D by ~13%
                                };
                                // store conservative recommendation for later use in plots
                                recommended_pd_conservative[axis_index] = Some(recommended_ratio);

                                // Calculate aggressive recommendation (more damping)
                                let aggressive_ratio = recommended_ratio
                                    * crate::constants::PD_RATIO_AGGRESSIVE_MULTIPLIER
                                    / crate::constants::PD_RATIO_CONSERVATIVE_MULTIPLIER;
                                recommended_pd_aggressive[axis_index] = Some(aggressive_ratio);

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

                                    // calculate recommended D (and D-Min/D-Max if applicable) for aggressive ratio
                                    let (rec_d_agg, rec_d_min_agg, rec_d_max_agg) =
                                        if axis_index == 0 {
                                            pid_metadata.roll.calculate_goal_d_with_range(
                                                aggressive_ratio,
                                                dmax_enabled,
                                            )
                                        } else {
                                            pid_metadata.pitch.calculate_goal_d_with_range(
                                                aggressive_ratio,
                                                dmax_enabled,
                                            )
                                        };
                                    recommended_d_aggressive[axis_index] = rec_d_agg;
                                    recommended_d_min_aggressive[axis_index] = rec_d_min_agg;
                                    recommended_d_max_aggressive[axis_index] = rec_d_max_agg;
                                }

                                println!("{axis_name}: Peak={peak_value:.3} → {assessment}");
                                if (recommended_ratio - current_pd_ratio).abs()
                                    > crate::constants::PD_RATIO_MIN_CHANGE_THRESHOLD
                                {
                                    let axis_pid = if axis_index == 0 {
                                        &pid_metadata.roll
                                    } else {
                                        &pid_metadata.pitch
                                    };

                                    if let Some(p_val) = axis_pid.p {
                                        // Check if D-Min/D-Max is enabled
                                        let dmax_enabled = pid_metadata.is_dmax_enabled();

                                        // Conservative recommendation
                                        let (rec_d, rec_d_min, rec_d_max) = axis_pid
                                            .calculate_goal_d_with_range(
                                                recommended_ratio,
                                                dmax_enabled,
                                            );

                                        // Aggressive recommendation
                                        let aggressive_ratio = recommended_ratio
                                            * crate::constants::PD_RATIO_AGGRESSIVE_MULTIPLIER
                                            / crate::constants::PD_RATIO_CONSERVATIVE_MULTIPLIER;
                                        let (rec_d_agg, rec_d_min_agg, rec_d_max_agg) = axis_pid
                                            .calculate_goal_d_with_range(
                                                aggressive_ratio,
                                                dmax_enabled,
                                            );

                                        println!("  Current P:D={current_pd_ratio:.2}");

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
                                        if recommended_ratio
                                            < crate::constants::MIN_REASONABLE_PD_RATIO
                                        {
                                            println!("  ⚠️  WARNING: Recommended P:D ratio ({recommended_ratio:.2}) is very low");
                                            println!(
                                                "      Consider increasing P instead of only adding D"
                                            );
                                        } else if recommended_ratio
                                            > crate::constants::MAX_REASONABLE_PD_RATIO
                                        {
                                            println!("  ⚠️  WARNING: Recommended P:D ratio ({recommended_ratio:.2}) is very high");
                                            println!("      Consider decreasing P or checking for overdamped response");
                                        }

                                        // Show conservative recommendation
                                        if dmax_enabled
                                            && (rec_d_min.is_some() || rec_d_max.is_some())
                                        {
                                            // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                                            let d_min_str = rec_d_min
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            let d_max_str = rec_d_max
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            println!("    Conservative: P:D={recommended_ratio:.2} → D-Min≈{d_min_str}, D-Max≈{d_max_str} (P={p_val})");
                                        } else if let Some(recommended_d) = rec_d {
                                            // D-Min/D-Max disabled: show only base D
                                            println!("    Conservative: P:D={recommended_ratio:.2} → D≈{recommended_d} (P={p_val})");
                                        }

                                        // Show aggressive recommendation
                                        if dmax_enabled
                                            && (rec_d_min_agg.is_some() || rec_d_max_agg.is_some())
                                        {
                                            // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                                            let d_min_str = rec_d_min_agg
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            let d_max_str = rec_d_max_agg
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            println!("    Aggressive:   P:D={aggressive_ratio:.2} → D-Min≈{d_min_str}, D-Max≈{d_max_str} (P={p_val})");
                                        } else if let Some(recommended_d_agg) = rec_d_agg {
                                            // D-Min/D-Max disabled: show only base D
                                            println!("    Aggressive:   P:D={aggressive_ratio:.2} → D≈{recommended_d_agg} (P={p_val})");
                                        }
                                    } else {
                                        println!("  Current P:D={current_pd_ratio:.2} → Recommendations available but P value missing");
                                    }
                                } else {
                                    println!("  Current P:D={current_pd_ratio:.2} seems fair");
                                }
                            }
                        }
                    }
                }
            }
        }
        println!();
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
        &recommended_pd_conservative,
        &recommended_d_conservative,
        &recommended_d_min_conservative,
        &recommended_d_max_conservative,
        &recommended_pd_aggressive,
        &recommended_d_aggressive,
        &recommended_d_min_aggressive,
        &recommended_d_max_aggressive,
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
    let input_files = expand_input_paths(&input_paths);

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
