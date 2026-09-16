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
mod report;
mod types;

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array1;

use crate::axis_names::AXIS_COUNT;
use crate::data_analysis::torque_inertia_profiler::{extract_punch_ratios, AircraftProfile};
use crate::types::{LogParseResult, StepResponseResults};

// Build version string from git info with fallbacks for builds without vergen metadata
fn get_version_string() -> String {
    let sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    let date = option_env!("VERGEN_GIT_COMMIT_DATE").unwrap_or("unknown");
    format!("{sha} ({date})")
}

// Plot configuration struct
// NOTE: When adding a field here, update none(), Default::default(), and all() accordingly.
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
    pub rc_command_activity: bool,
    pub motor_erpm: bool,
}

impl Default for PlotConfig {
    /// Core plots only (enabled by default)
    fn default() -> Self {
        Self {
            step_response: true,
            pidsum_error_setpoint: false,
            setpoint_vs_gyro: true,
            setpoint_derivative: false,
            gyro_vs_unfilt: true,
            gyro_spectrums: true,
            d_term_psd: false,
            d_term_spectrums: true,
            psd: false,
            psd_db_heatmap: false,
            throttle_freq_heatmap: false,
            d_term_heatmap: false,
            motor_spectrums: true,
            bode: false,
            pid_activity: false,
            rc_command_activity: true,
            motor_erpm: false,
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
            rc_command_activity: false,
            motor_erpm: false,
        }
    }

    fn all() -> Self {
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
            bode: false, // Bode requires specialized logs
            pid_activity: true,
            rc_command_activity: true,
            motor_erpm: true,
        }
    }
}

// Analysis options struct to group related analysis parameters
#[derive(Debug, Clone)]
struct AnalysisOptions {
    pub setpoint_threshold: f64,
    pub show_legend: bool,
    pub debug_mode: bool,
    pub show_butterworth: bool,
    pub estimate_optimal_p: bool,
    /// Trim start, in seconds relative to the log's first row. `None` = log start.
    pub trim_start: Option<f64>,
    /// Trim end, in seconds relative to the log's first row. `None` = log end.
    pub trim_end: Option<f64>,
    /// Raw `--start` argument text, as typed. Used verbatim in the output filename suffix so
    /// two distinct user-supplied values can never collide there, unlike a rounded re-format
    /// of the parsed f64 (`None` when `--start` was omitted; the filename then uses a fixed
    /// formatted default, which is deterministic and so can't collide with itself either).
    pub trim_start_str: Option<String>,
    /// Raw `--end` argument text, as typed. Same rationale as `trim_start_str`.
    pub trim_end_str: Option<String>,
}

use crate::constants::{
    DEFAULT_SETPOINT_THRESHOLD, EXCLUDE_END_S, EXCLUDE_START_S, FRAME_LENGTH_S,
    TRIM_START_DEFAULT_S,
};

// Specific plot function imports
use crate::plot_functions::motor_desync::{detect_motor_desync, fallback_oscillation_overlaps};
use crate::plot_functions::plot_bode::plot_bode_analysis;
use crate::plot_functions::plot_d_term_heatmap::plot_d_term_heatmap;
use crate::plot_functions::plot_d_term_psd::plot_d_term_psd;
use crate::plot_functions::plot_d_term_spectrums::plot_d_term_spectrums;
use crate::plot_functions::plot_gyro_spectrums::plot_gyro_spectrums;
use crate::plot_functions::plot_gyro_vs_unfilt::plot_gyro_vs_unfilt;
use crate::plot_functions::plot_motor_erpm::plot_motor_erpm;
use crate::plot_functions::plot_motor_spectrums::{
    detect_motor_oscillations, plot_motor_spectrums,
};
use crate::plot_functions::plot_pid_activity::plot_pid_activity;
use crate::plot_functions::plot_pidsum_error_setpoint::plot_pidsum_error_setpoint;
use crate::plot_functions::plot_psd::plot_psd;
use crate::plot_functions::plot_psd_db_heatmap::plot_psd_db_heatmap;
use crate::plot_functions::plot_rc_command_activity::plot_rc_command_activity;
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
                "⚠️  Failed to restore original directory to {}: {}",
                self.original_dir.display(),
                e
            );
        }
    }
}

// Data input import
use crate::data_input::log_data::LogRowData;
use crate::data_input::log_parser::{self, parse_log_file};
use crate::data_input::pid_metadata::parse_pid_metadata;

// PID context import
use crate::pid_context::PidContext;

// Data analysis imports
use crate::data_analysis::calc_step_response;
use crate::data_analysis::calc_step_response::{compute_setpoint_authority, SetpointAuthority};
use crate::data_analysis::filter_response;

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
                        eprintln!("⚠️  Skipping header file: {}", input_path_str);
                    } else {
                        csv_files.push(input_path_str.clone());
                    }
                } else {
                    eprintln!("⚠️  Skipping non-CSV file: {}", input_path_str);
                }
            } else {
                eprintln!("⚠️  Skipping file without extension: {}", input_path_str);
            }
        } else if input_path.is_dir() {
            // It's a directory, find CSV files (recursive only if flag is set)
            match find_csv_files_in_dir(input_path, recursive, debug_mode) {
                Ok((mut dir_csv_files, skipped_count)) => {
                    csv_files.append(&mut dir_csv_files);
                    total_skipped += skipped_count;
                }
                Err(err) => eprintln!("⚠️  Error processing directory {}: {}", input_path_str, err),
            }
        } else {
            // Path doesn't exist or isn't accessible
            eprintln!("⚠️  Path not found or not accessible: {}", input_path_str);
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
                "⚠️  Cannot canonicalize directory path: {}",
                dir_path.display()
            );
            return Ok((csv_files, skipped_count));
        }
    };

    // Check if we've already visited this directory (symlink loop detection)
    if visited.contains(&canonical_path) {
        eprintln!(
            "⚠️  Skipping directory due to symlink loop: {}",
            dir_path.display()
        );
        return Ok((csv_files, skipped_count));
    }
    visited.insert(canonical_path);

    let entries = match fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!(
                "⚠️  Cannot read directory '{}': {}",
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
                    "⚠️  Error reading directory entry in '{}': {}",
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
                        "⚠️  Error processing subdirectory '{}': {}",
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
                            eprintln!("⚠️  Skipping header file: {}", path_str);
                        } else {
                            csv_files.push(path_str.to_string());
                        }
                    } else {
                        eprintln!("⚠️  Skipping file with non-UTF-8 path: {}", path.display());
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
    eprintln!("\nUsage: {program_name} <input1> [<input2> ...] [OPTIONS]");
    eprintln!();
    eprintln!("--- INPUT/OUTPUT OPTIONS ---");
    eprintln!();
    eprintln!(
        "  <inputX>: CSV files, directories, or wildcards (*.csv). Header files auto-excluded."
    );
    eprintln!("  -O, --output-dir <directory>: Output directory (default: source folder).");
    eprintln!("  -R, --recursive: Recursively find CSV files in subdirectories.");
    eprintln!();
    eprintln!("--- PLOT TYPE SELECTION ---");
    eprintln!();
    eprintln!("  --core           [default] Step Response, Gyro Spectrums, D-term Spectrums,");
    eprintln!("                   Setpoint vs Gyro, Gyro vs Unfiltered, Motor Spectrums,");
    eprintln!("                   RC Command Activity.");
    eprintln!("  --extended       All plots except Bode — adds PIDsum/Error, PID Activity,");
    eprintln!("                   Setpoint Derivative, Gyro PSD, D-term PSD, heatmaps, and");
    eprintln!("                   Motor vs eRPM (requires bidirectional DShot telemetry).");
    eprintln!("  --step           Step response only.");
    eprintln!("  --bode           Bode only (requires chirp/sweep system-id test flight).");
    eprintln!("  --desync         Motor vs eRPM plot only (needs eRPM telemetry). Desync");
    eprintln!("                   detection itself (Fallback tier) still runs without it.");
    eprintln!();
    eprintln!("--- ANALYSIS OPTIONS ---");
    eprintln!();
    eprintln!("  --butterworth    Show Butterworth PT1 cutoffs on gyro/D-term spectrum plots.");
    eprintln!(
        "  --dps <value>    Deg/s threshold for detailed step response plots (positive number)."
    );
    eprintln!("  --estimate-optimal-p  [EXPERIMENTAL] Optimal P estimation from throttle-punch");
    eprintln!("                        dynamics. Requires .headers.csv; skips if absent.");
    eprintln!("                        Its Td target always profiles the full file, ignoring");
    eprintln!("                        --start/--end; only its Td measurement is trimmed.");
    eprintln!();
    eprintln!("--- TIME WINDOW ---");
    eprintln!();
    eprintln!("  --start <seconds>  Trim analysis to this offset onward, relative to the");
    eprintln!("                     log's first row. Omit to start at the log start.");
    eprintln!("  --end <seconds>    Trim analysis up to this offset, relative to the log's");
    eprintln!("                     first row. Omit to end at the log end.");
    eprintln!("                     Independent — use either or both. Applies before every");
    eprintln!("                     analysis and plot (step response may skip if the");
    eprintln!("                     trimmed window is too short).");
    eprintln!();
    eprintln!("--- GENERAL ---");
    eprintln!();
    eprintln!("  --debug          Show detailed metadata during processing.");
    eprintln!("  -h, --help       Show this help message and exit.");
    eprintln!("  -V, --version    Show version information.");
    std::process::exit(1);
}

/// Parses a `--start`/`--end` style flag: consumes its numeric value, rejects a negative or a
/// repeated flag, and advances `i` past the value. Exits via `print_usage_and_exit` on error.
/// Also captures the argument's raw text into `raw_slot` for use in the output filename suffix
/// (see `AnalysisOptions::trim_start_str` for why).
fn parse_trim_flag(
    flag: &str,
    slot: &mut Option<f64>,
    raw_slot: &mut Option<String>,
    args: &[String],
    i: &mut usize,
    program_name: &str,
) {
    if slot.is_some() {
        eprintln!("Error: {flag} argument specified more than once.");
        print_usage_and_exit(program_name);
    }
    if *i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a numeric value (seconds).");
        print_usage_and_exit(program_name);
    }
    match args[*i + 1].parse::<f64>() {
        Ok(val) if val >= TRIM_START_DEFAULT_S => {
            *slot = Some(val);
            *raw_slot = Some(args[*i + 1].clone());
            *i += 1;
        }
        Ok(val) => {
            eprintln!("Error: {flag} must be >= {TRIM_START_DEFAULT_S}: {val}");
            print_usage_and_exit(program_name);
        }
        Err(_) => {
            eprintln!("Error: Invalid numeric value for {flag}: {}", args[*i + 1]);
            print_usage_and_exit(program_name);
        }
    }
}

/// Extract an aircraft grouping key from a file path alone.
///
/// Fallback key used when the log header has no usable `Craft name` (see
/// `aircraft_group_key`) — e.g. no `.headers.csv` sidecar and no embedded header block.
/// Strips the date-time portion (`_YYYYMMDD_HHMMSS`) so files from the same aircraft
/// across multiple sessions share one key.  When the craft name follows the timestamp
/// (e.g. the standard Betaflight naming scheme), it is appended to the prefix so that
/// different craft logged under the same generic prefix remain distinct. This filename
/// heuristic cannot help when the name was manually changed or is a generic MSC-mode
/// export with nothing after the timestamp — such files still collapse into one group.
///
/// Examples:
///   `EMUF_BLACKBOX_LOG_FOXEERF722V4_426_20240406_132335_notes.19.csv`
///     → `EMUF_BLACKBOX_LOG_FOXEERF722V4_426`  (craft name precedes date)
///   `BTFL_BLACKBOX_LOG_20250517_130413_STELLARH7DEV.02.csv`
///     → `BTFL_BLACKBOX_LOG_STELLARH7DEV`  (generic prefix; craft name appended from suffix)
///
/// Files without a date pattern are treated as unique aircraft (full stem as key).
fn extract_aircraft_key(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // Scan for the pattern _YYYYMMDD_HHMMSS (8-digit date, underscore, 6-digit time).
    let bytes = stem.as_bytes();
    let min_len = 16; // '_' + 8 digits + '_' + 6 digits
    if bytes.len() >= min_len {
        for i in 0..=(bytes.len() - min_len) {
            if bytes[i] == b'_'
                && bytes[i + 1..i + 9].iter().all(|&b| b.is_ascii_digit())
                && bytes[i + 9] == b'_'
                && bytes[i + 10..i + 16].iter().all(|&b| b.is_ascii_digit())
            {
                // i == 0 would give an empty key; fall through to full stem.
                if i > 0 {
                    let prefix = &stem[..i];
                    let after_datetime = &stem[(i + 16)..];
                    // When the prefix is a generic firmware placeholder (ends with
                    // "BLACKBOX_LOG"), the craft name sits in the suffix; include it
                    // so files from different craft don't collapse into one group.
                    if !after_datetime.is_empty() && prefix.ends_with("BLACKBOX_LOG") {
                        let craft = after_datetime.strip_prefix('_').unwrap_or(after_datetime);
                        // Strip trailing log-number suffix (e.g. ".02", ".19").
                        let craft = if let Some(dot) = craft.rfind('.') {
                            let tail = &craft[dot + 1..];
                            if tail.chars().all(|c| c.is_ascii_digit()) {
                                &craft[..dot]
                            } else {
                                craft
                            }
                        } else {
                            craft
                        };
                        if !craft.is_empty() {
                            return format!("{}_{}", prefix, craft);
                        }
                    }
                    return prefix.to_string();
                }
            }
        }
    }

    // No date pattern found — use full stem.
    stem.to_string()
}

/// Resolve the aircraft grouping key for one file.
///
/// The log header's `Craft name` is authoritative when present and non-empty — it survives
/// a pilot renaming the file and is the only signal available for generic/MSC-mode exports,
/// where the filename carries no craft-distinguishing suffix at all (see
/// `extract_aircraft_key`). Falls back to the filename-derived key only when no header
/// metadata is available (no `.headers.csv` sidecar and none embedded in the CSV) or the
/// header has no `Craft name` entry.
///
/// A header-sourced key is namespaced with a `craft:` prefix so it can never collide with a
/// filename-derived key by coincidence — the two are drawn from different identity sources
/// and must never be treated as equal. Mixing craft names across two genuinely different
/// aircraft that share the same pilot-set name is an accepted limitation: it requires pilots
/// to keep craft names unique, same as any other tool keyed on it.
///
/// `use_header_craft_name` skips the header pre-scan entirely when false — grouping only
/// affects the optimal-P profiler's aircraft grouping, so a run without `--estimate-optimal-p`
/// has no reason to pay for reading each file's header a second time.
fn aircraft_group_key(path: &Path, debug_mode: bool, use_header_craft_name: bool) -> String {
    if use_header_craft_name {
        if let Some(craft_name) = log_parser::peek_craft_name(path, debug_mode) {
            return format!("craft:{craft_name}");
        }
    }
    extract_aircraft_key(path)
}

/// Group a list of file paths by aircraft key.
///
/// Returns a `BTreeMap` (sorted by key) mapping each aircraft key to its files.
fn group_files_by_aircraft(
    input_files: &[String],
    debug_mode: bool,
    use_header_craft_name: bool,
) -> BTreeMap<String, Vec<String>> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in input_files {
        let key = aircraft_group_key(Path::new(file), debug_mode, use_header_craft_name);
        groups.entry(key).or_default().push(file.clone());
    }
    groups
}

/// Parse all files in a group and collect torque-inertia ratio estimates.
///
/// This is the Phase 1 profiling pass. Each file is parsed once and the result
/// is cached and returned alongside the profile. Phase 2 (`process_file`) reuses
/// the cache entry instead of re-parsing the same file.
fn profile_aircraft_group(
    files: &[String],
    debug_mode: bool,
) -> (AircraftProfile, Vec<LogParseResult>) {
    let mut all_axis_ratios: [Vec<f64>; crate::axis_names::AXIS_COUNT] =
        std::array::from_fn(|_| Vec::new());
    let mut files_profiled: usize = 0;
    let mut parsed_cache: Vec<LogParseResult> = Vec::with_capacity(files.len());

    for file_str in files {
        let path = Path::new(file_str);
        let parse_result = parse_log_file(path, debug_mode);
        match &parse_result {
            Ok((log_data, Some(sr), ..)) => {
                let ratios = extract_punch_ratios(log_data, *sr);
                let total_events: usize = ratios.iter().map(|v| v.len()).sum();
                for (axis, axis_ratio_vec) in all_axis_ratios.iter_mut().enumerate() {
                    axis_ratio_vec.extend_from_slice(&ratios[axis]);
                }
                if debug_mode {
                    println!(
                        "  [torque-profile] {}: {} throttle-punch events across all axes",
                        path.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(file_str),
                        total_events
                    );
                }
                if total_events > 0 {
                    files_profiled += 1;
                }
            }
            Ok((_, None, ..)) => {
                if debug_mode {
                    eprintln!("  [torque-profile] Skipping {} (no sample rate)", file_str);
                }
            }
            Err(e) => {
                if debug_mode {
                    eprintln!("  [torque-profile] Failed to parse {}: {}", file_str, e);
                }
            }
        }
        parsed_cache.push(parse_result);
    }

    if debug_mode {
        println!(
            "  [torque-profile] Profiled {} file(s). Events: Roll={}, Pitch={}, Yaw={}",
            files_profiled,
            all_axis_ratios[0].len(),
            all_axis_ratios[1].len(),
            all_axis_ratios[2].len()
        );
    }

    let mut profile = AircraftProfile::from_axis_ratios(all_axis_ratios);
    profile.file_count = files_profiled;
    (profile, parsed_cache)
}

fn process_file(
    input_file_str: &str,
    use_dir_prefix: bool,
    output_dir: Option<&Path>,
    plot_config: PlotConfig,
    analysis_opts: AnalysisOptions,
    aircraft_profile: &AircraftProfile,
    pre_parsed: Option<LogParseResult>,
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
    let mut root_name_string: String = if use_dir_prefix {
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
        format!("{dir_prefix_to_add}{file_stem_cow}")
    } else {
        file_stem_cow.into_owned()
    };

    // --- Data Reading and Header Status ---
    let (
        mut all_log_data,
        sample_rate,
        f_term_header_found,
        setpoint_header_found,
        gyro_header_found,
        _gyro_unfilt_header_found,
        _debug_header_found,
        using_debug_fallback,
        header_metadata,
    ) = match pre_parsed
        // A cached parse failure may be transient (file locked/still being flushed at
        // profiling time); retry with a fresh parse rather than reusing a stale error.
        .filter(|r| r.is_ok())
        .unwrap_or_else(|| parse_log_file(input_path, analysis_opts.debug_mode))
    {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error: Parsing log file {input_file_str}: {e}");
            return Ok(()); // Continue to next file
        }
    };

    if all_log_data.is_empty() {
        println!("No valid data rows read from {input_file_str}, cannot generate plots.");
        return Ok(());
    }

    // Captured before zeroing below — this note is the only place absolute FC uptime is shown,
    // for cross-referencing against OSD/video overlays that also display FC uptime.
    let log_first = all_log_data.first().and_then(|row| row.time_sec);
    let log_last = all_log_data.last().and_then(|row| row.time_sec);
    if let (Some(f), Some(l)) = (log_first, log_last) {
        if l > f {
            println!(
                "Note: Log spans absolute time {f:.3}s-{l:.3}s (duration {:.3}s) of {input_file_str}",
                l - f
            );
        }
    }

    // --- Apply --start/--end time-window trim ---
    // Runs before every analysis/plot module below, so all of them see only the trimmed rows —
    // except Motor Desync Detection, which needs the full untrimmed log for its baseline
    // statistics (see `desync_full_log_data`/`desync_report_window` below and IT #182).
    let mut trim_window: Option<(f64, f64, usize)> = None;
    // Motor Desync Detection baselines must not shrink with the trim window (IT #182) — captured
    // only when a trim actually runs; `None` for both means "use all_log_data itself, unfiltered",
    // identical to today's untrimmed behavior.
    let mut desync_full_log_data: Option<Vec<LogRowData>> = None;
    let mut desync_report_window: Option<(f64, f64)> = None;
    if analysis_opts.trim_start.is_some() || analysis_opts.trim_end.is_some() {
        let (log_first, log_last) = match (log_first, log_last) {
            (Some(f), Some(l)) if l > f => (f, l),
            _ => {
                eprintln!(
                    "Error: Cannot apply --start/--end trim to {input_file_str}: time data unavailable."
                );
                return Ok(());
            }
        };
        let duration = log_last - log_first;
        let rel_start = analysis_opts.trim_start.unwrap_or(TRIM_START_DEFAULT_S);
        let rel_end = analysis_opts.trim_end.unwrap_or(duration);
        if rel_start >= duration {
            eprintln!(
                "Error: --start {rel_start:.3}s is at or beyond log duration ({duration:.3}s) for {input_file_str}"
            );
            return Ok(());
        }
        if rel_end > duration {
            eprintln!(
                "Error: --end {rel_end:.3}s exceeds log duration ({duration:.3}s) for {input_file_str}"
            );
            return Ok(());
        }
        if rel_start >= rel_end {
            eprintln!(
                "Error: --start ({rel_start:.3}s) must be less than --end ({rel_end:.3}s) for {input_file_str}"
            );
            return Ok(());
        }
        let window_start = log_first + rel_start;
        let window_end = log_first + rel_end;
        desync_full_log_data = Some(all_log_data.clone());
        desync_report_window = Some((window_start, window_end));
        all_log_data.retain(|row| {
            row.time_sec
                .is_some_and(|t| t >= window_start && t <= window_end)
        });
        if all_log_data.is_empty() {
            eprintln!(
                "Error: --start/--end window leaves no data rows for {input_file_str} (log spans 0.000s-{duration:.3}s)"
            );
            return Ok(());
        }
        println!(
            "Note: Trimmed to {rel_start:.3}s-{rel_end:.3}s ({} rows, log duration {duration:.3}s) of {input_file_str}",
            all_log_data.len()
        );
        if rel_end - rel_start < EXCLUDE_START_S + EXCLUDE_END_S {
            println!(
                "Note: Trimmed window ({:.3}s) is shorter than the {:.1}s step-response margin — step response will likely be skipped.",
                rel_end - rel_start,
                EXCLUDE_START_S + EXCLUDE_END_S
            );
        }
        trim_window = Some((rel_start, rel_end, all_log_data.len()));
        // Use the raw --start/--end text the user typed, not a rounded re-format of the parsed
        // f64: two distinct inputs then can never collide on the same filename (a rounded
        // format could, e.g. --start 1.0001 and --start 1.0004 both rounding to "1.000"). The
        // omitted side (no raw text) falls back to a fixed formatted default, which is the same
        // for every run of the same file and so can't newly collide with itself either.
        let start_label = analysis_opts
            .trim_start_str
            .clone()
            .unwrap_or_else(|| format!("{rel_start:.3}"));
        let end_label = analysis_opts
            .trim_end_str
            .clone()
            .unwrap_or_else(|| format!("{rel_end:.3}"));
        root_name_string = format!("{root_name_string}_trim{start_label}s-{end_label}s");
    }

    // Zero the timeline to the log's own start (regardless of --start/--end trimming) so every
    // plot and console/report timestamp reads flight-relative time, not raw FC uptime, while a
    // trimmed window still shows its true offset into the flight instead of resetting to 0.
    // Applied to all_log_data and, when present, to the untrimmed clone/window Motor Desync
    // Detection uses for its full-log baseline — both must share the same zero point as the
    // plots, or its reported event timestamps would disagree with everything else.
    if let Some(origin) = log_first {
        for row in &mut all_log_data {
            if let Some(t) = row.time_sec.as_mut() {
                *t -= origin;
            }
        }
        if let Some(full_data) = desync_full_log_data.as_mut() {
            for row in full_data.iter_mut() {
                if let Some(t) = row.time_sec.as_mut() {
                    *t -= origin;
                }
            }
        }
        if let Some((window_start, window_end)) = desync_report_window.as_mut() {
            *window_start -= origin;
            *window_end -= origin;
        }
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

    let pd_ratios_for_report: [Option<f64>; AXIS_COUNT] = [
        pid_metadata.roll.calculate_pd_ratio(),
        pid_metadata.pitch.calculate_pd_ratio(),
        pid_metadata.yaw.calculate_pd_ratio(),
    ];

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
INFO: Skipping Step Response data collection for {input_file_str}: Setpoint or Gyro headers missing."
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
INFO: Skipping Step Response input data filtering for {input_file_str}: {reason}."
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

    // Setpoint authority per axis (captured for report)
    let mut setpoint_authority_names: [Option<&'static str>; AXIS_COUNT] =
        std::array::from_fn(|_| None);
    let mut setpoint_authority_means: [Option<f32>; AXIS_COUNT] = std::array::from_fn(|_| None);

    // Step response warnings per axis (captured for report)
    let mut step_warnings: [Vec<String>; AXIS_COUNT] = std::array::from_fn(|_| Vec::new());

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
        println!("Note: These are STARTING POINTS based on step response analysis.");
        println!("      These recommendations focus on D-term tuning (P:D ratio).");
        if analysis_opts.estimate_optimal_p {
            println!(
                "      See 'Optimal P Estimation' below for P gain magnitude recommendations."
            );
        }
        println!("      Always test in a safe environment. Conservative = safer first step.");
        println!("      Moderate = for experienced pilots (test carefully to avoid hot motors).");
        for axis_index in 0..crate::axis_names::ROLL_PITCH_AXIS_COUNT {
            // Only Roll (0) and Pitch (1)
            let axis_name = crate::axis_names::AXIS_NAMES[axis_index];

            if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) =
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
                                // Peak zones (see constants.rs for threshold values):
                                //   < PEAK_UNDERSHOOT_MAX        = undershoot:   Recommendation (conservative) proportional D decrease
                                //   PEAK_UNDERSHOOT_MAX..PEAK_OPTIMAL_MIN = near optimal: Recommendation (none)
                                //   PEAK_OPTIMAL_MIN..PEAK_ACCEPTABLE_MIN = optimal:      Recommendation (none)
                                //   PEAK_ACCEPTABLE_MIN..PEAK_ACCEPTABLE_MAX = acceptable: Recommendation (conservative)
                                //   PEAK_ACCEPTABLE_MAX..PEAK_SIGNIFICANT_MIN = overshoot:  Recommendation (conservative)
                                //   > PEAK_SIGNIFICANT_MIN       = significant:  conservative + moderate + aggressive
                                let (assessment, recommended_ratio) = if peak_value
                                    > crate::constants::PEAK_SIGNIFICANT_MIN
                                {
                                    (
                                        "Significant overshoot",
                                        current_pd_ratio
                                            * crate::constants::PD_RATIO_CONSERVATIVE_MULTIPLIER,
                                    )
                                } else if peak_value > crate::constants::PEAK_ACCEPTABLE_MAX {
                                    (
                                        "Overshoot",
                                        current_pd_ratio
                                            * crate::constants::PEAK_OVERSHOOT_MULTIPLIER,
                                    )
                                } else if peak_value >= crate::constants::PEAK_ACCEPTABLE_MIN {
                                    (
                                        "Acceptable",
                                        current_pd_ratio
                                            * crate::constants::PEAK_ACCEPTABLE_MULTIPLIER,
                                    )
                                } else if peak_value >= crate::constants::PEAK_OPTIMAL_MIN {
                                    ("Optimal", current_pd_ratio)
                                } else if peak_value >= crate::constants::PEAK_UNDERSHOOT_MAX {
                                    ("Near optimal", current_pd_ratio)
                                } else {
                                    // Proportional D decrease: scale P:D toward the optimal sweet-spot centre
                                    let multiplier = crate::constants::PEAK_OPTIMAL_TARGET
                                        / peak_value.max(crate::constants::PEAK_VALUE_MIN_CLAMP);
                                    ("Undershoot", current_pd_ratio * multiplier)
                                };

                                // Store peak value, current P:D ratio, and assessment for plot legends
                                peak_values[axis_index] = Some(peak_value);
                                current_pd_ratios[axis_index] = Some(current_pd_ratio);
                                assessments[axis_index] = Some(assessment);

                                // Always store recommendations for zones where the delta is small by
                                // design (Acceptable ≈2%, Undershoot near-boundary ≈5% at low P:D)
                                // to avoid the 5% gate suppressing valid recommendations.
                                if assessment == "Acceptable"
                                    || assessment == "Undershoot"
                                    || (recommended_ratio - current_pd_ratio).abs()
                                        > crate::constants::PD_RATIO_MIN_CHANGE_THRESHOLD
                                {
                                    // store conservative recommendation for later use in plots
                                    recommended_pd_conservative[axis_index] =
                                        Some(recommended_ratio);

                                    // Calculate moderate recommendation for any overshoot (>PEAK_ACCEPTABLE_MAX)
                                    // Base directly on current_pd_ratio so the result is always
                                    // current * PD_RATIO_MODERATE_MULTIPLIER regardless of which
                                    // conservative-tier multiplier was applied to recommended_ratio.
                                    let moderate_ratio =
                                        if peak_value > crate::constants::PEAK_ACCEPTABLE_MAX {
                                            let ratio = current_pd_ratio
                                                * crate::constants::PD_RATIO_MODERATE_MULTIPLIER;
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

                                println!("{axis_name}: Actual Peak={peak_value:.3} → {assessment}");

                                // Always show current P:D ratio with quality assessment
                                let axis_pid = if axis_index == 0 {
                                    &pid_metadata.roll
                                } else {
                                    &pid_metadata.pitch
                                };

                                if axis_pid.p.is_some() {
                                    println!("  Current P:D={current_pd_ratio:.2}");
                                    // Needed in both branches below
                                    let dmax_enabled = pid_metadata.is_dmax_enabled();

                                    // Setpoint Authority from mean of per-window max setpoints
                                    let (authority, authority_mean) = compute_setpoint_authority(
                                        valid_window_max_setpoints.as_slice().unwrap_or(&[]),
                                    )
                                    .unwrap_or((SetpointAuthority::Low, 0.0));
                                    setpoint_authority_names[axis_index] = Some(authority.name());
                                    setpoint_authority_means[axis_index] = Some(authority_mean);
                                    println!(
                                        "  Setpoint Authority: {} (mean={:.0}dps \u{22a2}\u{2265}{}dps)",
                                        authority.name(),
                                        authority_mean,
                                        crate::constants::LOW_AUTHORITY_SETPOINT_THRESHOLD_DEG_S as u32
                                    );

                                    // Show recommendations if they were computed (threshold exceeded)
                                    if recommended_pd_conservative[axis_index].is_some() {
                                        // Check for extreme overshoot (may indicate deeper issues)
                                        if peak_value > crate::constants::SEVERE_OVERSHOOT_THRESHOLD
                                        {
                                            println!("  ⚠️  Severe overshoot (Peak={peak_value:.2}) may indicate:");
                                            println!(
                                                "      - P value too high, or mechanical issues"
                                            );
                                            println!(
                                                "      - Check for bent props, loose hardware, or damaged motors"
                                            );
                                            step_warnings[axis_index].push(format!(
                                                "Severe overshoot (Peak={peak_value:.2}): P may be too high, or check for bent props/loose hardware/damaged motors"
                                            ));
                                        }

                                        // Check for unreasonable P:D ratios
                                        let rec_ratio =
                                            recommended_pd_conservative[axis_index].unwrap();
                                        if rec_ratio < crate::constants::MIN_REASONABLE_PD_RATIO {
                                            println!("  ⚠️  Recommended P:D ratio ({rec_ratio:.2}) is very low");
                                            println!(
                                                "      Consider increasing P instead of only adding D"
                                            );
                                            step_warnings[axis_index].push(format!(
                                                "Recommended P:D ratio ({rec_ratio:.2}) is very low — consider increasing P instead of only adding D"
                                            ));
                                        } else if rec_ratio
                                            > crate::constants::MAX_REASONABLE_PD_RATIO
                                        {
                                            println!("  ⚠️  Recommended P:D ratio ({rec_ratio:.2}) is very high");
                                            println!("      Consider decreasing P or checking for overdamped response");
                                            step_warnings[axis_index].push(format!(
                                                "Recommended P:D ratio ({rec_ratio:.2}) is very high — consider decreasing P or checking for overdamped response"
                                            ));
                                        }

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
                                            println!("  Recommendation (conservative): P:D={:.2} (D-Min≈{}, D-Max≈{})",
                                                recommended_pd_conservative[axis_index].unwrap(),
                                                d_min_str, d_max_str);
                                        } else if let Some(recommended_d) =
                                            recommended_d_conservative[axis_index]
                                        {
                                            // D-Min/D-Max disabled: show only base D
                                            println!(
                                                "  Recommendation (conservative): P:D={:.2} (D≈{})",
                                                recommended_pd_conservative[axis_index].unwrap(),
                                                recommended_d
                                            );
                                        }

                                        // Show secondary (moderate) recommendation
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
                                            println!("  Recommendation (moderate): P:D={:.2} (D-Min≈{}, D-Max≈{})",
                                                recommended_pd_aggressive[axis_index].unwrap(),
                                                d_min_str, d_max_str);
                                        } else if let Some(recommended_d_mod) =
                                            recommended_d_aggressive[axis_index]
                                        {
                                            // D-Min/D-Max disabled: show only base D
                                            println!(
                                                "  Recommendation (moderate): P:D={:.2} (D≈{})",
                                                recommended_pd_aggressive[axis_index].unwrap(),
                                                recommended_d_mod
                                            );
                                        }

                                        // Show tertiary (aggressive) recommendation for significant overshoot only
                                        if assessment == "Significant overshoot" {
                                            let aggressive_pd = current_pd_ratio
                                                * crate::constants::PD_RATIO_AGGRESSIVE_MULTIPLIER;
                                            let (rec_d_agg, rec_d_min_agg, rec_d_max_agg) =
                                                if axis_index == 0 {
                                                    pid_metadata.roll.calculate_goal_d_with_range(
                                                        aggressive_pd,
                                                        dmax_enabled,
                                                    )
                                                } else {
                                                    pid_metadata.pitch.calculate_goal_d_with_range(
                                                        aggressive_pd,
                                                        dmax_enabled,
                                                    )
                                                };
                                            if dmax_enabled
                                                && (rec_d_min_agg.is_some()
                                                    || rec_d_max_agg.is_some())
                                            {
                                                let d_min_str = rec_d_min_agg
                                                    .map_or("N/A".to_string(), |v| v.to_string());
                                                let d_max_str = rec_d_max_agg
                                                    .map_or("N/A".to_string(), |v| v.to_string());
                                                println!("  Recommendation (aggressive): P:D={:.2} (D-Min≈{}, D-Max≈{})",
                                                    aggressive_pd, d_min_str, d_max_str);
                                            } else if let Some(rec_d3) = rec_d_agg {
                                                println!("  Recommendation (aggressive): P:D={:.2} (D≈{})",
                                                    aggressive_pd, rec_d3);
                                            }
                                        }
                                    } else if assessment == "Near optimal" {
                                        // Near optimal (1.00–1.02): D−1 hint only — no "none" line,
                                        // since "none" and a concrete suggestion are contradictory.
                                        if dmax_enabled {
                                            let d_min_str = axis_pid
                                                .d_min
                                                .map(|v| {
                                                    v.saturating_sub(
                                                        crate::constants::D_STEP_OPTIONAL,
                                                    )
                                                })
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            let d_max_str = axis_pid
                                                .d_max
                                                .or(axis_pid.d)
                                                .map(|v| {
                                                    v.saturating_sub(
                                                        crate::constants::D_STEP_OPTIONAL,
                                                    )
                                                })
                                                .map_or("N/A".to_string(), |v| v.to_string());
                                            println!(
                                                "  Recommendation (conservative): D-Min≈{}, D-Max≈{} [optional D−1]",
                                                d_min_str, d_max_str
                                            );
                                        } else if let Some(current_d) = axis_pid.d {
                                            println!(
                                                "  Recommendation (conservative): D≈{} [optional D−1]",
                                                current_d.saturating_sub(
                                                    crate::constants::D_STEP_OPTIONAL
                                                )
                                            );
                                        }
                                    } else {
                                        // Optimal zone (1.02–1.08): no adjustment needed
                                        println!(
                                            "  Recommendation (none): No obvious tuning adjustments needed"
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

    // Collect step response analysis into typed report structs
    let mut step_reports: Vec<report::StepAxisReport> = Vec::new();
    {
        let dmax_enabled = pid_metadata.is_dmax_enabled();
        for axis_index in 0..crate::axis_names::ROLL_PITCH_AXIS_COUNT {
            if let (Some(peak_value), Some(current_pd_ratio), Some(assessment)) = (
                peak_values[axis_index],
                current_pd_ratios[axis_index],
                assessments[axis_index],
            ) {
                let conservative =
                    recommended_pd_conservative[axis_index].map(|pd| report::DTermRec {
                        pd_ratio: pd,
                        d: recommended_d_conservative[axis_index],
                        d_min: recommended_d_min_conservative[axis_index],
                        d_max: recommended_d_max_conservative[axis_index],
                    });
                let moderate = recommended_pd_aggressive[axis_index].map(|pd| report::DTermRec {
                    pd_ratio: pd,
                    d: recommended_d_aggressive[axis_index],
                    d_min: recommended_d_min_aggressive[axis_index],
                    d_max: recommended_d_max_aggressive[axis_index],
                });
                let aggressive = if assessment == "Significant overshoot" {
                    let aggressive_pd =
                        current_pd_ratio * crate::constants::PD_RATIO_AGGRESSIVE_MULTIPLIER;
                    let (rec_d, rec_d_min, rec_d_max) = if axis_index == 0 {
                        pid_metadata
                            .roll
                            .calculate_goal_d_with_range(aggressive_pd, dmax_enabled)
                    } else {
                        pid_metadata
                            .pitch
                            .calculate_goal_d_with_range(aggressive_pd, dmax_enabled)
                    };
                    Some(report::DTermRec {
                        pd_ratio: aggressive_pd,
                        d: rec_d,
                        d_min: rec_d_min,
                        d_max: rec_d_max,
                    })
                } else {
                    None
                };
                step_reports.push(report::StepAxisReport {
                    axis_name: crate::axis_names::AXIS_NAMES[axis_index],
                    peak_value,
                    assessment,
                    current_pd_ratio,
                    conservative,
                    moderate,
                    aggressive,
                    setpoint_authority_name: setpoint_authority_names[axis_index],
                    setpoint_authority_mean: setpoint_authority_means[axis_index],
                    warnings: std::mem::take(&mut step_warnings[axis_index]),
                });
            }
        }
    }

    // Optimal P Estimation Analysis (if enabled)
    // Store results for both console output and PNG overlay
    let mut optimal_p_analyses: [Option<
        crate::data_analysis::optimal_p_estimation::OptimalPAnalysis,
    >; crate::axis_names::AXIS_COUNT] = std::array::from_fn(|_| None);
    let mut optimal_p_skip_reasons: [Option<String>; crate::axis_names::AXIS_COUNT] =
        std::array::from_fn(|_| None);

    if analysis_opts.estimate_optimal_p {
        if let Some(sr) = sample_rate {
            let group_or_file = if aircraft_profile.file_count > 1 {
                "group"
            } else {
                "file"
            };
            println!("\n--- Optimal P (Experimental, log-derived) ---");
            println!(
                "Td target: physics-derived from throttle-punch events in log {group_or_file}."
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
                            optimal_p_skip_reasons[axis_index] =
                                Some("No valid Td measurements".to_string());
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

                            // Compute physics-derived Td target for this axis
                            // using the group's torque-inertia profile and current P gain.
                            let physics_td = aircraft_profile.axes[axis_index].td_target_ms(p_gain);

                            if physics_td.is_none() {
                                let events = aircraft_profile.axes[axis_index].event_count;
                                if events < crate::constants::TORQUE_PROFILER_MIN_EVENTS {
                                    let msg = format!(
                                        "SKIPPED: insufficient throttle dynamics ({} events, need >={})",
                                        events, crate::constants::TORQUE_PROFILER_MIN_EVENTS
                                    );
                                    println!("  {}: {}.", axis_name, msg);
                                    println!("    Provide logs with more throttle variation or fly deliberate punch sequences.");
                                    optimal_p_skip_reasons[axis_index] = Some(msg);
                                } else {
                                    let msg =
                                        "SKIPPED: could not compute Td target from profiling data"
                                            .to_string();
                                    println!("  {}: {}.", axis_name, msg);
                                    optimal_p_skip_reasons[axis_index] = Some(msg);
                                }
                                continue;
                            }

                            // Perform optimal P analysis
                            match crate::data_analysis::optimal_p_estimation::OptimalPAnalysis::analyze(
                            &td_samples_ms,
                            p_gain,
                            current_d,
                            hf_energy_ratio,
                            recommended_pd_conservative[axis_index],
                            physics_td,
                        ) {
                                Ok(mut analysis) => {
                                    analysis.source_events =
                                        aircraft_profile.axes[axis_index].event_count;
                                    analysis.source_files = aircraft_profile.file_count;
                                    // Print console output
                                    println!("{}", analysis.format_console_output(axis_name));
                                    // Store for PNG overlay (move instead of clone)
                                    optimal_p_analyses[axis_index] = Some(analysis);
                                }
                                Err(e) => {
                                    // Log the error for user visibility
                                    eprintln!("⚠️  {}", e);
                                    optimal_p_skip_reasons[axis_index] = Some(e.to_string());
                                }
                            }
                        } else {
                            let msg = "SKIPPED: P gain not available".to_string();
                            println!("  {axis_name}: {msg}");
                            optimal_p_skip_reasons[axis_index] = Some(msg);
                        }
                    }
                }
            }
            println!();
        }
    }

    let optimal_p_for_report = optimal_p_analyses.clone();

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

    // A colliding root name across two input logs in the same --output-dir must not let this
    // log's report link a PNG a previous log's run left on disk (see push_if_written below).
    plot_framework::reset_written_this_run();

    // Create PID context for centralized PID metadata and related parameters
    let pid_context = PidContext::new(sample_rate, pid_metadata, root_name_string.clone());

    if plot_config.step_response {
        // Group related parameters into structs for cleaner API
        use crate::plot_functions::plot_step_response::{
            ConservativeRecommendations, CurrentPeakAndRatios, ModerateRecommendations,
            OptimalPConfig, PdRecommendations, PlotDisplayConfig,
        };

        let current = CurrentPeakAndRatios {
            peak_values,
            pd_ratios: current_pd_ratios,
            assessments,
        };

        let conservative = ConservativeRecommendations(PdRecommendations {
            pd_ratios: recommended_pd_conservative,
            d_values: recommended_d_conservative,
            d_min_values: recommended_d_min_conservative,
            d_max_values: recommended_d_max_conservative,
        });

        let moderate = ModerateRecommendations(PdRecommendations {
            pd_ratios: recommended_pd_aggressive,
            d_values: recommended_d_aggressive,
            d_min_values: recommended_d_min_aggressive,
            d_max_values: recommended_d_max_aggressive,
        });

        let display = PlotDisplayConfig {
            has_nonzero_f_term: has_nonzero_f_term_data,
            setpoint_threshold: analysis_opts.setpoint_threshold,
            show_legend: analysis_opts.show_legend,
        };

        let optimal_p = OptimalPConfig {
            analyses: optimal_p_analyses,
            skip_reasons: optimal_p_skip_reasons,
        };

        plot_step_response(
            &step_response_calculation_results,
            &root_name_string,
            sample_rate,
            &pid_context.pid_metadata,
            &current,
            &conservative,
            &moderate,
            &display,
            &optimal_p,
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

    let gyro_analysis = if plot_config.gyro_spectrums {
        Some(plot_gyro_spectrums(
            &all_log_data,
            &root_name_string,
            sample_rate,
            Some(&header_metadata),
            analysis_opts.show_butterworth,
            using_debug_fallback,
            debug_mode_label,
        )?)
    } else {
        None
    };

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

    let dterm_results = if plot_config.d_term_spectrums {
        plot_d_term_spectrums(
            &all_log_data,
            &root_name_string,
            sample_rate,
            Some(&header_metadata),
            analysis_opts.show_butterworth,
            using_debug_fallback,
            debug_mode_label,
        )?
    } else {
        vec![]
    };

    let motor_results = if plot_config.motor_spectrums {
        plot_motor_spectrums(&all_log_data, &root_name_string, sample_rate)?
    } else if plot_config.motor_erpm {
        // Oscillation detection alone (no plot) — keeps the Fallback-tier overlap caveat and
        // the Motor Oscillation report section available under --desync without also writing
        // the Motor Spectrums PNG, which --desync's "only the eRPM plot" contract excludes.
        detect_motor_oscillations(&all_log_data, sample_rate)
    } else {
        vec![]
    };

    let motor_desync_results = if plot_config.motor_spectrums || plot_config.motor_erpm {
        // Baseline always comes from the full untrimmed log; report_window (when a trim is
        // active) only restricts which flagged events get reported (IT #182).
        let desync_source = desync_full_log_data.as_deref().unwrap_or(&all_log_data);
        detect_motor_desync(desync_source, desync_report_window)
    } else {
        vec![]
    };

    for (motor_idx, count) in fallback_oscillation_overlaps(&motor_desync_results, &motor_results) {
        println!(
            "  ⚠️  Motor {motor_idx} Fallback desync event(s) ({count} overlapping) coincides with a Motor Oscillation detection — see Fallback table for exact times"
        );
    }

    if plot_config.motor_erpm {
        plot_motor_erpm(&all_log_data, &root_name_string, &motor_desync_results)?;
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

    let bode_results = if plot_config.bode {
        eprintln!();
        eprintln!("⚠️  Bode plots are designed for controlled test flights with system-identification inputs.");
        eprintln!(
            "    For normal flight log analysis, use spectrum plots (default behavior) instead."
        );
        eprintln!();
        plot_bode_analysis(
            &all_log_data,
            &root_name_string,
            sample_rate,
            analysis_opts.debug_mode,
        )?
    } else {
        vec![]
    };

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

    let rc_command_steps = if plot_config.rc_command_activity {
        plot_rc_command_activity(&all_log_data, &root_name_string)?
    } else {
        vec![]
    };

    // --- Filter configuration (from header metadata, independent of CSV data) ---
    let filter_config = Some(filter_response::parse_filter_config(&header_metadata));
    let dynamic_notch = filter_response::extract_dynamic_notch_range(Some(&header_metadata));
    let rpm_filter = filter_response::extract_rpm_filter_config(Some(&header_metadata));

    // --- Collect generated PNG filenames ---
    // Only link files this run actually wrote: a plot type being enabled doesn't mean its PNG
    // was written — plot_framework.rs skips writing when every axis is data-unavailable — and
    // Path::exists() can't tell such a skip from a stale PNG an earlier run with a colliding
    // root name left on disk. skipped_plots records the human-readable label for each enabled
    // plot type whose file wasn't produced, so the report can note it (see push_if_written below).
    let mut png_links: Vec<String> = Vec::new();
    let mut skipped_plots: Vec<String> = Vec::new();
    let push_if_written =
        |links: &mut Vec<String>, skipped: &mut Vec<String>, label: &str, filename: String| {
            if plot_framework::was_written_this_run(&filename) {
                links.push(filename);
            } else {
                skipped.push(label.to_string());
            }
        };

    if plot_config.step_response {
        // Step response filename includes duration and optional dps suffix, so look it up by
        // prefix among this run's writes rather than a fixed name.
        let prefix = format!("{root_name_string}_Step_Response_stacked_plot_");
        let matches = plot_framework::written_this_run_with_prefix(&prefix);
        if matches.is_empty() {
            skipped_plots.push("Step Response".to_string());
        } else {
            png_links.extend(matches);
        }
    }
    if plot_config.pidsum_error_setpoint {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "PIDsum/PIDerror/Setpoint",
            format!("{root_name_string}_PIDsum_PIDerror_Setpoint_stacked.png"),
        );
    }
    if plot_config.setpoint_vs_gyro {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Setpoint/Gyro",
            format!("{root_name_string}_SetpointVsGyro_stacked.png"),
        );
    }
    if plot_config.setpoint_derivative {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Setpoint Derivative",
            format!("{root_name_string}_SetpointDerivative_stacked.png"),
        );
    }
    if plot_config.gyro_vs_unfilt {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Gyro/UnfiltGyro",
            format!("{root_name_string}_GyroVsUnfilt_stacked.png"),
        );
    }
    if plot_config.gyro_spectrums {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Gyro Spectrums",
            format!("{root_name_string}_Gyro_Spectrums_comparative.png"),
        );
    }
    if plot_config.d_term_psd {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "D-term PSD",
            format!("{root_name_string}_D_Term_PSD_comparative.png"),
        );
    }
    if plot_config.d_term_spectrums {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "D-term Spectrums",
            format!("{root_name_string}_D_Term_Spectrums_comparative.png"),
        );
    }
    if plot_config.motor_spectrums {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Motor Spectrums",
            format!("{root_name_string}_Motor_Spectrums_stacked.png"),
        );
    }
    if plot_config.motor_erpm {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Motor vs eRPM",
            format!("{root_name_string}_Motor_vs_eRPM_stacked.png"),
        );
    }
    if plot_config.psd {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Gyro PSD",
            format!("{root_name_string}_Gyro_PSD_comparative.png"),
        );
    }
    if plot_config.psd_db_heatmap {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Gyro PSD Spectrogram",
            format!("{root_name_string}_Gyro_PSD_Spectrogram_comparative.png"),
        );
    }
    if plot_config.throttle_freq_heatmap {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Throttle-Frequency Heatmap",
            format!("{root_name_string}_Throttle_Freq_Heatmap_comparative.png"),
        );
    }
    if plot_config.d_term_heatmap {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "D-Term Throttle-Frequency Heatmap",
            format!("{root_name_string}_D_Term_Heatmap_comparative.png"),
        );
    }
    if plot_config.bode {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "Bode Analysis",
            format!("{root_name_string}_Bode_Analysis.png"),
        );
    }
    if plot_config.pid_activity {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "P, I, D Activity",
            format!("{root_name_string}_PID_Activity_stacked.png"),
        );
    }
    if plot_config.rc_command_activity {
        push_if_written(
            &mut png_links,
            &mut skipped_plots,
            "RC Command Activity",
            format!("{root_name_string}_RC_Command_Activity_stacked.png"),
        );
    }

    // --- Markdown Report ---
    // Must run after all plots so png_links is complete.
    let report_filename = format!("{root_name_string}_report.md");
    let report_path = std::path::Path::new(&report_filename);
    println!("\n--- Generating Report: {report_filename} ---");
    let flight_report = report::FlightReport {
        root_name: root_name_string.clone(),
        sample_rate,
        trim_window,
        header_metadata,
        pd_ratios: pd_ratios_for_report,
        step_reports,
        optimal_p: optimal_p_for_report,
        gyro_analysis,
        dterm_results,
        bode_results,
        motor_results,
        motor_desync_results,
        rc_command_steps,
        png_links,
        skipped_plots,
        filter_config,
        dynamic_notch,
        rpm_filter,
        debug_fallback: using_debug_fallback,
        debug_mode_name: debug_mode_label,
    };
    report::generate_markdown_report(&flight_report, report_path)
        .map_err(|e| format!("Report generation failed: {e}"))?;
    println!("  [OK] Report written.");
    println!();

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
    let mut core_requested = false;
    let mut extended_requested = false;
    let mut step_requested = false;
    let mut bode_requested = false;
    let mut desync_requested = false;
    let mut recursive = false;
    let mut estimate_optimal_p = false;
    let mut trim_start: Option<f64> = None;
    let mut trim_end: Option<f64> = None;
    let mut trim_start_str: Option<String> = None;
    let mut trim_end_str: Option<String> = None;

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
        } else if arg == "--core" {
            core_requested = true;
        } else if arg == "--extended" {
            extended_requested = true;
        } else if arg == "--step" {
            step_requested = true;
        } else if arg == "--bode" {
            bode_requested = true;
        } else if arg == "--desync" {
            desync_requested = true;
        } else if arg == "--estimate-optimal-p" {
            estimate_optimal_p = true;
        } else if arg == "--start" {
            parse_trim_flag(
                "--start",
                &mut trim_start,
                &mut trim_start_str,
                &args,
                &mut i,
                program_name,
            );
        } else if arg == "--end" {
            parse_trim_flag(
                "--end",
                &mut trim_end,
                &mut trim_end_str,
                &args,
                &mut i,
                program_name,
            );
        } else if arg.starts_with("--") {
            eprintln!("Error: Unknown option '{arg}'");
            print_usage_and_exit(program_name);
        } else {
            input_paths.push(arg.clone());
        }
        i += 1;
    }

    if core_requested && extended_requested {
        eprintln!("Error: --core and --extended are mutually exclusive.");
        print_usage_and_exit(program_name);
    }

    if let (Some(start), Some(end)) = (trim_start, trim_end) {
        if start >= end {
            eprintln!("Error: --start ({start:.3}s) must be less than --end ({end:.3}s).");
            print_usage_and_exit(program_name);
        }
    }

    // Derive plot configuration from flags
    let plot_config = if extended_requested {
        let mut cfg = PlotConfig::all();
        if bode_requested {
            cfg.bode = true;
        }
        cfg
    } else if step_requested || bode_requested || desync_requested {
        let mut cfg = PlotConfig::none();
        if step_requested {
            cfg.step_response = true;
        }
        if bode_requested {
            cfg.bode = true;
        }
        if desync_requested {
            cfg.motor_erpm = true;
        }
        cfg
    } else {
        PlotConfig::default()
    };

    // Show debug information when the runtime --debug flag is present
    if debug_mode {
        println!(
            "DEBUG: extended={}, step={}, bode={}, desync={}, plot_config={:?}",
            extended_requested, step_requested, bode_requested, desync_requested, plot_config
        );
    }

    // Exit if only --version flag was set
    if version_flag_set {
        return Ok(());
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

    // Construct AnalysisOptions once before the loop; cloned per file at the process_file call
    // site below (not Copy — carries the raw --start/--end argument text as owned Strings).
    let analysis_opts = AnalysisOptions {
        setpoint_threshold,
        show_legend,
        debug_mode,
        show_butterworth,
        estimate_optimal_p,
        trim_start,
        trim_end,
        trim_start_str,
        trim_end_str,
    };

    // Group input files by aircraft key for two-phase processing.
    // Phase 1 (profiling) aggregates throttle-punch events across all logs for each group.
    // Phase 2 (processing) runs the standard per-file analysis with the group's Td target.
    let grouped_files =
        group_files_by_aircraft(&input_files, debug_mode, analysis_opts.estimate_optimal_p);

    let mut overall_success = true;
    for (craft_key, group_files) in &grouped_files {
        // Phase 1: torque-inertia profiling across all files in the group.
        // Reuses each file's parse result in Phase 2 below instead of re-parsing.
        let (aircraft_profile, parsed_cache) = if analysis_opts.estimate_optimal_p {
            println!(
                "\n--- Torque-Inertia Profiling: '{}' ({} file(s)) ---",
                craft_key,
                group_files.len()
            );
            let (profile, cache) = profile_aircraft_group(group_files, debug_mode);
            print!("{}", profile.summary());
            (profile, cache)
        } else {
            (AircraftProfile::default(), Vec::new())
        };

        // Phase 2: process each file in the group. parsed_cache (when non-empty) is
        // ordered identically to group_files, so a plain iterator pairs them up.
        let mut parsed_cache_iter = parsed_cache.into_iter();
        for input_file_str in group_files {
            let actual_output_dir = match &output_dir {
                None => Path::new(input_file_str).parent(),
                Some(dir) => Some(Path::new(dir)),
            };
            let pre_parsed = parsed_cache_iter.next();

            if let Err(e) = process_file(
                input_file_str,
                use_dir_prefix_for_root_name,
                actual_output_dir,
                plot_config,
                analysis_opts.clone(),
                &aircraft_profile,
                pre_parsed,
            ) {
                eprintln!("Error: Processing {input_file_str}: {e}");
                overall_success = false;
            }
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

#[cfg(test)]
mod aircraft_grouping_tests {
    use super::*;

    /// Writes a synthetic log CSV with an embedded header block, followed by one data row,
    /// inside a unique temp subdirectory. `cargo test` runs tests concurrently by default, so
    /// two tests writing/removing the same OS temp-dir path race non-deterministically; a
    /// unique subdirectory (rather than a filename suffix) keeps the exact filename callers
    /// ask for, since `extract_aircraft_key`'s date-pattern parsing is sensitive to it.
    /// `header_lines` are raw `key,value` lines emitted before the CSV data-header row.
    fn write_synthetic_log(file_name: &str, header_lines: &[&str]) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("bbcsvr_test_{}_{unique_id}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("failed to create synthetic test log dir");
        let path = dir.join(file_name);
        let mut contents = String::new();
        for line in header_lines {
            contents.push_str(line);
            contents.push('\n');
        }
        contents.push_str("time (us),axisP[0],gyroADC[0]\n0,0,0\n");
        std::fs::write(&path, contents).expect("failed to write synthetic test log");
        path
    }

    /// Removes a synthetic log and the unique temp subdirectory `write_synthetic_log` created
    /// for it.
    fn cleanup_synthetic_log(path: &Path) {
        if let Some(dir) = path.parent() {
            std::fs::remove_dir_all(dir).ok();
        }
    }

    #[test]
    fn header_craft_name_overrides_generic_filename_key() {
        let path = write_synthetic_log("bbcsvr_test_craft_override.csv", &["Craft name,MyQuad"]);
        let key = aircraft_group_key(&path, false, true);
        cleanup_synthetic_log(&path);
        assert_eq!(key, "craft:MyQuad");
    }

    #[test]
    fn header_lookup_is_skipped_when_disabled() {
        let path = write_synthetic_log("bbcsvr_test_craft_disabled.csv", &["Craft name,MyQuad"]);
        let key = aircraft_group_key(&path, false, false);
        let expected = extract_aircraft_key(&path);
        cleanup_synthetic_log(&path);
        assert_eq!(key, expected);
        assert!(!key.starts_with("craft:"));
    }

    #[test]
    fn missing_craft_name_header_falls_back_to_filename_key() {
        let path = write_synthetic_log(
            "bbcsvr_test_no_craft_header.csv",
            &["Firmware revision,4.5.0"],
        );
        let key = aircraft_group_key(&path, false, true);
        let expected = extract_aircraft_key(&path);
        cleanup_synthetic_log(&path);
        assert_eq!(key, expected);
        assert!(!key.starts_with("craft:"));
    }

    #[test]
    fn no_header_metadata_at_all_falls_back_to_filename_key() {
        let path = write_synthetic_log("bbcsvr_test_no_header_at_all.csv", &[]);
        let key = aircraft_group_key(&path, false, true);
        let expected = extract_aircraft_key(&path);
        cleanup_synthetic_log(&path);
        assert_eq!(key, expected);
    }

    #[test]
    fn whitespace_only_craft_name_is_treated_as_absent() {
        let path = write_synthetic_log("bbcsvr_test_blank_craft_name.csv", &["Craft name,   "]);
        let key = aircraft_group_key(&path, false, true);
        let expected = extract_aircraft_key(&path);
        cleanup_synthetic_log(&path);
        assert_eq!(key, expected);
        assert!(!key.starts_with("craft:"));
    }

    /// A blank `Craft name` line ahead of a real one (e.g. a stray or malformed duplicate
    /// header) must not shadow the later, valid entry — the first match by key alone would
    /// return `None` and lose real craft-identity data that was present in the file.
    #[test]
    fn blank_craft_name_entry_does_not_shadow_a_later_valid_one() {
        let path = write_synthetic_log(
            "bbcsvr_test_duplicate_craft_name.csv",
            &["Craft name,", "Craft name,MyQuad"],
        );
        let key = aircraft_group_key(&path, false, true);
        cleanup_synthetic_log(&path);
        assert_eq!(key, "craft:MyQuad");
    }

    /// Two different aircraft exported with an identical generic filename pattern (no craft
    /// suffix after the timestamp, e.g. an MSC-mode export) used to collapse into one
    /// `extract_aircraft_key` group. A distinct header `Craft name` on each file must now keep
    /// them in separate groups.
    #[test]
    fn distinct_craft_names_split_groups_despite_identical_generic_filename_key() {
        let path_a = write_synthetic_log(
            "BTFL_BLACKBOX_LOG_20250101_120000.csv",
            &["Craft name,QuadA"],
        );
        let path_b = write_synthetic_log(
            "BTFL_BLACKBOX_LOG_20250101_120001.csv",
            &["Craft name,QuadB"],
        );

        // Both files collapse to the same bare filename-derived key (no craft suffix present).
        assert_eq!(extract_aircraft_key(&path_a), extract_aircraft_key(&path_b));

        let key_a = aircraft_group_key(&path_a, false, true);
        let key_b = aircraft_group_key(&path_b, false, true);
        cleanup_synthetic_log(&path_a);
        cleanup_synthetic_log(&path_b);

        assert_ne!(key_a, key_b);
        assert_eq!(key_a, "craft:QuadA");
        assert_eq!(key_b, "craft:QuadB");
    }

    /// Documents a known, accepted limitation (not a bug to fix): when neither file carries a
    /// usable header `Craft name` AND both filenames collapse to the identical generic key,
    /// there is no remaining signal to distinguish two different physical aircraft — they
    /// still merge into one group. `craft:`-namespacing only prevents a header-sourced key
    /// from coincidentally colliding with a filename-derived one; it cannot invent an identity
    /// signal where none of the inputs provide one.
    #[test]
    fn two_generic_filenames_with_no_craft_name_still_merge() {
        let path_a = write_synthetic_log("BTFL_BLACKBOX_LOG_20250101_120000.csv", &[]);
        let path_b = write_synthetic_log("BTFL_BLACKBOX_LOG_20250101_120001.csv", &[]);

        let key_a = aircraft_group_key(&path_a, false, true);
        let key_b = aircraft_group_key(&path_b, false, true);
        cleanup_synthetic_log(&path_a);
        cleanup_synthetic_log(&path_b);

        assert_eq!(key_a, key_b);
        assert!(!key_a.starts_with("craft:"));
    }
}
