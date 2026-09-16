// src/types.rs
// Type aliases to reduce complexity warnings

#![allow(dead_code)] // Allow unused type aliases - they're for future use

use crate::axis_names::AXIS_COUNT;
use ndarray::{Array1, Array2};
use std::error::Error;

// Compile-time assertion: AXIS_COUNT must be 3 for backward compatibility.
// Public type aliases (StepResponseResults, AllFFTData, AllPSDData, etc.) depend on
// AXIS_COUNT. Changing AXIS_COUNT is a breaking change and requires API migration.
const _: () = assert!(
    AXIS_COUNT == 3,
    "AXIS_COUNT must be 3 for backward compatibility"
);

// Step response calculation types
pub type StepResponseResult = (Array1<f64>, Array2<f32>, Array1<f32>);
pub type StepResponseResults = [Option<StepResponseResult>; AXIS_COUNT];

// Log parser return type
// Note: [bool; 3] and [bool; 4] arrays are header detection flags (not axis-specific).
// They must remain hardcoded and do NOT reference AXIS_COUNT.
// Input expansion return type (main.rs `expand_input_paths`/`find_csv_files_in_dir`):
// list of resolved CSV file paths, count of skipped subdirectories, a map from each BBL-derived
// scratch CSV path to its source `.bbl`'s own parent directory (for output-dir defaulting), and
// the list of scratch directories that must be removed at end of program (empty entries for
// `--keep` output, which must never be removed) — see `bbl_reader::expand_bbl_to_scratch_csvs`.
pub type InputExpansionResult = (
    Vec<String>,
    usize,
    std::collections::HashMap<String, std::path::PathBuf>,
    Vec<std::path::PathBuf>,
);

// BBL expansion return type (bbl_reader::expand_bbl_to_scratch_csvs): one (scratch_csv_path,
// original_bbl_parent_dir) pair per exported flight, and the scratch directory to remove
// afterward (None under --keep — see the function's own doc comment for the full contract).
pub type BblExpansionResult = (
    Vec<(String, std::path::PathBuf)>,
    Option<std::path::PathBuf>,
);

pub type LogParseResult = Result<
    (
        Vec<crate::data_input::log_data::LogRowData>,
        Option<f64>,
        [bool; 3],             // f_term_header_found (fixed: 3 fields)
        [bool; 4],             // setpoint_header_found (fixed: 4 fields)
        [bool; 3],             // gyro_header_found (fixed: 3 fields)
        [bool; 3],             // gyro_unfilt_header_found (fixed: 3 fields)
        [bool; 4],             // debug_header_found (fixed: 4 fields)
        bool,                  // using_debug_fallback (debug_mode-validated, see log_parser)
        Vec<(String, String)>, // header_metadata
    ),
    Box<dyn Error>,
>;

// FFT data types for plotting
pub type AxisFFTData = (
    Vec<(f64, f64)>, // gyro_filtered
    Vec<(f64, f64)>, // gyro_unfiltered
    Vec<(f64, f64)>, // noise
    Vec<(f64, f64)>, // blackbox_noise
);
pub type AllFFTData = [Option<AxisFFTData>; AXIS_COUNT];

// PSD data types for plotting
pub type AxisPSDData = (
    Vec<(f64, f64)>, // gyro_filtered
    Vec<(f64, f64)>, // gyro_unfiltered
    Vec<(f64, f64)>, // noise
    Vec<(f64, f64)>, // blackbox_noise
);
pub type AllPSDData = [Option<AxisPSDData>; AXIS_COUNT];

// Plot data types
pub type AxisPlotData2 = Vec<(f64, Option<f64>, Option<f64>)>;
pub type AxisPlotData2Simple = Vec<(f64, f64)>; // For derivative and simple time-series plots
pub type AxisPlotData3 = Vec<(f64, Option<f64>, Option<f64>, Option<f64>)>;
pub type AllAxisPlotData2 = [AxisPlotData2; AXIS_COUNT];
pub type AllAxisPlotData2Simple = [AxisPlotData2Simple; AXIS_COUNT];
pub type AllAxisPlotData3 = [AxisPlotData3; AXIS_COUNT];

// Step response plot data
pub type StepResponsePlotData = (
    String,                                 // title
    std::ops::Range<f64>,                   // x_range
    std::ops::Range<f64>,                   // y_range
    Vec<crate::plot_framework::PlotSeries>, // series
    String,                                 // x_label
    String,                                 // y_label
);
pub type AllStepResponsePlotData = [Option<StepResponsePlotData>; AXIS_COUNT];
