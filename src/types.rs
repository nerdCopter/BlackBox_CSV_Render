// src/types.rs
// Type aliases to reduce complexity warnings

#![allow(dead_code)] // Allow unused type aliases - they're for future use

use ndarray::{Array1, Array2};
use std::error::Error;

// Step response calculation types
pub type StepResponseResult = (Array1<f64>, Array2<f32>, Array1<f32>);
pub type StepResponseResults = [Option<StepResponseResult>; 3];

// Log parser return type
pub type LogParseResult = Result<
    (
        Vec<crate::data_input::log_data::LogRowData>,
        Option<f64>,
        [bool; 3],             // f_term_header_found
        [bool; 4],             // setpoint_header_found
        [bool; 3],             // gyro_header_found
        [bool; 3],             // gyro_unfilt_header_found
        [bool; 4],             // debug_header_found
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
pub type AllFFTData = [Option<AxisFFTData>; 3];

// PSD data types for plotting
pub type AxisPSDData = (
    Vec<(f64, f64)>, // gyro_filtered
    Vec<(f64, f64)>, // gyro_unfiltered
    Vec<(f64, f64)>, // noise
    Vec<(f64, f64)>, // blackbox_noise
);
pub type AllPSDData = [Option<AxisPSDData>; 3];

// Plot data types
pub type AxisPlotData2 = Vec<(f64, Option<f64>, Option<f64>)>;
pub type AxisPlotData3 = Vec<(f64, Option<f64>, Option<f64>, Option<f64>)>;
pub type AllAxisPlotData2 = [AxisPlotData2; 3];
pub type AllAxisPlotData3 = [AxisPlotData3; 3];

// Step response plot data
pub type StepResponsePlotData = (
    String,                                 // title
    std::ops::Range<f64>,                   // x_range
    std::ops::Range<f64>,                   // y_range
    Vec<crate::plot_framework::PlotSeries>, // series
    String,                                 // x_label
    String,                                 // y_label
);
pub type AllStepResponsePlotData = [Option<StepResponsePlotData>; 3];
