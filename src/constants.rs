// src/constants.rs

use plotters::prelude::full_palette::ORANGE;
use plotters::style::RGBColor;
use plotters::prelude::RED; // Import RED

// Plot dimensions.
pub const PLOT_WIDTH: u32 = 1920;
pub const PLOT_HEIGHT: u32 = 1080;

// Step response plot duration in seconds.
pub const STEP_RESPONSE_PLOT_DURATION_S: f64 = 0.5;

// Constants for the new step response calculation method (inspired by Python)
pub const FRAME_LENGTH_S: f64 = 1.0; // Length of each window in seconds
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep from each window
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows within a frame length
#[allow(dead_code)] // Not currently used in deconvolution, but kept as it was in original
pub const CUTOFF_FREQUENCY_HZ: f64 = 25.0; // Cutoff frequency for Wiener filter noise spectrum (used in SN calculation, though simplified)
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)
pub const SETPOINT_THRESHOLD: f64 = 500.0; // Threshold for low/high setpoint masking

// Constants for filtering data based on movement and flight phase.
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0; // Minimum setpoint/gyro magnitude for a window to be considered for analysis (from PTstepcalc.m minInput)
pub const EXCLUDE_START_S: f64 = 5.0; // Exclude this many seconds from the start of the log
pub const EXCLUDE_END_S: f64 = 5.0; // Exclude this many seconds from the end of the log

// Constant for post-averaging smoothing of the step response curves.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 5; // Moving average window size (in samples)

// Constants for individual window step response quality control (from PTstepcalc.m)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check within the response window
pub const STEADY_STATE_END_S: f64 = 0.5; // End time for steady-state check within the response window
pub const STEADY_STATE_MIN_VAL: f64 = 0.5; // Minimum allowed value in steady-state for quality control
pub const STEADY_STATE_MAX_VAL: f64 = 3.0; // Maximum allowed value in steady-state for quality control

// Plot colors
pub const COLOR_PIDSUM: usize = 1;
pub const COLOR_SETPOINT: usize = 2;
pub const COLOR_PIDSUM_VS_SETPOINT: usize = 0;
pub const COLOR_STEP_RESPONSE_LOW_SP: usize = 3;
pub const COLOR_STEP_RESPONSE_HIGH_SP: &RGBColor = &ORANGE;
pub const COLOR_STEP_RESPONSE_COMBINED: &RGBColor = &RED; // New color for combined response
pub const COLOR_GYRO_UNFILT: usize = 4;
pub const COLOR_GYRO_FILT: usize = 5;