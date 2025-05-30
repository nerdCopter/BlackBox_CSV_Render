// src/constants.rs

// Import specific colors needed
use plotters::style::RGBColor;
use plotters::style::colors::full_palette::{GREEN, AMBER, ORANGE, LIGHTBLUE, RED, PURPLE};

// Plot dimensions.
pub const PLOT_WIDTH: u32 = 1920;
pub const PLOT_HEIGHT: u32 = 1080;

// Step response plot duration in seconds.
pub const STEP_RESPONSE_PLOT_DURATION_S: f64 = 0.5;

// Constants for the step response calculation method (mimicking PTstepcalc.m)
pub const FRAME_LENGTH_S: f64 = 2.0; // Length of each window in seconds (Matlab uses 2s)
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep (500ms)
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows (can be tuned)
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)

// Initial Gyro Smoothing (applied before deconvolution)
pub const INITIAL_GYRO_SMOOTHING_WINDOW: usize = 15; // Set to 15 based on user's last test

// Individual Response "Y-Correction" (Normalization before averaging)
pub const APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION: bool = true;
// If Y-correction is applied, this is the minimum absolute mean of the unnormalized steady-state
// required to attempt the correction. Prevents extreme scaling/division by near-zero.
pub const Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS: f32 = 0.1; // Tune this threshold if needed (0.1 is a common starting point)

// Quality Control for *individually Y-corrected* (or uncorrected if Y_CORRECTION flag is false) responses
// These values mimic PTstepcalc.m's QC but are applied to responses targeting 1.0.
pub const NORMALIZED_STEADY_STATE_MIN_VAL: f32 = 0.5; // From PTB
pub const NORMALIZED_STEADY_STATE_MAX_VAL: f32 = 3.0; // From PTB

// Optional: Additional check on the mean of the Y-corrected steady-state.
pub const ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK: bool = true;
pub const NORMALIZED_STEADY_STATE_MEAN_MIN: f32 = 0.75; // e.g., mean should be > 0.75 after Y-correction
pub const NORMALIZED_STEADY_STATE_MEAN_MAX: f32 = 1.25; // e.g., mean should be < 1.25 after Y-correction

// Steady-state definition for Y-correction and QC (matches PTstepcalc.m: 200ms to 500ms of the 500ms response)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check (200ms into the 500ms response)
pub const STEADY_STATE_END_S: f64 = 0.5;   // End time for steady-state check (effectively to the end of RESPONSE_LENGTH_S)

// Constant for post-averaging smoothing of the final step response curves.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 15; // Kept at 15 as per user's last setting

/*
// Constants for previous unnormalized step-response Quality-Control (Now superseded by Y-correction approach)
pub const UNNORMALIZED_MEAN_THRESHOLD_FOR_RELATIVE_STD_CHECK: f32 = 1.0;
pub const UNNORMALIZED_RELATIVE_STD_DEV_MAX: f32 = 0.35;
pub const UNNORMALIZED_ABSOLUTE_STD_DEV_MAX_FOR_SMALL_MEAN: f32 = 0.55;
*/

// Default setpoint threshold, can be overridden at runtime for categorizing responses
pub const DEFAULT_SETPOINT_THRESHOLD: f64 = 500.0;

// Constants for filtering data based on movement and flight phase.
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0; // Minimum setpoint/gyro magnitude (from PTB/PlasmaTree)
pub const EXCLUDE_START_S: f64 = 3.0; // Exclude seconds from the start of the log
pub const EXCLUDE_END_S: f64 = 3.0; // Exclude seconds from the end of the log

// Constants for the spectrum plot (linear amplitude)
pub const SPECTRUM_Y_AXIS_FLOOR: f64 = 20000.0;
pub const SPECTRUM_NOISE_FLOOR_HZ: f64 = 70.0;
pub const SPECTRUM_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2;
pub const PEAK_LABEL_MIN_AMPLITUDE: f64 = 1000.0;

// Constants for PSD plots (dB scale)
pub const PSD_Y_AXIS_FLOOR_DB: f64 = -80.0;
pub const PSD_Y_AXIS_HEADROOM_FACTOR_DB: f64 = 10.0;
pub const PSD_PEAK_LABEL_MIN_VALUE_DB: f64 = -60.0;

// Constants for Spectrogram/Heatmap plots
pub const STFT_WINDOW_DURATION_S: f64 = 0.1;
pub const STFT_OVERLAP_FACTOR: f64 = 0.75;
pub const HEATMAP_MIN_PSD_DB: f64 = -80.0;
pub const HEATMAP_MAX_PSD_DB: f64 = -10.0;

// Constants for Throttle-Frequency Heatmap
pub const THROTTLE_Y_BINS_COUNT: usize = 50;
pub const THROTTLE_Y_MIN_VALUE: f64 = 0.0;
pub const THROTTLE_Y_MAX_VALUE: f64 = 1000.0;

// Constants for spectrum peak labeling
pub const MAX_PEAKS_TO_LABEL: usize = 3;
pub const MIN_SECONDARY_PEAK_RATIO: f64 = 0.05;
pub const MIN_PEAK_SEPARATION_HZ: f64 = 70.0;
pub const ENABLE_WINDOW_PEAK_DETECTION: bool = true;
pub const PEAK_DETECTION_WINDOW_RADIUS: usize = 3;

// --- Plot Color Assignments ---
pub const COLOR_PIDSUM_MAIN: &RGBColor = &GREEN;
pub const COLOR_PIDERROR_MAIN: &RGBColor = &PURPLE;
pub const COLOR_SETPOINT_MAIN: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_SP: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_GYRO: &RGBColor = &LIGHTBLUE;
pub const COLOR_GYRO_VS_UNFILT_FILT: &RGBColor = &LIGHTBLUE;
pub const COLOR_GYRO_VS_UNFILT_UNFILT: &RGBColor = &AMBER;
pub const COLOR_STEP_RESPONSE_LOW_SP: &RGBColor = &LIGHTBLUE;
pub const COLOR_STEP_RESPONSE_HIGH_SP: &RGBColor = &ORANGE;
pub const COLOR_STEP_RESPONSE_COMBINED: &RGBColor = &RED;

// Stroke widths for lines
pub const LINE_WIDTH_PLOT: u32 = 1;
pub const LINE_WIDTH_LEGEND: u32 = 2;

// src/constants.rs
