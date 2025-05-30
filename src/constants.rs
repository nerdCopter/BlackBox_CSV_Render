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
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep (500ms, matches PTstepcalc's wnd/StepRespDuration_ms)
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows within a frame length (can be tuned)
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window, matches PTstepcalc)

// Initial Gyro Smoothing (applied before deconvolution)
// 0 for no smoothing. Otherwise, window size for moving average.
// PTstepcalc uses LOESS with variable window (e.g., 20, 40, 60 samples based on smoothFactor).
// A simple moving average is implemented here; 10-20 might be a starting point if > 0.
// For PTB similarity, some level of pre-smoothing is often beneficial.
pub const INITIAL_GYRO_SMOOTHING_WINDOW: usize = 15; // Example: Set to 10 or 20 to enable, or 0 to disable

// Individual Response "Y-Correction" (Normalization before averaging, mimics PTB)
pub const APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION: bool = true;
// If Y-correction is applied, this is the minimum absolute mean of the unnormalized steady-state
// required to attempt the correction. Prevents extreme scaling/division by near-zero.
pub const Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS: f32 = 0.1; // Tune this threshold if needed

// Quality Control for *individually Y-corrected* (or uncorrected if Y_CORRECTION flag is false) responses
// These values mimic PTstepcalc.m's QC: min > 0.5 && max < 3 on the (potentially) Y-corrected steady-state.
pub const NORMALIZED_STEADY_STATE_MIN_VAL: f32 = 0.5;
pub const NORMALIZED_STEADY_STATE_MAX_VAL: f32 = 3.0;

// Optional: Additional check on the mean of the Y-corrected steady-state.
// If enabled, the mean of the Y-corrected steady state should be within these bounds.
pub const ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK: bool = true;
pub const NORMALIZED_STEADY_STATE_MEAN_MIN: f32 = 0.75; // e.g., mean should be > 0.75 after Y-correction
pub const NORMALIZED_STEADY_STATE_MEAN_MAX: f32 = 1.25; // e.g., mean should be < 1.25 after Y-correction

// Steady-state definition for Y-correction and QC (matches PTstepcalc.m: 200ms to 500ms of the 500ms response)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check (200ms into the 500ms response)
pub const STEADY_STATE_END_S: f64 = 0.5;   // End time for steady-state check (effectively to the end of RESPONSE_LENGTH_S)

// Constant for post-averaging smoothing of the final step response curves.
// This is applied in the plotting stage, after responses are averaged.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 15; // Moving average window size (in samples), you set this to 15

/*
// Constants for previous unnormalized step-response Quality-Control (Now superseded by Y-correction approach)
pub const UNNORMALIZED_MEAN_THRESHOLD_FOR_RELATIVE_STD_CHECK: f32 = 1.0;
pub const UNNORMALIZED_RELATIVE_STD_DEV_MAX: f32 = 0.35;
pub const UNNORMALIZED_ABSOLUTE_STD_DEV_MAX_FOR_SMALL_MEAN: f32 = 0.55;
*/

// Default setpoint threshold, can be overridden at runtime for categorizing responses
pub const DEFAULT_SETPOINT_THRESHOLD: f64 = 500.0;

// Constants for filtering data based on movement and flight phase.
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0; // Minimum setpoint/gyro magnitude for a window to be considered (from PTB/PlasmaTree)
pub const EXCLUDE_START_S: f64 = 3.0; // Exclude this many seconds from the start of the log
pub const EXCLUDE_END_S: f64 = 3.0; // Exclude this many seconds from the end of the log


// Constants for the spectrum plot (linear amplitude)
pub const SPECTRUM_Y_AXIS_FLOOR: f64 = 20000.0; // Maximum amplitude for spectrum plots.
pub const SPECTRUM_NOISE_FLOOR_HZ: f64 = 70.0; // Frequency threshold below which to ignore for dynamic Y-axis scaling
pub const SPECTRUM_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2; // Factor to extend Y-axis above the highest peak
pub const PEAK_LABEL_MIN_AMPLITUDE: f64 = 1000.0;

// Constants for PSD plots (dB scale)
pub const PSD_Y_AXIS_FLOOR_DB: f64 = -80.0; // A reasonable floor for PSD values in dB
pub const PSD_Y_AXIS_HEADROOM_FACTOR_DB: f64 = 10.0; // Factor to extend Y-axis above the highest peak (in dB)
pub const PSD_PEAK_LABEL_MIN_VALUE_DB: f64 = -60.0; // Minimum PSD value in dB for a peak to be labeled.

// Constants for Spectrogram/Heatmap plots (re-introduced)
pub const STFT_WINDOW_DURATION_S: f64 = 0.1; // Duration of each STFT window in seconds
pub const STFT_OVERLAP_FACTOR: f64 = 0.75; // Overlap between windows (e.g., 0.75 for 75% overlap)
pub const HEATMAP_MIN_PSD_DB: f64 = -80.0; // Minimum PSD value in dB for heatmap color scaling
pub const HEATMAP_MAX_PSD_DB: f64 = -10.0; // Maximum PSD value in dB for heatmap color scaling

// Constants for Throttle-Frequency Heatmap
pub const THROTTLE_Y_BINS_COUNT: usize = 50; // Number of bins for the throttle (Y) axis
pub const THROTTLE_Y_MIN_VALUE: f64 = 0.0; // Minimum throttle value for plotting range
pub const THROTTLE_Y_MAX_VALUE: f64 = 1000.0; // Maximum throttle value for plotting range

// Constants for spectrum peak labeling
pub const MAX_PEAKS_TO_LABEL: usize = 3; // Max number of peaks (including primary) to label on spectrum plots
pub const MIN_SECONDARY_PEAK_RATIO: f64 = 0.05; // Secondary peak must be ≥ this linear ratio of the primary peak’s amplitude
pub const MIN_PEAK_SEPARATION_HZ: f64 = 70.0; // Minimum frequency separation between reported peaks on spectrum plots
// Constants for advanced peak detection
pub const ENABLE_WINDOW_PEAK_DETECTION: bool = true; // Set to true to use window-based peak detection
pub const PEAK_DETECTION_WINDOW_RADIUS: usize = 3;   // Radius W for peak detection window (total 2*W+1 points).


// --- Plot Color Assignments (Based on Screenshots) ---

// PIDsum vs PID Error vs Setpoint Plot
pub const COLOR_PIDSUM_MAIN: &RGBColor = &GREEN;
pub const COLOR_PIDERROR_MAIN: &RGBColor = &PURPLE;
pub const COLOR_SETPOINT_MAIN: &RGBColor = &ORANGE;

// Setpoint vs Gyro Plot
pub const COLOR_SETPOINT_VS_GYRO_SP: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_GYRO: &RGBColor = &LIGHTBLUE;

// Gyro vs Unfilt Gyro Plot
pub const COLOR_GYRO_VS_UNFILT_FILT: &RGBColor = &LIGHTBLUE;
pub const COLOR_GYRO_VS_UNFILT_UNFILT: &RGBColor = &AMBER;

// Step Response Plot
pub const COLOR_STEP_RESPONSE_LOW_SP: &RGBColor = &LIGHTBLUE;
pub const COLOR_STEP_RESPONSE_HIGH_SP: &RGBColor = &ORANGE;
pub const COLOR_STEP_RESPONSE_COMBINED: &RGBColor = &RED;

// Stroke widths for lines
pub const LINE_WIDTH_PLOT: u32 = 1; // Width for plot lines
pub const LINE_WIDTH_LEGEND: u32 = 2; // Width for legend lines

// src/constants.rs