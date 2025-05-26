// src/constants.rs

// Import specific colors needed
use plotters::style::RGBColor;
use plotters::style::colors::full_palette::{GREEN, AMBER, ORANGE, LIGHTBLUE, RED, PURPLE};

// Plot dimensions.
pub const PLOT_WIDTH: u32 = 1920;
pub const PLOT_HEIGHT: u32 = 1080;

// Step response plot duration in seconds.
pub const STEP_RESPONSE_PLOT_DURATION_S: f64 = 0.5;

// Constants for the new step response calculation method (inspired by Python)
pub const FRAME_LENGTH_S: f64 = 1.0; // Length of each window in seconds
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep from each window
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows within a frame length
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)
pub const SETPOINT_THRESHOLD: f64 = 500.0; // Threshold for low/high setpoint masking

// Constants for filtering data based on movement and flight phase.
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0; // Minimum setpoint/gyro magnitude for a window to be considered for analysis (from PTstepcalc.m minInput)
pub const EXCLUDE_START_S: f64 = 3.0; // Exclude this many seconds from the start of the log
pub const EXCLUDE_END_S: f64 = 3.0; // Exclude this many seconds from the end of the log

// Constant for post-averaging smoothing of the step response curves.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 5; // Moving average window size (in samples)

// Constants for the spectrum plot
pub const SPECTRUM_Y_AXIS_FLOOR: f64 = 20000.0; // Maximum amplitude for spectrum plots.
pub const SPECTRUM_NOISE_FLOOR_HZ: f64 = 70.0; // Frequency threshold below which to ignore for dynamic Y-axis scaling (e.g., motor idle noise).
pub const SPECTRUM_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2; // Factor to extend Y-axis above the highest peak (after noise floor) for better visibility.
pub const PEAK_LABEL_MIN_AMPLITUDE: f64 = 1000.0;

// Constants for PSD plots (newly added)
pub const PSD_Y_AXIS_FLOOR: f64 = 1e-6; // A reasonable floor for PSD values to prevent log scaling issues with zero.
pub const PSD_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2; // Factor to extend Y-axis above the highest peak.
pub const PSD_PEAK_LABEL_MIN_VALUE: f64 = 1e-3; // Minimum PSD value for a peak to be labeled.

// Constants for Spectrogram/Heatmap plots (NEW)
pub const STFT_WINDOW_DURATION_S: f64 = 0.1; // Duration of each STFT window in seconds
pub const STFT_OVERLAP_FACTOR: f64 = 0.75; // Overlap between windows (e.g., 0.75 for 75% overlap)
pub const HEATMAP_MIN_PSD_DB: f64 = -80.0; // Minimum PSD value in dB for heatmap color scaling
pub const HEATMAP_MAX_PSD_DB: f64 = -10.0; // Maximum PSD value in dB for heatmap color scaling (adjust based on typical values)

// Constants for spectrum peak labeling
pub const MAX_PEAKS_TO_LABEL: usize = 3; // Max number of peaks (including primary) to label on spectrum plots
pub const MIN_SECONDARY_PEAK_FACTOR: f64 = 0.05; // Secondary peak must be at least this factor of primary peak's amplitude
pub const MIN_PEAK_SEPARATION_HZ: f64 = 70.0; // Minimum frequency separation between reported peaks on spectrum plots
// Constants for advanced peak detection
pub const ENABLE_WINDOW_PEAK_DETECTION: bool = true; // Set to true to use window-based peak detection
                                                     // Set to false to use the previous 3-point (amp > prev && amp >= next) logic.
pub const PEAK_DETECTION_WINDOW_RADIUS: usize = 3;   // Radius W for peak detection window (total 2*W+1 points).

// Constants for individual window step response quality control (from PTstepcalc.m)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_END_S: f64 = 0.5; // End time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_MIN_VAL: f64 = 0.5; // Minimum allowed value in steady-state for quality control (applied to UN-NORMALIZED response mean)
pub const STEADY_STATE_MAX_VAL: f64 = 3.0; // Maximum allowed value in steady-state for quality control (applied to UN-NORMALIZED response)

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
