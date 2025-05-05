// src/constants.rs

// Import specific colors needed
use plotters::style::RGBColor;
// Import standard colors from full_palette
// Removed unused imports BLUE_900, GREEN_900 from full_palette
// Added YELLOW back for spectrograph colormap
// Removed GREY import as it's used in plotting_utils, not here
use plotters::style::colors::full_palette::{GREEN,AMBER,ORANGE,LIGHTBLUE,RED,BLUE, YELLOW};
// Removed unused manual CORNFLOWERBLUE definition
// pub const COLOR_CORNFLOWERBLUE: RGBColor = RGBColor(100, 149, 237);
// BLACK and WHITE are needed for spectrograph colormap stops
use plotters::style::colors::{BLACK, WHITE};


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
// Throttle threshold for filtering data (e.g., for spectrograph during flight)
pub const THROTTLE_THRESHOLD: f64 = 100.0; // Assuming throttle is on a 0-1000 scale, adjust if needed
pub const EXCLUDE_END_S: f64 = 5.0; // Exclude this many seconds from the end of the log

// Maximum raw throttle value for percentage conversion (assuming 0-1000 scale based on user correction)
pub const MAX_RAW_THROTTLE: f64 = 1000.0;


// Constant for post-averaging smoothing of the step response curves.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 5; // Moving average window size (in samples)

// Constants for individual window step response quality control (from PTstepcalc.m)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_END_S: f64 = 0.5; // End time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_MIN_VAL: f64 = 0.5; // Minimum allowed value in steady-state for quality control (applied to UN-NORMALIZED response mean)
pub const STEADY_STATE_MAX_VAL: f64 = 3.0; // Maximum allowed value in steady-state for quality control (applied to UN-NORMALIZED response)


// --- Spectrograph Constants ---
pub const SPECTROGRAPH_WINDOW_S: f64 = 0.1; // Window length for STFT in seconds (e.g., 100ms)
pub const SPECTROGRAPH_OVERLAP_RATIO: f64 = 0.75; // Overlap between windows (0.0 to 1.0, 0.75 is typical)
// SPECTROGRAPH_MAX_FREQ_HZ is now just a hint for default plotting range if data is empty
pub const SPECTROGRAPH_MAX_FREQ_HZ: f64 = 400.0; // Maximum frequency to show in spectrograph plots

// Spectrograph Color Mapping (Log Power dB)
pub const SPECTROGRAPH_MIN_LOG_POWER_DB: f64 = -80.0; // Minimum log power (dB) mapped to the lowest color
pub const SPECTROGRAPH_MAX_LOG_POWER_DB: f64 = 0.0; // Maximum log power (dB) mapped to the highest color

// Spectrograph Colormap stops (4 stops: Black, Red, Yellow, White)
// Using colors from the full_palette image or defined manually
pub const COLOR_STOP_0: &RGBColor = &BLACK; // Black (lowest power)
pub const COLOR_STOP_1: &RGBColor = &RED; // Red
pub const COLOR_STOP_2: &RGBColor = &YELLOW; // Yellow
pub const COLOR_STOP_3: &RGBColor = &WHITE; // White (highest power)


// --- Plot Color Assignments ---

// PIDsum vs PID Error vs Setpoint Plot
pub const COLOR_PIDSUM_MAIN: &RGBColor = &GREEN;
pub const COLOR_PIDERROR_MAIN: &RGBColor = &BLUE;
pub const COLOR_SETPOINT_MAIN: &RGBColor = &ORANGE;

// Setpoint vs PIDsum Plot
pub const COLOR_SETPOINT_VS_PIDSUM_SP: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_PIDSUM_PID: &RGBColor = &GREEN;

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