// src/constants.rs

// Import specific colors needed
use plotters::style::{RGBColor}; 
use plotters::style::colors::full_palette::{GREEN, AMBER, ORANGE, LIGHTBLUE, RED, PURPLE, WHITE, YELLOW};


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

// Constants for individual window step response quality control (from PTstepcalc.m)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_END_S: f64 = 0.5; // End time for steady-state check within the response window (relative to response start)
pub const STEADY_STATE_MIN_VAL: f64 = 0.5; // Minimum allowed value in steady-state for quality control (applied to UN-NORMALIZED response mean)
pub const STEADY_STATE_MAX_VAL: f64 = 3.0; // Maximum allowed value in steady-state for quality control (applied to UN-NORMALIZED response)

// --- Plot Color Assignments (Based on Screenshots) ---
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

// --- Spectrogram Constants ---
pub const SPECTROGRAM_THROTTLE_BINS: usize = 100; // BBE uses 100. You had 512, adjust if needed.
pub const SPECTROGRAM_FFT_TIME_WINDOW_MS: f64 = 300.0; // BBE uses 300ms
pub const SPECTROGRAM_FFT_OVERLAP_FACTOR: usize = 6; // BBE uses 6 (for hop size = window/6) -> ~83% overlap

pub const SPECTROGRAM_MAX_FREQ_HZ: f32 = 1000.0; // Set based on Nyquist (e.g., sample_rate / 2)

// For LOGARITHMIC scaling:
pub const MIN_POWER_FOR_LOG_SCALE: f32 = 0.0001; // EXPERIMENT! Linear power for bottom of log scale.
// SPECTROGRAM_POWER_CLIP_MAX is now auto-determined by calculate_throttle_psd and scaled by AUTO_CLIP_MAX_SCALE_FACTOR
pub const AUTO_CLIP_MAX_SCALE_FACTOR: f32 = 1.0; // 1.0 means clip at detected max. >1 gives headroom. <1 saturates more.

pub const SPECTROGRAM_NUM_COLORS: usize = 256; 

pub const HOT_COLORMAP_ANCHORS: &[(f32, RGBColor)] = &[
    (0.0, plotters::style::colors::BLACK), 
    (0.1, RGBColor(80, 0, 0)),   
    (0.25, RGBColor(180, 0, 0)), 
    (0.4, RED),                  
    (0.6, ORANGE),               
    (0.8, YELLOW),              
    (1.0, WHITE),                
];

pub const SPECTROGRAM_TEXT_COLOR: &RGBColor = &WHITE;
pub const SPECTROGRAM_GRID_COLOR: RGBColor = RGBColor(80,80,80);

// src/constants.rs
