// src/constants.rs

// Import specific colors needed
use plotters::style::colors::full_palette::{AMBER, GREEN, LIGHTBLUE, ORANGE, PURPLE, RED, WHITE};
use plotters::style::{RGBAColor, RGBColor};

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
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0;
pub const EXCLUDE_START_S: f64 = 3.0;
pub const EXCLUDE_END_S: f64 = 3.0;

// Constant for post-averaging smoothing of the step response curves.
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 5;

// Constants for individual window step response quality control
pub const STEADY_STATE_START_S: f64 = 0.2;
pub const STEADY_STATE_END_S: f64 = 0.5;
pub const STEADY_STATE_MIN_VAL: f64 = 0.5;
pub const STEADY_STATE_MAX_VAL: f64 = 3.0;

// --- Plot Color Assignments ---
pub const COLOR_PIDSUM_MAIN: &RGBColor = &GREEN;
pub const COLOR_PIDERROR_MAIN: &RGBColor = &PURPLE;
pub const COLOR_SETPOINT_MAIN: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_SP: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_GYRO: &RGBColor = &LIGHTBLUE;
pub const COLOR_GYRO_VS_UNFILT_FILT: RGBAColor = RGBAColor(LIGHTBLUE.0, LIGHTBLUE.1, LIGHTBLUE.2, 0.1);
pub const COLOR_GYRO_VS_UNFILT_UNFILT: RGBAColor = RGBAColor(AMBER.0, AMBER.1, AMBER.2, 1.0);
pub const COLOR_STEP_RESPONSE_LOW_SP: &RGBColor = &LIGHTBLUE;
pub const COLOR_STEP_RESPONSE_HIGH_SP: &RGBColor = &ORANGE;
pub const COLOR_STEP_RESPONSE_COMBINED: &RGBColor = &RED;

// Stroke widths for lines
pub const LINE_WIDTH_PLOT: u32 = 1;
pub const LINE_WIDTH_LEGEND: u32 = 2;

// --- Spectrogram Constants ---
pub const SPECTROGRAM_THROTTLE_BINS: usize = 100;
pub const SPECTROGRAM_FFT_TIME_WINDOW_MS: f64 = 300.0;
pub const SPECTROGRAM_FFT_OVERLAP_FACTOR: usize = 6;

pub const SPECTROGRAM_MAX_FREQ_HZ: f32 = 1000.0;

// BBE uses a fixed scaling factor for its heatmap cell colors.
// An averaged magnitude of BBE_SCALE_HEATMAP in a cell corresponds to 100% lightness (white) *before gamma*.
pub const BBE_SCALE_HEATMAP: f32 = 1.3;
pub const MIN_VISIBLE_SPECTROGRAM_LIGHTNESS: f32 = 0.05; // A very small non-zero lightness value to ensure visibility

// MIN_POWER_FOR_LOG_SCALE is used as a general small value, e.g. for calculating mean of non-trivial values
pub const MIN_POWER_FOR_LOG_SCALE: f32 = 0.00001;

// Gamma value for spectrogram color mapping to adjust brightness curve.
pub const SPECTROGRAM_GAMMA: f32 = 0.8; // Threshold for N-normalized, 2x-scaled averaged amplitude, below which it's mapped to black.
pub const SPECTROGRAM_BLACK_THRESHOLD: f32 = 0.005;

pub const SPECTROGRAM_TEXT_COLOR: &RGBColor = &WHITE;
pub const SPECTROGRAM_GRID_COLOR: RGBColor = RGBColor(80, 80, 80);

// src/constants.rs
