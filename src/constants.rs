// src/constants.rs

// Import specific colors needed
use plotters::style::{RGBColor}; // Removed unused TRANSPARENT
use plotters::style::colors::full_palette::{GREEN, AMBER, ORANGE, LIGHTBLUE, RED, PURPLE, BLACK, WHITE, YELLOW};


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

// --- Spectrogram Constants ---
pub const SPECTROGRAM_THROTTLE_BINS: usize = 100;
pub const SPECTROGRAM_FFT_WINDOW_SIZE_TARGET: usize = 128;
pub const SPECTROGRAM_MAX_FREQ_HZ: f32 = 1000.0;
pub const SPECTROGRAM_POWER_CLIP_MAX: f32 = 5000.0; // Adjusted this based on potential peak values
// Revised "hot" colormap with 16 points for smoother/more detailed gradient
// Values are normalized power (0.0 to 1.0)
pub const SPECTROGRAM_COLOR_SCALE: [(f32, RGBColor); 16] = [
    (0.0/15.0, BLACK),                     // 1. Black
    (1.0/15.0, RGBColor(32, 0, 0)),       // 2. Very Dark Red
    (2.0/15.0, RGBColor(64, 0, 0)),       // 3. Dark Red
    (3.0/15.0, RGBColor(96, 0, 0)),       // 4.
    (4.0/15.0, RGBColor(128, 0, 0)),      // 5. Medium Dark Red
    (5.0/15.0, RGBColor(160, 0, 0)),      // 6.
    (6.0/15.0, RGBColor(192, 0, 0)),      // 7. Brighter Dark Red
    (7.0/15.0, RED),                       // 8. Red (220,0,0)
    (8.0/15.0, RGBColor(230, 55, 0)),     // 9. Red-Orange
    (9.0/15.0, RGBColor(242, 110, 0)),    // 10. Orange-Red
    (10.0/15.0, ORANGE),                   // 11. Orange (255,165,0)
    (11.0/15.0, RGBColor(255, 195, 0)),    // 12. Light Orange / Orange-Yellow
    (12.0/15.0, RGBColor(255, 225, 0)),    // 13. Yellow-Orange
    (13.0/15.0, YELLOW),                   // 14. Yellow (255,255,0)
    (14.0/15.0, RGBColor(255, 255, 128)),  // 15. Pale Yellow
    (15.0/15.0, WHITE),                    // 16. White
];
pub const SPECTROGRAM_TEXT_COLOR: &RGBColor = &WHITE; // THIS LINE IS IMPORTANT
pub const SPECTROGRAM_GRID_COLOR: RGBColor = RGBColor(80,80,80);

// src/constants.rs