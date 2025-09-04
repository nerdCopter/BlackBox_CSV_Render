// src/constants.rs

// Import specific colors needed
use plotters::style::colors::full_palette::{AMBER, GREEN, LIGHTBLUE, ORANGE, PURPLE, RED};
use plotters::style::RGBColor;

// Plot dimensions.
pub const PLOT_WIDTH: u32 = 1920;
pub const PLOT_HEIGHT: u32 = 1080;

// Constants for the step response calculation method (mimicking PlasmaTree and PTB PTstepcalc.m)
pub const FRAME_LENGTH_S: f64 = 2.0; // Length of each window in seconds (PTB uses 2s)
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep (500ms typical)
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows (can be tuned)
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)

pub const INITIAL_GYRO_SMOOTHING_WINDOW: usize = 15; // // Initial Gyro Smoothing (applied before deconvolution)
pub const POST_AVERAGING_SMOOTHING_WINDOW: usize = 15; // Constant for post-averaging smoothing of the final step response curves.

pub const APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION: bool = true; // Individual Response "Y-Correction" (Normalization before averaging)
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

// Steady-state definition for Y-correction and QC (matches PTB PTstepcalc.m: 200ms to end of response)
pub const STEADY_STATE_START_S: f64 = 0.2; // Start time for steady-state check (200ms into the response)
pub const STEADY_STATE_END_S: f64 = RESPONSE_LENGTH_S; // End time for steady-state check (to the end of RESPONSE_LENGTH_S); can be decoupled if desired, but do NOT exceed RESPONSE_LENGTH_S

pub const FINAL_NORMALIZED_STEADY_STATE_TOLERANCE: f64 = 0.15; // Final tolerance for normalized steady-state mean in step response plot
pub const DEFAULT_SETPOINT_THRESHOLD: f64 = 500.0; // Default setpoint threshold, can be overridden at runtime for categorizing responses

// Constants for filtering data based on movement and flight phase.
pub const MOVEMENT_THRESHOLD_DEG_S: f64 = 20.0; // Minimum setpoint/gyro magnitude (from PTB/PlasmaTree)
pub const EXCLUDE_START_S: f64 = 3.0; // Exclude seconds from the start of the log
pub const EXCLUDE_END_S: f64 = 3.0; // Exclude seconds from the end of the log

// Constants for the spectrum plot (linear amplitude)
pub const SPECTRUM_Y_AXIS_FLOOR: f64 = 20000.0; // Maximum amplitude for spectrum plots.
pub const SPECTRUM_NOISE_FLOOR_HZ: f64 = 70.0; // Frequency threshold below which to ignore for dynamic Y-axis scaling (e.g., motor idle noise).
pub const SPECTRUM_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2; // Factor to extend Y-axis above the highest peak (after noise floor) for better visibility.
pub const PEAK_LABEL_MIN_AMPLITUDE: f64 = 1000.0; // Ignore peaks under this; Tunable (gyro spectrums only)

// Intelligent threshold for filtered D-term peak detection
pub const FILTERED_D_TERM_MIN_THRESHOLD: f64 = 100000.0; // Filtered D-term peaks below 100k (0.1% of typical 100M unfiltered) are not meaningful

// Intelligent threshold for filtered gyro peak detection
pub const FILTERED_GYRO_MIN_THRESHOLD: f64 = 2000.0; // Filtered gyro peaks below 2k are typically noise (based on user feedback)

// Constants for PSD plots (dB scale)
pub const PSD_Y_AXIS_FLOOR_DB: f64 = -80.0; // A reasonable floor for PSD values in dB
pub const PSD_Y_AXIS_HEADROOM_FACTOR_DB: f64 = 10.0; // Factor to extend Y-axis above the highest peak (in dB, e.g., 10 dB headroom)
pub const PSD_PEAK_LABEL_MIN_VALUE_DB: f64 = -60.0; // Minimum PSD value in dB for a peak to be labeled. (Tune as needed)

// Constants for Spectrogram/Heatmap plots
pub const STFT_WINDOW_DURATION_S: f64 = 0.1; // Duration of each STFT window in seconds
pub const STFT_OVERLAP_FACTOR: f64 = 0.75; // Overlap between windows (e.g., 0.75 for 75% overlap)
pub const HEATMAP_MIN_PSD_DB: f64 = -80.0; // Minimum PSD value in dB for heatmap color scaling

// Constants for Throttle-Frequency Heatmap
pub const THROTTLE_Y_BINS_COUNT: usize = 50; // Number of bins for the throttle (Y) axis
pub const THROTTLE_Y_MIN_VALUE: f64 = 0.0; // Minimum throttle value for plotting range
pub const THROTTLE_Y_MAX_VALUE: f64 = 1000.0; // Maximum throttle value for plotting range

// Constants for spectrum peak labeling
pub const MAX_PEAKS_TO_LABEL: usize = 3; // Max number of peaks (including primary) to label on spectrum plots
pub const MIN_SECONDARY_PEAK_RATIO: f64 = 0.05; // Secondary peak must be ≥ this linear ratio of the primary peak’s amplitude
pub const MIN_SECONDARY_PEAK_DB: f64 = 6.0; // Minimum dB difference for secondary peaks in dB domain (6 dB = 4x power ratio)
pub const MIN_PEAK_SEPARATION_HZ: f64 = 70.0; // Minimum frequency separation between reported peaks on spectrum plots

// Constants for advanced peak detection
pub const ENABLE_WINDOW_PEAK_DETECTION: bool = true; // Set to true to use window-based peak detection
                                                     // Set to false to use the previous 3-point (amp > prev && amp >= next) logic.
pub const PEAK_DETECTION_WINDOW_RADIUS: usize = 3; // Radius W for peak detection window (total 2*W+1 points).

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

// D-term Plot Colors (distinct from gyro colors)
pub const COLOR_D_TERM_FILT: &RGBColor = &LIGHTBLUE; // Keep blue for filtered (consistent)
pub const COLOR_D_TERM_UNFILT: &RGBColor = &ORANGE; // Use orange for unfiltered (better contrast than yellow)

// Step Response Plot
pub const COLOR_STEP_RESPONSE_LOW_SP: &RGBColor = &LIGHTBLUE;
pub const COLOR_STEP_RESPONSE_HIGH_SP: &RGBColor = &ORANGE;
pub const COLOR_STEP_RESPONSE_COMBINED: &RGBColor = &RED;

// Stroke widths for lines
pub const LINE_WIDTH_PLOT: u32 = 1; // Width for plot lines
pub const LINE_WIDTH_LEGEND: u32 = 2; // Width for legend lines

// Filtering delay analysis thresholds and parameters
#[allow(dead_code)] // Used in enhanced cross-correlation analysis
pub const MIN_CORRELATION_THRESHOLD: f32 = 0.3; // Minimum correlation for reliable delay
pub const FALLBACK_CORRELATION_THRESHOLD: f32 = 0.2; // Lower threshold for fallback
pub const MAX_DELAY_FRACTION: usize = 10; // Search up to 1/10 of signal length
pub const MAX_DELAY_SAMPLES: usize = 200; // Maximum delay samples to search
pub const MIN_SAMPLES_FOR_DELAY: usize = 100; // Minimum samples required for delay analysis

// D-term specific analysis constants
pub const MIN_D_TERM_SAMPLES_FOR_ANALYSIS: usize = 100; // Minimum D-term samples needed for meaningful analysis
pub const D_TERM_MIN_THRESHOLD: f32 = 1e-6; // Very small threshold for detecting "effectively zero" D-terms
pub const D_TERM_MIN_STD_DEV: f32 = 1e-6; // Minimum standard deviation for meaningful D-term variation
pub const D_TERM_CORRELATION_THRESHOLD: f64 = 0.1; // More lenient threshold for D-term cross-correlation
pub const MIN_SAMPLES_FOR_D_TERM_CORR: usize = 50; // Lower sample requirement for D-term correlation

// src/constants.rs
