// src/constants.rs

// Import specific colors needed
use plotters::style::colors::full_palette::{AMBER, GREEN, LIGHTBLUE, ORANGE, PURPLE, RED};
use plotters::style::RGBColor;

// Plot dimensions.
pub const PLOT_WIDTH: u32 = 2560;
pub const PLOT_HEIGHT: u32 = 1400;

// Font sizes for plots and labels
pub const FONT_SIZE_MAIN_TITLE: i32 = 24; // Main title at top of entire image
pub const FONT_SIZE_CHART_TITLE: i32 = 20; // Individual chart/subplot titles
pub const FONT_SIZE_AXIS_LABEL: i32 = 18; // X and Y axis labels and tick labels (increased to be clearly larger than legend)
pub const FONT_SIZE_LEGEND: i32 = 18; // Legend labels
pub const FONT_SIZE_PEAK_LABEL: i32 = 18; // Peak detection labels on spectrum plots
pub const FONT_SIZE_MESSAGE: i32 = 20; // "Data Unavailable" and other info messages

// Constants for the step response calculation method (mimicking PlasmaTree and PTB PTstepcalc.m)
pub const FRAME_LENGTH_S: f64 = 2.0; // Length of each window in seconds (PTB uses 2s)
pub const RESPONSE_LENGTH_S: f64 = 0.5; // Length of the step response to keep (500ms typical)
pub const SUPERPOSITION_FACTOR: usize = 16; // Number of overlapping windows (can be tuned)
pub const TUKEY_ALPHA: f64 = 1.0; // Alpha for Tukey window (1.0 is Hanning window)

pub const INITIAL_GYRO_SMOOTHING_WINDOW: usize = 15; // Initial Gyro Smoothing (applied before deconvolution)
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

// Motor spectrum analysis constants
pub const MOTOR_OSCILLATION_FREQ_MIN_HZ: f64 = 50.0; // Lower bound for motor oscillation detection (Hz)
pub const MOTOR_OSCILLATION_FREQ_MAX_HZ: f64 = 200.0; // Upper bound for motor oscillation detection (Hz)
pub const MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER: f64 = 3.0; // Peak must be > N× average to flag oscillation (unitless multiplier)
pub const MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD: f64 = 10.0; // Absolute amplitude threshold (normalized linear amplitude units)
pub const MOTOR_SPECTRUM_Y_AXIS_MAX: f64 = 5.0; // Static Y-axis maximum for motor spectrum plots (normalized linear amplitude units)

// Minimum samples required for a meaningful FFT on motor outputs
pub const MIN_FFT_SAMPLES: usize = 128; // Minimum samples for reliable oscillation detection (provides ~2-3 frequency bins in 50-200 Hz range at 8kHz)

// TODO: Verify that MOTOR_OSCILLATION_FREQ_MIN_HZ..MOTOR_OSCILLATION_FREQ_MAX_HZ (50–200 Hz)
// matches the expected motor/prop/ESC oscillation ranges for our target hardware.
// If some target platforms (large props, geared motors, etc.) show relevant oscillations
// outside this range, consider adjusting these bounds or making them configurable.

// Constants for the spectrum plot (linear amplitude)
pub const SPECTRUM_Y_AXIS_FLOOR: f64 = 20000.0; // Maximum amplitude for spectrum plots.
pub const SPECTRUM_NOISE_FLOOR_HZ: f64 = 70.0; // Frequency threshold below which to ignore for dynamic Y-axis scaling (e.g., motor idle noise).
pub const SPECTRUM_Y_AXIS_HEADROOM_FACTOR: f64 = 1.2; // Factor to extend Y-axis above the highest peak (after noise floor) for better visibility.
pub const PEAK_LABEL_MIN_AMPLITUDE: f64 = 1000.0; // Ignore peaks under this; Tunable (gyro spectrums only)

// Intelligent threshold for filtered D-term peak detection
pub const FILTERED_D_TERM_MIN_THRESHOLD: f64 = 100000.0; // Filtered D-term peaks below 100k (0.1% of typical 100M unfiltered) are not meaningful

// Intelligent threshold for filtered gyro peak detection
#[allow(dead_code)]
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
pub const MIN_SECONDARY_PEAK_RATIO: f64 = 0.05; // Secondary peak must be ≥ this linear ratio of the primary peak's amplitude
pub const MIN_SECONDARY_PEAK_DB: f64 = 6.0; // Minimum dB difference for secondary peaks in dB domain (6 dB = 4x power ratio)
pub const MIN_PEAK_SEPARATION_HZ: f64 = 70.0; // Minimum frequency separation between reported peaks on spectrum plots

// Peak label positioning constants
pub const PEAK_LABEL_BOTTOM_MARGIN_PX: i32 = 72; // Pixels above the bottom edge of plot area for peak labels

// Peak label font measurement constants
pub const AVG_CHAR_WIDTH_RATIO: f32 = 0.56; // Monospace character width as ratio of font size (used to subtract trailing space from rusttype advance widths). Tuned for the bundled TTF; must be revisited if font changes.
pub const TRIANGLE_WIDTH_RATIO: f32 = 0.5; // DejaVu Sans Mono triangle ▲ width as ratio of font size. Tuned for the bundled TTF; must be revisited if font changes.
pub const RIGHT_ALIGN_THRESHOLD: f32 = 0.90; // Peaks in rightmost portion of plot area (threshold * 100%) use right-aligned labels

// Constants for advanced peak detection
pub const ENABLE_WINDOW_PEAK_DETECTION: bool = true; // Enable window-based peak detection for more robust analysis
                                                     // Set to false to use the previous 3-point (amp > prev && amp >= next) logic.
pub const PEAK_DETECTION_WINDOW_RADIUS: usize = 3; // Radius W for peak detection window (total window size: 2*W+1 points)

// PIDsum vs PID Error vs Setpoint Plot
pub const COLOR_PIDSUM_MAIN: &RGBColor = &GREEN;
pub const COLOR_PIDERROR_MAIN: &RGBColor = &PURPLE;
pub const COLOR_SETPOINT_MAIN: &RGBColor = &ORANGE;

// Setpoint vs Gyro Plot
pub const COLOR_SETPOINT_VS_GYRO_SP: &RGBColor = &ORANGE;
pub const COLOR_SETPOINT_VS_GYRO_GYRO: &RGBColor = &LIGHTBLUE;

// Setpoint Derivative Plot
pub const COLOR_SETPOINT_DERIVATIVE: &RGBColor = &PURPLE;

// Setpoint Derivative Y-axis scaling and analysis constants
// Based on analysis of 146 flight logs: P95 of all P95 values = 3175 deg/s².
// P95-based scaling provides better visualization for typical flights while preserving outliers.

// Y-axis headroom scale factor for percentile-based scaling.
// Applies to both unified (gyro/setpoint) and setpoint derivative plots.
// P95 * 1.2 provides cushion to avoid clipping high-rate maneuvers while preserving typical flight visualization.
pub const UNIFIED_Y_AXIS_HEADROOM_SCALE: f64 = 1.2;

// Minimum Y-axis scale for setpoint derivative plots (deg/s², symmetric range).
// Recommended minimum based on P95 analysis: 3175 * 1.2 = 3810 deg/s².
pub const SETPOINT_DERIVATIVE_Y_AXIS_MIN: f64 = 3810.0;

// Percentile for robust expansion detection (95th is standard statistical practice).
// Captures 95% of normal data, excludes top 5% (acro/freestyle).
pub const SETPOINT_DERIVATIVE_EXPANSION_PERCENTILE: f64 = 0.95;

// Visual headroom added to final Y-axis range (multiplier: 1.0 + FACTOR).
// 5% avoids tight cramped plots.
pub const SETPOINT_DERIVATIVE_Y_AXIS_HEADROOM_FACTOR: f64 = 0.05;

// Maximum plausible setpoint derivative rate (deg/s²). Filters logging artifacts
// (data gaps, corrupt timestamps). Real-world rates max ~50-80k; >100k is almost
// certainly corrupt data. Derivatives above threshold are excluded with warning.
pub const SETPOINT_DERIVATIVE_OUTLIER_THRESHOLD: f64 = 100_000.0;

// Minimum reasonable time delta (seconds) between consecutive setpoint samples.
// For 8 kHz logs (~125µs); <50µs indicates glitches or missing rows. Excluded to
// prevent spurious spikes from corrupted time stamps.
pub const SETPOINT_DERIVATIVE_MIN_DT: f64 = 0.00005;

// Gyro vs Unfilt Gyro Plot
pub const COLOR_GYRO_VS_UNFILT_FILT: &RGBColor = &LIGHTBLUE;
pub const COLOR_GYRO_VS_UNFILT_UNFILT: &RGBColor = &AMBER;

// P, I, D Term Activity Plot
pub const COLOR_P_TERM: &RGBColor = &RED;
pub const COLOR_I_TERM: &RGBColor = &LIGHTBLUE;
pub const COLOR_D_TERM_ACTIVITY: &RGBColor = &GREEN;

// Minimum Y-axis scale for gyro/setpoint plots (deg/s, symmetric range)
pub const UNIFIED_Y_AXIS_MIN_SCALE: f64 = 200.0;

// Y-axis scaling strategy for gyro/setpoint plots
// Using 95th percentile provides better visualization for typical flight data
// while outliers (crashes, hard landings) still visible but don't compress normal data
// Analysis of 148 flight logs shows P95 is typically only 27% of absolute max
// Headroom scale factor (UNIFIED_Y_AXIS_HEADROOM_SCALE) applied to P95 values
pub const UNIFIED_Y_AXIS_PERCENTILE: f64 = 0.95; // Use 95th percentile for Y-axis scaling

// Minimum Y-axis scale for P, I, D activity plots (symmetric range)
// 200.0 provides good visibility for human interpretation
pub const PID_ACTIVITY_Y_AXIS_MIN: f64 = 200.0;

// D-term Plot Colors (distinct from gyro colors)
pub const COLOR_D_TERM_FILT: &RGBColor = &GREEN; // Use green for filtered D-term (distinct from gyro blue/amber)
pub const COLOR_D_TERM_UNFILT: &RGBColor = &ORANGE; // Use orange for unfiltered D-term (distinct from gyro yellow)

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

// Gyro PSD heatmap scaling constants
pub const GYRO_PSD_HEATMAP_MAX_DB: f64 = -10.0; // Maximum dB for gyro PSD heatmap color scaling

// P:D Ratio Recommendation Constants
pub const PD_RATIO_MIN_CHANGE_THRESHOLD: f64 = 0.05; // Minimum P:D ratio change to warrant a recommendation (5%)

// P:D Ratio adjustment multipliers for different recommendation styles
// Based on control theory and step response analysis:
// - Conservative (~+18% D): Safe incremental improvement, 2-3 iterations to optimal
// - Moderate (~+33% D): Balanced approach for experienced pilots, 1-2 iterations to optimal
// Note: Works for all aircraft sizes including 10"+ where D > P (P:D < 1.0)
pub const PD_RATIO_CONSERVATIVE_MULTIPLIER: f64 = 0.85; // Conservative: reduce P:D by 15% (≈+17.6% D)
pub const PD_RATIO_MODERATE_MULTIPLIER: f64 = 0.75; // Moderate: reduce P:D by 25% (≈+33.3% D)

// Peak range adjustment multipliers for different overshoot levels
// These create a graduated response based on step response quality
pub const PEAK_ACCEPTABLE_MULTIPLIER: f64 = 0.95; // Acceptable (1.05-1.10): Small adjustment, +≈5.3% D
pub const PEAK_MINOR_MULTIPLIER: f64 = 0.92; // Minor overshoot (1.11-1.15): Moderate adjustment, +≈8.7% D
pub const PEAK_MODERATE_MULTIPLIER: f64 = 0.88; // Moderate overshoot (1.16-1.20): Larger adjustment, +≈13.6% D

// Peak range thresholds for step response quality assessment
pub const PEAK_OPTIMAL_MIN: f64 = 0.95; // Optimal response: 0.95-1.04 (0-5% overshoot/undershoot)
#[allow(dead_code)]
pub const PEAK_OPTIMAL_MAX: f64 = 1.04;
pub const PEAK_ACCEPTABLE_MIN: f64 = 1.05; // Acceptable: 1.05-1.10 (5-10% overshoot)
pub const PEAK_ACCEPTABLE_MAX: f64 = 1.10;
#[allow(dead_code)]
pub const PEAK_MINOR_MIN: f64 = 1.11; // Minor overshoot: 1.11-1.15 (11-15% overshoot)
pub const PEAK_MINOR_MAX: f64 = 1.15;
pub const PEAK_MODERATE_MIN: f64 = 1.16; // Moderate overshoot: 1.16-1.20 (16-20% overshoot)
#[allow(dead_code)]
pub const PEAK_MODERATE_MAX: f64 = 1.20;
pub const PEAK_SIGNIFICANT_MIN: f64 = 1.20; // Significant overshoot: >1.20 (>20% overshoot)

// Sanity check limits for P:D ratio recommendations
// Note: MIN_REASONABLE_PD_RATIO of 0.3 accommodates large aircraft where D > P
pub const MIN_REASONABLE_PD_RATIO: f64 = 0.3; // Don't recommend D > 3.3× P (was 0.5, adjusted for 10"+ aircraft)
pub const MAX_REASONABLE_PD_RATIO: f64 = 3.0; // Don't recommend D < P/3
pub const SEVERE_OVERSHOOT_THRESHOLD: f64 = 1.5; // Peak > 1.5 suggests deeper issues than just D tuning

// Bode analysis and transfer function estimation constants
pub const COHERENCE_HIGH_THRESHOLD: f64 = 0.7; // High confidence threshold for stability margins
pub const COHERENCE_MEDIUM_THRESHOLD: f64 = 0.4; // Medium confidence threshold for stability margins
pub const FREQUENCY_EPSILON: f64 = 1e-12; // Guard against division by zero for frequency differences
pub const VALUE_EPSILON: f64 = 1e-12; // Guard against division by zero for value (magnitude/phase) differences
pub const PSD_EPSILON: f64 = 1e-12; // Guard against division by zero for PSD values

// Bode plot margin constants
pub const MAGNITUDE_PLOT_MARGIN_DB: f64 = 10.0; // Padding above/below magnitude data for plot range
pub const PHASE_PLOT_MARGIN_DEG: f64 = 30.0; // Padding above/below phase data for plot range

// Optimal P Estimation Constants
// Frame-class-aware Td (time to 50%) targets in milliseconds
// Provisional estimates based on torque-to-rotational-inertia scaling: Td ∝ 1/(mass × radius²)
// TODO: Validate with bench tests and actual flight data across all frame classes

/// Td target specification for a frame class
#[derive(Debug, Clone, Copy)]
pub struct TdTargetSpec {
    pub target_ms: f64,
    pub tolerance_ms: f64,
    #[allow(dead_code)] // Will be used in Phase 2 physics-based calculations
    pub typical_weight_g: f64,
}

impl TdTargetSpec {
    /// Create a new TdTargetSpec with automatic 25% tolerance calculation and typical weight
    #[allow(dead_code)] // Will be used in Phase 2 physics-based calculations
    pub const fn new(target_ms: f64, typical_weight_g: f64) -> Self {
        Self {
            target_ms,
            tolerance_ms: target_ms * 0.25,
            typical_weight_g,
        }
    }

    /// Create without typical weight (for existing empirical targets)
    pub const fn new_simple(target_ms: f64) -> Self {
        Self {
            target_ms,
            tolerance_ms: target_ms * 0.25,
            typical_weight_g: 0.0, // Not used for empirical targets
        }
    }

    /// Get TdTargetSpec for a given frame size in inches (1-15)
    /// Returns None if the size is out of valid range
    pub fn for_frame_inches(inches: usize) -> Option<&'static TdTargetSpec> {
        if (1..=15).contains(&inches) {
            Some(&TD_TARGETS[inches - 1])
        } else {
            None
        }
    }
}

/// Td targets for all frame classes (1" through 15")
/// Index: 0=1", 1=2", ..., 14=15"
pub const TD_TARGETS: [TdTargetSpec; 15] = [
    TdTargetSpec::new_simple(40.0),  // 1" tiny whoop (30-50ms)
    TdTargetSpec::new_simple(35.0),  // 2" micro (26-44ms)
    TdTargetSpec::new_simple(30.0),  // 3" toothpick/cinewhoop (23-38ms)
    TdTargetSpec::new_simple(25.0),  // 4" racing (19-31ms)
    TdTargetSpec::new_simple(20.0),  // 5" freestyle/racing (15-25ms, common baseline)
    TdTargetSpec::new_simple(28.0),  // 6" long-range (21-35ms)
    TdTargetSpec::new_simple(37.5),  // 7" long-range (28-47ms)
    TdTargetSpec::new_simple(47.0),  // 8" long-range (35-59ms)
    TdTargetSpec::new_simple(56.0),  // 9" cinelifter (42-70ms)
    TdTargetSpec::new_simple(65.0),  // 10" cinelifter (49-81ms)
    TdTargetSpec::new_simple(75.0),  // 11" heavy-lift (56-94ms)
    TdTargetSpec::new_simple(85.0),  // 12" heavy-lift (64-106ms)
    TdTargetSpec::new_simple(95.0),  // 13" heavy-lift (71-119ms)
    TdTargetSpec::new_simple(105.0), // 14" heavy-lift (79-131ms)
    TdTargetSpec::new_simple(115.0), // 15" heavy-lift (86-144ms)
];

// High-frequency noise analysis for P headroom estimation
// D-term energy above this frequency threshold indicates noise constraints
pub const DTERM_HF_CUTOFF_HZ: f64 = 200.0; // Frequency above which high-frequency noise is measured
pub const DTERM_HF_ENERGY_THRESHOLD: f64 = 0.15; // 15% of total D-term energy (high noise level)
pub const DTERM_HF_ENERGY_MODERATE: f64 = 0.10; // 10% of total D-term energy (moderate noise level)

// Response consistency quality control
// Ensures Td measurements are reliable across multiple step responses
pub const TD_CONSISTENCY_MIN_THRESHOLD: f64 = 0.85; // 85% of responses should be within ±1 std dev
pub const TD_COEFFICIENT_OF_VARIATION_MAX: f64 = 0.20; // 20% CV (std/mean) is acceptable

// P headroom estimation multipliers
// Conservative approach for users who want safe incremental improvements
pub const P_HEADROOM_CONSERVATIVE_MULTIPLIER: f64 = 1.05; // +5% from current P
                                                          // Moderate approach for experienced pilots
pub const P_HEADROOM_MODERATE_MULTIPLIER: f64 = 1.10; // +10% from current P
                                                      // Aggressive approach for optimization (use with caution)
#[allow(dead_code)]
pub const P_HEADROOM_AGGRESSIVE_MULTIPLIER: f64 = 1.15; // +15% from current P (reserved for future use)

// P reduction multipliers (when Td is too fast or noise is too high)
pub const P_REDUCTION_MODERATE_MULTIPLIER: f64 = 0.95; // -5% from current P
#[allow(dead_code)]
pub const P_REDUCTION_AGGRESSIVE_MULTIPLIER: f64 = 0.90; // -10% from current P

// Td statistics computation constants
pub const TD_MEAN_EPSILON: f64 = 1e-12; // Threshold for near-zero mean values (avoid division by zero)
pub const TD_SAMPLES_MIN_FOR_STDDEV: usize = 2; // Minimum samples needed for std dev calculation

// Td deviation thresholds (percentage deviation from target)
pub const TD_DEVIATION_SIGNIFICANTLY_SLOWER_THRESHOLD: f64 = 30.0; // > 30% slower
pub const TD_DEVIATION_MODERATELY_SLOWER_THRESHOLD: f64 = 15.0; // > 15% slower
pub const TD_DEVIATION_SIGNIFICANTLY_FASTER_THRESHOLD: f64 = -15.0; // < -15% faster

// Optimal P estimation data collection thresholds
pub const OPTIMAL_P_SECONDS_TO_MS_MULTIPLIER: f64 = 1000.0; // Convert seconds to milliseconds
pub const OPTIMAL_P_MIN_DTERM_SAMPLES: usize = 100; // Minimum D-term samples for noise analysis

// Mass distribution constants for physics-based calculations
// NOTE: Pilot provides total weight via --weight (scale reading in grams).
// In physics: weight = mass × gravity (force), but scale readings represent mass.
// We distribute total mass across components for rotational inertia I = Σ(m·r²).
pub const MASS_FRACTION_MOTORS: f64 = 0.16; // Motors: ~16% of total (4 motors, ~40g each on 5")
pub const MASS_FRACTION_PROPS: f64 = 0.02; // Props: ~2% of total (4 props, ~5g each on 5")
pub const MASS_FRACTION_FRAME: f64 = 0.12; // Frame: ~12% of total (carbon arms, hardware)
pub const MASS_FRACTION_BATTERY: f64 = 0.30; // Battery: ~30% of total (most concentrated mass)
pub const _MASS_FRACTION_CENTRAL: f64 = 0.10; // Central electronics: ~10% (FC, ESCs, VTX, camera)
pub const _MASS_FRACTION_MISC: f64 = 0.30; // Remaining: ~30% (wiring, props screws, straps, etc.)
                                           // Total should equal 1.0 (validated at compile time via assertion in code)

// Battery offset from center for COG balance (typically rear-mounted)
pub const BATTERY_OFFSET_FROM_CENTER_MM: f64 = 30.0; // 30mm behind FC center
