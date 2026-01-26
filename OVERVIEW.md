## Code Overview and Step Response Calculation

## Table of Contents
- [Code Overview and Step Response Calculation](#code-overview-and-step-response-calculation)
  - [Configuration](#configuration)
  - [Core Functionality](#core-functionality)
  - [Filtering Delay Calculation](#filtering-delay-calculation)
  - [Enhanced Cross-Correlation Method (Primary Implementation)](#enhanced-cross-correlation-method-primary-implementation)
  - [Implementation Details](#implementation-details)
  - [Filter Response Curves](#filter-response-curves)
  - [Bode Plot Analysis (Optional)](#bode-plot-analysis-optional)
  - [Step-Response Comparison with Other Analysis Tools](#step-response-comparison-with-other-analysis-tools)
    - [Compared to PIDtoolbox/Matlab (PTstepcalc.m)](#compared-to-pidtoolboxmatlab-ptstepcalcm)
    - [Compared to PlasmaTree/Python (PID-Analyzer.py)](#compared-to-plasmatreepython-pid-analyzerpy)

The Rust program processes Betaflight Blackbox CSV logs to generate various plots. Here's a concise overview:

### Configuration

All analysis parameters, thresholds, plot dimensions, and algorithmic constants are centrally defined in `src/constants.rs`, making the implementation highly configurable for different analysis needs and flight controller characteristics.

### Core Functionality

1.  **Argument Parsing (`src/main.rs`):**
    * Parses command-line arguments: input CSV file(s), an optional `--dps` parameter (requires a numeric threshold value for detailed step response plots with low/high split), an optional `--output-dir` for specifying the output directory, and an optional `--step` flag to generate only step response plots.
    * Additional options include `--help` and `--version` for user assistance.
    * The `--output-dir` parameter now requires a directory path when specified. If omitted, plots are saved in the source folder (input file's directory).
    * Handles multiple input files and determines if a directory prefix should be added to output filenames to avoid collisions when processing files from different directories.

2.  **File Processing (`src/main.rs:process_file`):**
    * For each input CSV:
        * **Path Setup:** Determines input and output paths.
        * **Data Parsing (`src/data_input/log_parser.rs:parse_log_file`):** Reads the CSV, extracts log data (time, gyro, setpoint, F-term, etc.), sample rate, and checks for the presence of essential data headers.
        * **Step Response Data Preparation:**
            * Filters log data, excluding the start and end seconds (constants `EXCLUDE_START_S`, `EXCLUDE_END_S` from `src/constants.rs`) to create contiguous segments of time, setpoint, and gyro data for each axis. This data is stored in `contiguous_sr_input_data`.
        * **Step Response Calculation (`src/data_analysis/calc_step_response.rs:calculate_step_response`):**
            * This is the core of the step response analysis. It implements **non-parametric system identification** using Wiener deconvolution rather than traditional first-order or second-order curve fitting. This approach directly extracts the system's actual step response without assuming a specific mathematical model, allowing it to capture complex, higher-order dynamics and non-linearities.
            * For each axis (Roll, Pitch, Yaw):
                * It takes the prepared time, setpoint, and (optionally smoothed via `INITIAL_GYRO_SMOOTHING_WINDOW`) gyro data arrays and the sample rate.
                * **Windowing:** The input signals (setpoint and gyro) are segmented into overlapping windows (`winstacker_contiguous`) of `FRAME_LENGTH_S` duration. A Tukey window (`tukeywin` with `TUKEY_ALPHA`) is applied to each segment to reduce spectral leakage.
                * **Movement Threshold:** Windows are discarded if the maximum absolute setpoint value within them is below `MOVEMENT_THRESHOLD_DEG_S`.
                * **Deconvolution:** For each valid window, Wiener deconvolution (`wiener_deconvolution_window`) is performed between the windowed setpoint (input) and gyro (output) signals in the frequency domain. This estimates the impulse response of the system. A regularization term (`0.0001`) helps stabilize the deconvolution.
                * **Impulse to Step Response:** The resulting impulse response is converted to a step response by cumulative summation (`cumulative_sum`). This step response is then truncated to `RESPONSE_LENGTH_S`.
                * **Y-Correction (Normalization):**
                    * If `APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION` is true, each individual step response window is normalized. The mean of its steady-state portion (defined by `STEADY_STATE_START_S` and `STEADY_STATE_END_S`) is calculated. If this mean is significant (abs > `Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS`), the entire response window is divided by this mean, aiming to make its steady-state value approach 1.0.
                * **Quality Control (QC):**
                    * Each (potentially Y-corrected) step response window undergoes QC. The minimum and maximum values of its steady-state portion are checked against `NORMALIZED_STEADY_STATE_MIN_VAL` and `NORMALIZED_STEADY_STATE_MAX_VAL`.
                    * Optionally (`ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK`), the mean of the steady-state portion is also checked against `NORMALIZED_STEADY_STATE_MEAN_MIN` and `NORMALIZED_STEADY_STATE_MEAN_MAX`.
                    * Only windows passing QC are kept.
                * The function returns:
                    1.  A time vector for the step response plot (`response_time`).
                    2.  A 2D array (`valid_stacked_responses`) containing all step response windows that passed QC.
                    3.  A 1D array (`valid_window_max_setpoints`) of the maximum setpoint values for each corresponding valid window.
        * **Plot Generation (various functions in `src/plot_functions/`):**
            * `plot_step_response` (in `src/plot_functions/plot_step_response.rs`): Takes the results from `calculate_step_response`.
                * Separates the QC'd responses into "low" and "high" setpoint groups based on the `setpoint_threshold` if the `--dps` parameter (and thus `show_legend`) is provided.
                * **Averaging & Final Normalization (`plot_functions::plot_step_response::process_response`):**
                    * The QC'd responses (either low, high, or combined) are averaged using `calc_step_response::average_responses`.
                    * The averaged response is smoothed using a moving average (`calc_step_response::moving_average_smooth_f64`) with `POST_AVERAGING_SMOOTHING_WINDOW`.
                    * The smoothed response is shifted to start at 0.0.
                    * A **final normalization** step is performed: the mean of the steady-state portion of this *averaged, smoothed, and shifted* response is calculated. The entire response is then divided by this mean to ensure the plotted average response aims for a steady-state of 1.0.
                    * The final response is only plotted if its steady-state mean (after this final normalization) is within `FINAL_NORMALIZED_STEADY_STATE_TOLERANCE` of 1.0.
                * Calculates peak value and delay time (Td) for each plotted average response using `calc_step_response::find_peak_value` and `calc_step_response::calculate_delay_time`.
                    * **Delay Time (Td) Calculation:** The delay time is calculated as the time for the step response to reach 50% of its final value. Linear interpolation is used for precise determination of the 50% threshold crossing. A fixed offset of -1ms is applied to the calculated time (in milliseconds) before converting to seconds, and the result is constrained to be non-negative.
                * Generates and saves the step response plot.
        * **P:D Ratio Recommendations (`src/main.rs`):**
            * Based on step response peak analysis, the system provides tuning recommendations:
                * **Important:** Peak value is measured as the first maximum after the response crosses the setpoint (1.0). The initial transient dip (visible in first ~30ms) is normal system behavior and not used for tuning recommendations.
                * **Peak Analysis Ranges (based on step response overshoot/undershoot):**
                    * Peak > 1.20: Significant overshoot (>20%) → P:D×0.85 (increase D by ~18%)
                    * Peak 1.16-1.20: Moderate overshoot (16-20%) → P:D×0.88 (increase D by ~14%)
                    * Peak 1.11-1.15: Minor overshoot (11-15%) → P:D×0.92 (increase D by ~9%)
                    * Peak 1.05-1.10: Acceptable response (5-10% overshoot) → P:D×0.95 (increase D by ~5%)
                    * Peak 0.95-1.04: Optimal response (0-5% overshoot/undershoot) → No change (ideal damping)
                    * Peak 0.85-0.94: Minor undershoot (6-15%) → P:D×1.05 (decrease D by ~5%)
                    * Peak < 0.85: Significant undershoot (>15%) → P:D×1.15 (decrease D by ~13%)
                * **Dual Recommendations:**
                    * **Conservative** (PD_RATIO_CONSERVATIVE_MULTIPLIER = 0.85): Reduces P:D ratio by 15%, increasing D by ~18%. Safe for most pilots, 2-3 iterations to optimal.
                    * **Moderate** (PD_RATIO_MODERATE_MULTIPLIER = 0.75): Reduces P:D ratio by 25%, increasing D by ~33%. For experienced pilots, 1-2 iterations to optimal.
                * **D-Min/D-Max Support:**
                    * Automatically detects if D-Min/D-Max system is enabled (Betaflight 4.0+)
                    * When enabled: Recommends proportional D-Min and D-Max values maintaining current ratio relationships
                    * When disabled: Recommends only base D value
                    * Handles firmware differences (BF <4.6: D=max, D-Min=min; BF 4.6+: D=min, D-Max=max)
                * **Safety Features:**
                    * Warnings for severe overshoot (Peak > 1.5) suggesting mechanical issues
                    * Warnings for unreasonable P:D ratios (< 0.3 or > 3.0)
                    * Only displays recommendations when change exceeds PD_RATIO_MIN_CHANGE_THRESHOLD (5%)
                    * Shows recommendations for improvable responses (acceptable/minor/moderate/significant overshoot)
                    * Clear disclaimers that recommendations are starting points, not absolute values
                    * Works for all aircraft sizes including 10"+ where D > P (P:D < 1.0)
                * Recommendations appear in both console output and step response plot legends
            * **Other plots generated (`src/plot_functions/`):** When the `--step` flag is not used, the following additional plots are generated. These per-plot gates are controlled by the `PlotConfig` struct (defaults to all enabled; `PlotConfig::step_only()` when `--step` is specified):
                * `plot_pidsum_error_setpoint`: PIDsum (P+I+D), PID Error (Setpoint - GyroADC), and Setpoint time-domain traces for each axis.
                * `plot_setpoint_vs_gyro`: Setpoint and filtered gyro time-domain comparison for each axis.
                * `plot_gyro_vs_unfilt`: Filtered vs. unfiltered gyro time-domain comparison for each axis. Includes enhanced cross-correlation filtering delay calculation.
                * `plot_gyro_spectrums`: Frequency-domain amplitude spectrums of filtered and unfiltered gyro data with intelligent peak detection and labeling using scale-aware thresholds (`FILTERED_GYRO_MIN_THRESHOLD` for filtered gyro data). Includes enhanced cross-correlation filtering delay calculation and flight firmware filter response curve overlays.
                * `plot_psd`: Power Spectral Density plots in dB scale with peak labeling. Includes enhanced cross-correlation filtering delay calculation.
                * `plot_d_term_spectrums`: Frequency-domain amplitude spectrums of D-term data with intelligent peak detection using scale-aware thresholds (`FILTERED_D_TERM_MIN_THRESHOLD` for filtered D-term data). Includes enhanced cross-correlation filtering delay calculation with intelligent D-term activity detection (skips axes where D gain = 0).
                * `plot_d_term_psd`: Power Spectral Density plots of D-term data in dB scale with intelligent threshold filtering (`PSD_PEAK_LABEL_MIN_VALUE_DB` for filtered data) and enhanced formatting. Includes enhanced cross-correlation filtering delay calculation with intelligent D-term activity detection (skips axes where D gain = 0).
                * `plot_d_term_heatmap`: D-term throttle-frequency heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze D-term energy distribution across different throttle levels.
                * `plot_psd_db_heatmap`: Spectrograms showing PSD vs. time as heatmaps using Short-Time Fourier Transform (STFT) with configurable window duration and overlap.
                * `plot_throttle_freq_heatmap`: Heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze noise characteristics across different throttle levels.

### Filtering Delay Calculation

***

### Enhanced Cross-Correlation Method (Primary Implementation)

* **Algorithm:** For each axis (Roll, Pitch, Yaw), calculates normalized cross-correlation between filtered (`gyroADC`) and unfiltered (`gyroUnfilt`) gyro signals at different time delays using double-precision (f64) arithmetic throughout.
* **Delay Detection:** Identifies the delay that produces the highest correlation coefficient and converts from samples to milliseconds using the sample rate.
* **Subsample Precision:** Uses parabolic interpolation around the peak correlation to achieve subsample delay accuracy, addressing precision limitations of basic sample-rate resolution.
* **Quality Control:** Requires correlation coefficients above configurable thresholds (`MIN_CORRELATION_THRESHOLD`, `FALLBACK_CORRELATION_THRESHOLD`) with fallback mechanisms for challenging signal conditions.
* **Error Handling:** Provides detailed error reporting (`DelayCalculationError`) for insufficient data, low correlation, and signal mismatches.
* **Precision Consistency:** All correlation calculations use f64 precision throughout for maximum accuracy.

***

### Implementation Details

* **Data Validation:** Performs comprehensive data availability diagnostics across all axes before analysis.
* **Averaging:** Individual axis delays are averaged to provide an overall system delay measurement when sufficient correlation is achieved.
* **Bounds Checking:** Comprehensive bounds checking with `saturating_sub()` and explicit runtime verification prevents array access violations. Limits maximum delay search range (`MAX_DELAY_FRACTION`, `MAX_DELAY_SAMPLES`) to prevent unrealistic results and ensures robust parabolic interpolation.
* **Confidence Value Clamping:** Confidence values are clamped to the valid range [0, 1] to handle numerical noise in correlation calculations that could cause values to slightly exceed 1.0.
* **Configurable Thresholds:** All correlation thresholds and delay search parameters are defined as named constants in `src/constants.rs` for maintainability and tuning.
* **Display:** Results are shown in console output with confidence metrics (as percentages), and estimates are integrated into plot legends as "Delay: X.Xms(c:XX%)" for `_GyroVsUnfilt_stacked.png`, `_Gyro_Spectrums_comparative.png`, and `_Gyro_PSD_comparative.png` outputs.

### Filter Response Curves

* **Flight Firmware Integration:** Automatically detects and parses filter configurations from Betaflight, EmuFlight, and INAV blackbox headers including filter types (PT1, PT2, PT3, PT4, BIQUAD), cutoff frequencies, and dynamic filter ranges.
* **Butterworth Correction Visualization (Optional):** When enabled with the `--butterworth` command-line parameter, displays per-stage PT1 implementation details for PT2/PT3/PT4 filters on gyro and D-term spectrum plots:
    * PT2/PT3/PT4 filters are implemented as cascaded PT1 stages with Butterworth correction factors (PT2: 1.554×, PT3: 1.961×, PT4: 2.299×)
    * Main curves (colored) show user-configured filter response (e.g., "LPF1 (PT2 @ 90Hz)")
    * Per-stage curves (gray, prefixed with ≈) show actual PT1 implementation (e.g., "≈ LPF1 (Two PT1 @ 140Hz per-stage)")
    * Vertical cutoff lines: solid for configured cutoff, dotted for per-stage cutoff
    * Formula: per-stage cutoff = configured cutoff × correction factor
    * Applies to all firmware (Betaflight, EmuFlight, INAV) using PT2/PT3/PT4 filters
    * IMUF configurations include version information when available (e.g., "IMUF v256")
* **Gyro Rate Detection:** Comprehensive parsing of gyro sampling rates from various header formats (`gyroSampleRateHz`, `looptime`, `gyro_sync_denom`) with case-insensitive matching and proper division-based denominator calculation.
* **Mathematical Implementation:**
    * **PT1 (1st order)**: $H(s) = 1/(1 + s/\omega_c)$ - Standard single-pole lowpass
    * **PT2 (2nd order)**: $H(s) = 1/(1 + \sqrt{2}\cdot s/\omega_c + (s/\omega_c)^2)$ - Butterworth response yielding -3dB at cutoff
    * **PT3 (3rd order)**: $|H(j\omega)| = 1/\sqrt{1 + (\omega/\omega_c)^6}$ - Simplified 3rd order approximation maintaining -3dB at cutoff
    * **PT4 (4th order)**: $|H(j\omega)| = 1/\sqrt{1 + (\omega/\omega_c)^8}$ - Simplified 4th order approximation maintaining -3dB at cutoff
    * **BIQUAD (2nd order)**: Currently implemented as PT2 Butterworth response (Q=0.707). Ready for Q-factor enhancement with $H(s) = \omega_0^2/(s^2 + (\omega_0/Q)\cdot s + \omega_0^2)$ where $\omega_0 = 2\pi\cdot f_c$
    * Note: $\omega_c$ represents angular cutoff frequency ($\omega_c = 2\pi\cdot f_c$ where $f_c$ is cutoff in Hz)
* **Curve Generation:** Logarithmic frequency spacing from 10% of cutoff frequency to gyro Nyquist frequency (gyro\_rate/2) with 1000 points for smooth visualization. Includes division-by-zero protection and edge case handling.
* **Visualization Integration:** Filter response curves are overlaid on spectrum plots (`plot_gyro_spectrums`) as red curves with clear legends showing filter type and cutoff frequency, enhancing spectrum analysis with theoretical filter characteristics.
* **Quality Assurance:** Comprehensive unit tests verify -3dB magnitude response at cutoff frequencies for all filter types and validate gyro rate extraction accuracy.

### Bode Plot Analysis (Optional)

* **Purpose:** Transfer function visualization for system identification via magnitude, phase, and coherence plots. Reveals frequency-domain behavior and control loop characteristics.
* **Activation:** Disabled by default; enable with `--bode` flag (requires explicit user action).
* **Recommended Use:** Controlled test flights with system-identification inputs (chirp/PRBS) on a tethered or otherwise secured aircraft. Provides reliable transfer function estimates with high coherence (γ² ≥ 0.6).
* **Limitations:** Normal operational flight logs produce low coherence due to nonlinearities, closed-loop feedback, and nonstationary maneuvers. Results in such cases are unreliable and not recommended for tuning decisions.
* **Warning:** A runtime warning is displayed when `--bode` is used to inform users of these requirements and recommend spectrum analysis for normal flights.

### Output and Tuning Recommendations

#### Generated PNG Plots

When `--step` flag is not used, all plots below are generated:

- **`*_Step_Response_stacked_plot_*.png`** — Step response visualization with P:D recommendations overlay
- **`*_PIDsum_PIDerror_Setpoint_stacked.png`** — Time-domain traces of PIDsum, PID error, and setpoint
- **`*_SetpointVsGyro_stacked.png`** — Setpoint command vs. filtered gyro response comparison
- **`*_GyroVsUnfilt_stacked.png`** — Filtered vs. unfiltered gyro comparison with calculated filtering delay
- **`*_Gyro_Spectrums_comparative.png`** — Frequency-domain amplitude spectrums (filtered and unfiltered gyro)
- **`*_Gyro_PSD_comparative.png`** — Gyro power spectral density in dB scale with peak detection
- **`*_D_Term_Spectrums_comparative.png`** — D-term frequency-domain amplitude spectrums with intelligent thresholding
- **`*_D_Term_PSD_comparative.png`** — D-term power spectral density with intelligent D-term activity detection
- **`*_D_Term_Heatmap_comparative.png`** — D-term energy distribution across throttle levels and frequencies
- **`*_Gyro_PSD_Spectrogram_comparative.png`** — Gyro spectrogram (PSD vs. time) using Short-Time Fourier Transform
- **`*_Throttle_Freq_Heatmap_comparative.png`** — System noise characteristics across throttle levels and frequencies
- **`*_PID_Activity_stacked.png`** — P, I, D term activity over time for each axis (Roll, Pitch, Yaw). Displays all three PID components on the same time-domain plot with unified Y-axis scaling for visual comparison. Each term shows min/avg/max statistics in the legend. Useful for visualizing PID contribution balance during flight and identifying control issues (persistent P-term offset, I-term wind direction, D-term phase lag).

#### P:D Ratio Recommendations

The system provides intelligent P:D tuning recommendations based on step-response peak analysis:

- **Conservative recommendations** (+≈18% D): Safe incremental steps; 2–3 iterations to optimal
- **Moderate recommendations** (+≈33% D): For experienced pilots; 1–2 iterations to optimal
- Automatically handles D-Min/D-Max systems (Betaflight 4.0+)
- Shows only base D when D-Min/D-Max is disabled
- Works for all aircraft sizes, including 10+ inch, where D can exceed P
- Includes warnings for severe overshoot or unreasonable P:D ratios
- Shows recommendations only when the step response needs improvement (skips optimal peak 0.95–1.04)
- **Note:** Peak value measures the first maximum after crossing the setpoint; the initial transient dip is normal system behavior

#### Optimal P Estimation (Optional)

Physics-aware P gain optimization based on response timing analysis:

- **Activation:** Disabled by default; enable with `--estimate-optimal-p` flag
- **Prop Size Selection:** Use `--prop-size <size>` to specify **propeller diameter** in inches (1.0-15.0, decimals allowed)
  - **Critical:** Match your actual prop size (e.g., 6" frame with 5" props → use `--prop-size 5`)
  - Supports decimal values (e.g., `--prop-size 5.5` for 5.5" props)
  - Defaults to 5.0 if not specified
  - Prop size determines rotational inertia (I ∝ radius²) which directly affects response time
  - Each prop size has physics-informed, empirically-derived Td (time to 50%) targets based on torque-to-rotational-inertia ratio
- **Prop Pitch Parameter:** Use `--prop-pitch <pitch>` to specify propeller pitch in inches (1.0-10.0)
  - Accounts for aerodynamic loading differences between prop pitches
  - Low pitch (e.g., 3.0"): Less drag → faster angular acceleration → faster response
  - High pitch (e.g., 6.0"): More drag → slower angular acceleration → slower response
  - Defaults to 4.5" (typical freestyle props) if not specified
  - Formula: pitch loading factor = (pitch / 4.5)^1.3
  - Example: 3.7" pitch has ~21% less drag than 4.5" baseline
  - **Note:** Currently used for information display only. Physics-based Td adjustment disabled to prevent circular logic issues. Uses empirically-validated frame-class targets for all comparisons.
- **Frame Geometry Parameters (Optional):** For asymmetric frame calculations (Roll vs Pitch response differences)
  - `--motor-diagonal <mm>`: M1→M4 diagonal motor spacing (rear-right to front-left)
  - `--motor-width <mm>`: M1→M3 side-to-side motor spacing (rear-right to rear-left)
  - `--weight <grams>`: Total aircraft weight (everything that flies)
  - When all provided, enables rotational inertia calculations accounting for frame asymmetry
- **Reserved Parameters (Not Currently Used):** The following parameters are parsed but not used in calculations. Reserved for potential future motor torque calculations:
  - `--motor-size <size>`: Motor stator dimensions (e.g., 2207) — reserved
  - `--motor-kv <kv>`: Motor KV rating — reserved
  - `--lipo <cells>`: Battery cell count (e.g., 4S, 5S, 6S) — reserved
  - **Why reserved:** Motor torque calculations would require voltage/current data not available in flight logs and could reintroduce circular logic issues. Current physics model uses rotational inertia only (mass distribution + geometry), which provides accurate results without these parameters.
- **Theory Foundation:** Based on BrianWhite's (PIDtoolbox author) insight that optimal response timing is aircraft-specific, not universal. The relationship between Td (time to 50%) and rotational inertia is **Td ∝ √(I/torque)**. While rotational inertia scales with mass × radius² (I ∝ mr²) for simple models, **actual Td is affected by many factors**: mass distribution (frame, motors, battery, props placement), motor torque characteristics, propeller aerodynamics, battery voltage, and ESC response. The frame-class targets below are **empirical estimates derived from flight data**, not pure physics calculations. Propeller size is used as a practical proxy for rotational inertia, but targets must be validated against actual flight logs for each specific build configuration.
- **Frame-Class Targets (Provisional - requires flight validation):**
  - **⚠️ IMPORTANT DISCLAIMER:** These targets are provisional empirical estimates and **MUST be validated through systematic flight testing**. They are derived from limited flight data and physics-informed intuition. Use as initial guidelines only. Validation data collection is ongoing.
  - **Constants Reference:** All targets are defined in `src/constants.rs` as the `TD_TARGETS` array (starting around line 309).
  - **Tolerance Ranges:** The (±) values represent acceptable response timing bands for each frame class—use these as recommended tuning acceptance ranges during flight validation, not measurement uncertainty or statistical confidence intervals.
  - 1" tiny whoop: 40ms ± 10.0ms (low power/torque)
  - 2" micro: 35ms ± 8.75ms
  - 3" toothpick/cinewhoop: 30ms ± 7.5ms
  - 4" racing: 25ms ± 6.25ms
  - 5" freestyle/racing: 20ms ± 5.0ms (common baseline)
  - 6" long-range: 28ms ± 7.0ms
  - 7" long-range: 37.5ms ± 9.375ms
  - 8" long-range: 47ms ± 11.75ms
  - 9" cinelifter: 56ms ± 14.0ms
  - 10" cinelifter: 65ms ± 16.25ms
  - 11" heavy-lift: 75ms ± 18.75ms
  - 12" heavy-lift: 85ms ± 21.25ms
  - 13" heavy-lift: 95ms ± 23.75ms
  - 14" heavy-lift: 105ms ± 26.25ms
  - 15" heavy-lift: 115ms ± 28.75ms
  - **How to Validate These Targets:**
    * **Method**: Run this tool on your flight logs with correct `--prop-size` and observe Td measurements
    * **Acceptance Criterion**: Your measured Td should fall within target ± tolerance range for your prop size
    * **Common Deviations**:
      - Faster than target + low noise = Excellent build, headroom for P increase
      - Slower than target + high noise = Mechanical issues or incorrect prop size specified
      - Within target + high noise = P at physical limits (optimal for this aircraft)
  - **Validation Plan (Provisional Targets):** These targets require systematic validation via flight data collection.
    * **Target Metrics:** Per frame class, measure Td mean and std dev across ≥10 flights (manual setpoint inputs or step-sticks); confidence threshold: Td within ±10% of predicted target.
    * **Data Collection Protocol:**
      - **Flight Logs:** Controlled stick inputs on tethered or low-altitude flights; log format: Betaflight CSV with gyro, setpoint, P/D gains recorded; sample ≥3 distinct P settings per frame class.
      - **System Documentation:** Record complete system specs (frame, motors, props, battery, AUW) for each test aircraft to correlate Td measurements with physical parameters.
      - **Note:** Bench testing isolated motors cannot validate Td targets—Td represents full system response including frame rotational inertia, which is absent in component-level tests.
    * **Test Matrix:** One representative aircraft per frame class (1", 3", 5", 7", 10"—minimum coverage); repeat with 2 different motor/prop combos per class to validate robustness.
    * **Tracking & Results:** Create GitHub issue template for each frame class linking to uploaded flight log summaries (mean Td, actual P setting, pilot feedback, system specs). Include pass/fail criteria: predicted Td ±10%, pass/fail per class.
    * **Timeline:** TBD (seeking community validation data collection - see GitHub issues for current status)
- **Analysis Components:**
  - Collects individual Td measurements from all valid step response windows
  - Calculates response consistency metrics (mean, std dev, coefficient of variation)
  - Compares measured Td against frame-class targets
  - Classifies Td deviation (significantly slower, moderately slower, within target, faster)
  - Provides P gain recommendations based on deviation and noise levels
- **Recommendation Types:**
  - **P Increase:** When Td is slower than target with acceptable noise levels
  - **Optimal:** When Td is within target range or at physical limits
  - **P Decrease:** When Td is faster than target with high noise (rare)
  - **Investigate:** When measurements suggest mechanical issues or incorrect frame class
- **Output Format:** Detailed console report with:
  - Current P gain and measured Td statistics
  - Frame class comparison and deviation percentage
  - Physical limit indicators (response speed, noise level, consistency)
  - Clear recommendation with reasoning
- **Relationship to P:D Recommendations:**
  - P:D ratio recommendations (existing feature): Analyze peak overshoot → adjust D-term
  - Optimal P estimation (new feature): Analyze response timing → adjust P magnitude
  - Both features are complementary and can run simultaneously for complete tuning guidance

### Step-Response Comparison with Other Analysis Tools

This implementation provides detailed and configurable analysis of flight controller performance. The modular design and centralized configuration system make it adaptable for various analysis requirements.

#### Compared to PIDtoolbox/Matlab (`PTstepcalc.m`)
    * **Deconvolution Method:** Both use Wiener deconvolution with a regularization term.
    * **Windowing:** This implementation uses Tukey (`TUKEY_ALPHA`) on input/output before FFT; Matlab uses Hann.
    * **Smoothing:** This implementation has optional initial gyro smoothing (`INITIAL_GYRO_SMOOTHING_WINDOW`) and mandatory post-average smoothing (`POST_AVERAGING_SMOOTHING_WINDOW`). Matlab smooths raw gyro input upfront.
    * **Normalization/Y-Correction:**
        * This implementation: Optional individual response Y-correction (normalize by own steady-state mean, `APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION`, `Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS`) followed by final normalization of the averaged response to target 1.0 (within `FINAL_NORMALIZED_STEADY_STATE_TOLERANCE`).
        * Matlab: Optional Y-correction on individual responses by calculating an offset to make the mean 1.0.
    * **Quality Control (QC):** Both apply QC to individual responses based on steady-state characteristics. This implementation uses `NORMALIZED_STEADY_STATE_MIN_VAL`, `NORMALIZED_STEADY_STATE_MAX_VAL`, and optionally `NORMALIZED_STEADY_STATE_MEAN_MIN`, `NORMALIZED_STEADY_STATE_MEAN_MAX`. Matlab uses `min(steadyStateResp) > 0.5 && max(steadyStateResp) < 3`.
    * **Output:** This implementation can plot low/high/combined responses based on `setpoint_threshold` if `--dps` is provided with a value. Matlab stacks all valid responses.

#### Compared to PlasmaTree/Python (`PID-Analyzer.py`)
    * **Deconvolution Method:** PlasmaTree also uses Wiener deconvolution with a signal-to-noise ratio (`sn`) term in the denominator, derived from a `cutfreq` parameter and smoothed, effectively acting as frequency-dependent regularization. This implementation uses a simpler constant regularization term (`0.0001`).
    * **Windowing:** PlasmaTree uses a Hanning window by default (or Tukey) applied to input and output segments before deconvolution. This implementation uses a Tukey window.
    * **Input for Deconvolution:** PlasmaTree calculates an `input` signal by reconstructing the setpoint as seen by the PID loop. This implementation directly uses the logged setpoint values.
    * **Smoothing:** PlasmaTree does not include initial gyro smoothing like this implementation's `INITIAL_GYRO_SMOOTHING_WINDOW`. For the final averaged response, PlasmaTree uses a 2D histogram and Gaussian smoothing to find the "mode" response, different from this implementation's direct moving average on the time-domain averaged response.
    * **Normalization/Y-Correction:**
        * This implementation: Individual responses can be normalized to their steady-state mean, followed by final normalization of the averaged response.
        * PlasmaTree: Uses `weighted_mode_avr` to find the most common trace shape from response collections, which inherently handles variations without explicit Y-correction.
    * **Quality Control (QC):** PlasmaTree has a `resp_quality` metric based on deviation from initial average and uses a `toolow_mask` for low input responses. This implementation uses direct steady-state value checks.
    * **Averaging:**
        * This implementation: `average_responses` performs weighted average of QC'd responses.
        * PlasmaTree: `weighted_mode_avr` uses 2D histogram analysis to determine representative traces.
    * **Output:** Both can plot low and high input responses based on configurable thresholds.

This Rust implementation offers a comprehensive and configurable analysis pipeline for flight controller performance evaluation with sophisticated signal processing techniques and detailed visualization capabilities.
