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
  - [Optimal P Estimation (Optional, Experimental)](#optimal-p-estimation-optional-experimental)
  - [Step-Response Comparison with Other Analysis Tools](#step-response-comparison-with-other-analysis-tools)
    - [Compared to PIDtoolbox/Matlab (PTstepcalc.m)](#compared-to-pidtoolboxmatlab-ptstepcalcm)
    - [Compared to PlasmaTree/Python (PID-Analyzer.py)](#compared-to-plasmatreepython-pid-analyzerpy)

The Rust program processes Betaflight Blackbox CSV logs to generate various plots. Here's a concise overview:

### Configuration

All analysis parameters, thresholds, plot dimensions, and algorithmic constants are centrally defined in `src/constants.rs`, making the implementation highly configurable for different analysis needs and flight controller characteristics.

### Core Functionality

1.  **Argument Parsing (`src/main.rs`):**
    * Parses command-line arguments: input CSV file(s), an optional `--dps` parameter (requires a numeric threshold value for detailed step response plots with low/high split), an optional `--output-dir` for specifying the output directory, and plot-selection flags (`--core` [default], `--extended`, `--step`, `--motor`, `--spectrums`, `--tracking`, `--gyro-filt`, `--setpoint`, `--pid`, `--psd`, `--heatmaps`, `--bode`).
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
                * **Windowing:** The input signals (setpoint and gyro) are segmented into overlapping windows (`winstacker_contiguous`) of `FRAME_LENGTH_S` duration — intentionally longer than the displayed `RESPONSE_LENGTH_S` to improve frequency-domain resolution for the Wiener deconvolution; only the first `RESPONSE_LENGTH_S` of each estimated step response is retained for display and analysis. A Tukey window (`tukeywin` with `TUKEY_ALPHA`) is applied to each segment to reduce spectral leakage.
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
                * **Peak Analysis Ranges (based on step response overshoot/undershoot):** Simplified 6-zone structure aligned with real-world practical tuning:
                    * **< 1.00 (Undershoot):** Too much D damping. Recommendation (conservative): P:D × (1.05 / peak) — proportional decrease targeting sweet-spot centre.
                    * **1.00–1.02 (Near Optimal):** Critically damped transition zone. Recommendation (none) + optional D−1 hint for fine-tuning.
                    * **1.02–1.08 (Optimal / Pro Sweet Spot):** Ideal response with responsive feel. Recommendation (none) — no adjustment needed.
                    * **1.08–1.12 (Acceptable):** Slight bounce but still locked-in. Recommendation (conservative): P:D×0.98 (increase D by ~2%).
                    * **1.12–1.20 (Overshoot):** Visible overshoot but manageable. Recommendation (conservative): P:D×0.92 (increase D by ~8.7%) + Recommendation (moderate): P:D×0.75 (increase D by ~33%).
                    * **> 1.20 (Significant Overshoot):** Excessive oscillation indicating P is too high. Recommendation (conservative) + (moderate) + Recommendation (aggressive): P:D×0.65 (increase D by ~54%).
                * **Multi-Tiered Recommendations:**
                    * **Conservative** (PD_RATIO_CONSERVATIVE_MULTIPLIER = 0.85): Reduces P:D ratio by 15%, increasing D by ~18%. Safe for most pilots.
                    * **Moderate** (PD_RATIO_MODERATE_MULTIPLIER = 0.75): Reduces P:D ratio by 25%, increasing D by ~33%. For experienced pilots.
                    * **Aggressive** (PD_RATIO_AGGRESSIVE_MULTIPLIER = 0.65): Reduces P:D ratio by 35%, increasing D by ~54%. For tuning significant overshoot zones (>1.20).
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
            * **Other plots generated (`src/plot_functions/`):** Which plots are generated is controlled by the `PlotConfig` struct. The default (`--core`) enables the core set; `--extended` enables all plots except Bode; individual flags enable specific subsets. Plots are gated per-field in `PlotConfig`:
                * **Core plots (default):**
                    * `plot_setpoint_vs_gyro`: Setpoint and filtered gyro time-domain comparison for each axis.
                    * `plot_gyro_vs_unfilt`: Filtered vs. unfiltered gyro time-domain comparison for each axis. Includes enhanced cross-correlation filtering delay calculation.
                    * `plot_gyro_spectrums`: Frequency-domain amplitude spectrums of filtered and unfiltered gyro data with intelligent peak detection and labeling using scale-aware thresholds (`FILTERED_GYRO_MIN_THRESHOLD` for filtered gyro data). Includes enhanced cross-correlation filtering delay calculation and flight firmware filter response curve overlays.
                    * `plot_d_term_spectrums`: Frequency-domain amplitude spectrums of D-term data with intelligent peak detection using scale-aware thresholds (`FILTERED_D_TERM_MIN_THRESHOLD` for filtered D-term data). Includes enhanced cross-correlation filtering delay calculation with intelligent D-term activity detection (skips axes where D gain = 0).
                    * `plot_motor_spectrums`: Motor output frequency analysis.
                * **Extended plots (`--extended` or individual flags):**
                    * `plot_pidsum_error_setpoint`: PIDsum (P+I+D), PID Error (Setpoint - GyroADC), and Setpoint time-domain traces for each axis. (`--pid`)
                    * `plot_pid_activity`: P, I, D term activity over time. (`--pid`)
                    * `plot_setpoint_derivative`: Setpoint rate-of-change (feed-forward proxy) for each axis. (`--setpoint`)
                    * `plot_psd`: Power Spectral Density plots in dB scale with peak labeling. Includes enhanced cross-correlation filtering delay calculation. (`--psd`)
                    * `plot_d_term_psd`: Power Spectral Density plots of D-term data in dB scale with intelligent threshold filtering (`PSD_PEAK_LABEL_MIN_VALUE_DB` for filtered data) and enhanced formatting. Includes enhanced cross-correlation filtering delay calculation with intelligent D-term activity detection (skips axes where D gain = 0). (`--psd`)
                    * `plot_d_term_heatmap`: D-term throttle-frequency heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze D-term energy distribution across different throttle levels. (`--heatmaps`)
                    * `plot_psd_db_heatmap`: Spectrograms showing PSD vs. time as heatmaps using Short-Time Fourier Transform (STFT) with configurable window duration and overlap. (`--heatmaps`)
                    * `plot_throttle_freq_heatmap`: Heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze noise characteristics across different throttle levels. (`--heatmaps`)

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

#### Optimal P Estimation (Optional, Experimental)

Physics-derived P gain optimization using a Torque-Inertia Profiler that measures aircraft-specific dynamics directly from flight log throttle-punch events. No prop-size input is required — the aircraft's torque-to-inertia ratio is derived from the logs.

- **Activation:** Disabled by default; enable with `--estimate-optimal-p` flag.
- **Requires:** A `.headers.csv` metadata file alongside each input CSV (produced by `blackbox_decode`). Without it, P gain values are unavailable and optimal P estimation is skipped with a skip-reason shown in console and PNG. All other analyses (step response, spectrums, P:D recommendations) remain unaffected.
- **⚠️ Status:** Experimental. `TORQUE_PROFILER_ACHIEVABILITY_FACTOR` bridges the gap between the theoretical physics formula and real-world flight performance (ESC lag, prop-wash, motor startup). It is empirically calibrated and may need adjustment for aircraft significantly different from a mid-size freestyle build.

##### Torque-Inertia Profiler (`src/data_analysis/torque_inertia_profiler.rs`)

- **Phase 1 — Aircraft Profiling (per group, before per-file processing):**
  - All logs sharing an aircraft key are processed together via `profile_aircraft_group()`.
  - `extract_punch_ratios()` detects throttle-punch events: `setpoint[3]` increases ≥ `THROTTLE_PUNCH_MIN_DELTA` within `THROTTLE_PUNCH_WINDOW_MS`.
  - For each punch, peak angular acceleration `|Δgyro/Δt|` in the response window (after `TORQUE_PROFILER_SETTLE_MS` of ESC/motor settle time, converted to samples at runtime from the actual log sample rate) is divided by the normalised throttle command delta → `torque_inertia_ratio`.
  - Ratios are aggregated into `AircraftProfile` (median + half-IQR spread per axis, Roll and Pitch only). Yaw ratios are collected but not used in optimal-P analysis because Yaw dynamics differ from Roll/Pitch and the current formula is not calibrated for Yaw.
  - Requires ≥ `TORQUE_PROFILER_MIN_EVENTS` punch events. If insufficient, a skip message appears in both console and PNG overlay.

- **Phase 2 — Per-File Optimal-P Analysis:**
  - Physics formula: `Td_ms = TORQUE_PROFILER_TD_CALC_K / sqrt((P / TORQUE_PROFILER_P_SCALE) × torque_inertia_ratio) × TORQUE_PROFILER_ACHIEVABILITY_FACTOR`
  - Per-file Td samples are collected from all valid step response windows and compared against the physics-derived target.
  - HF noise energy from D-term spectral analysis informs whether noise limits further P increase.
  - `OptimalPAnalysis` classifies the result as `Increase`, `Optimal`, `Decrease`, or `Investigate`.

- **Aircraft Grouping (`extract_aircraft_key()`):**
  - Strips `_YYYYMMDD_HHMMSS` timestamp from filename stem so logs from the same aircraft across multiple sessions share one key.
  - When the prefix ends with `BLACKBOX_LOG` (generic Betaflight/EmuFlight naming), the craft name following the timestamp is appended to prevent distinct aircraft from collapsing into one group (e.g., `BTFL_BLACKBOX_LOG_YYYYMMDD_HHMMSS_CRAFTNAME.NN.csv` → key `BTFL_BLACKBOX_LOG_CRAFTNAME`).

- **Key Constants (`src/constants.rs`):**
  - `THROTTLE_PUNCH_MIN_DELTA` — minimum throttle step (0–1000 units) to qualify as a punch
  - `THROTTLE_PUNCH_WINDOW_MS` — detection window for the throttle rise
  - `THROTTLE_RESPONSE_WINDOW_MS` — gyro response measurement window
  - `TORQUE_PROFILER_SETTLE_MS` — settle time (ms) skipped at response start (ESC/motor lag allowance); converted to samples at runtime so it is correct at all loop rates
  - `TORQUE_PROFILER_MIN_EVENTS` — minimum punches required for a reliable profile
  - `TORQUE_PROFILER_P_SCALE` — converts raw firmware P gain to physical units
  - `TORQUE_PROFILER_TD_CALC_K` — Td numerator constant (π × 500)
  - `TORQUE_PROFILER_ACHIEVABILITY_FACTOR` — empirical calibration coefficient

- **Recommendation Types:**
  - **P Increase:** Td slower than target with acceptable noise → P is conservative
  - **Optimal:** Td within target range or at physical limits → P is well-matched
  - **P Decrease:** Td faster than target with high noise → P is too high (rare)
  - **Investigate:** Measurements suggest mechanical issues or abnormal dynamics

- **Output:** Console and PNG legend overlay (identical content) per axis:
  - `Td:` — measured Td mean with target `±tolerance` and `windows=` count
  - `Td source:` — `File Group` (multi-file run) or `Single File`, with flight count and throttle-punch count that produced the physics target
  - `Noise:` — HF D-term energy level (`LOW` / `MODERATE` / `HIGH` / `UNKNOWN`) for the current flight
  - `Deviation:` — % difference between measured Td and physics target, with zone label
  - `Current P=` — P gain value from the flight's metadata
  - `Recommendation` — one of `(Conservative)`, `(Decrease)`, `Current P is optimal`, or `Investigate —` with reason; includes calculated D adjustment
  - `Reliable:` / `Unreliable:` — always shows both `Consistency=N% (⊢≥70%)` and `CV=N% (⊢≤40%)`; `Unreliable` is highlighted in orange when either threshold is not met
  - `Setpoint Authority:` — always shown; classifies flight inputs as `LOW`, `MODERATE`, or `HIGH` based on the **mean** of per-window max setpoints (see below). Orange for `LOW`.
  - When profiling is skipped (insufficient punch events), a skip reason replaces the above. See **Consistency and Reliability Interpretation** below.

- **Consistency and Reliability Interpretation (CV):**
  - **CV (Coefficient of Variation)** = standard deviation / mean of individual Td measurements across all valid step-response windows. It quantifies how scattered the measurements are relative to their average.
  - **Low CV:** Td measurements are tightly clustered — the log contains clean, repeatable dynamics and recommendations are trustworthy.
  - **High CV (exceeds `TD_COEFFICIENT_OF_VARIATION_MAX`):** Td measurements vary widely across windows — `Unreliable:` is shown (orange) in both console and PNG. Recommendations should be treated with caution.
  - **CV = N/A:** Fewer than `TD_SAMPLES_MIN_FOR_STDDEV` valid Td samples were available; standard deviation cannot be computed. The mean is still reported and `CV=N/A` appears in the reliability line.
  - **Setpoint Authority classification:** Always-visible line in both console and PNG. Uses the **mean** of per-window max setpoints (not the maximum) to classify the flight:
    - `LOW` (`mean < 100 dps`, orange) — hover/slow-cruise inputs; all P:D recommendations are still shown, but the pilot should treat them with caution.
    - `MODERATE` (`100–250 dps`) — normal sport/freestyle inputs.
    - `HIGH` (`> 250 dps`) — aggressive or race-pace inputs.
    Format: `Setpoint Authority: LOW (mean=68dps ⊢≥100dps)`. Using the mean rather than the max prevents a single high-input window from masking an otherwise gentle hover log.
  - **Why hover logs produce high CV:** Small setpoint inputs → deconvolution is noise-sensitive → each window captures a different noise realisation. The averaged response may appear plausible (noise averages out) while individual window variance remains high. CV exposes this where the mean alone cannot.
  - **Over-P limitation:** **When P is already too high and the aircraft oscillates, the profiler may report "Optimal" rather than "Decrease P."** An oscillatory step response produces a short, aggressive measured Td — which, fed into the physics formula, yields a P_optimal close to the current (excessive) P. The profiler cannot reliably distinguish a well-tuned fast response from an over-tuned oscillating one using Td alone. The indirect signal is CV: severe oscillation typically scatters Td samples widely and triggers the consistency warning. **If your gains feel high or the craft exhibits oscillation, start from a lower P before relying on these recommendations.** Optimal P estimation is most accurate when the craft is in a reasonable tuning range — it is a validator and refinement tool, not a recovery tool for badly mis-tuned aircraft. Only experienced pilots are likely to recognise this situation by feel.
  - **High CV without LOW authority:** If the consistency warning fires but `Setpoint Authority` is `MODERATE` or `HIGH`, the scatter is not caused by low-energy hover inputs. Remaining causes include propwash, inconsistent maneuvers, and oscillation from over-P. **If gains feel high, treat this combination as a prompt to verify the craft is not oscillating before acting on any recommendation.**
  - **Summary of dependability signals in output:**
    - `windows=` on the Td line — number of valid step-response windows contributing to the Td mean; more windows = more statistical weight
    - `Td source:` — flight and throttle-punch counts that calibrated the physics target; `File Group` means data was pooled across multiple logs
    - `Reliable:` / `Unreliable:` with `Consistency %` and `CV` — how repeatable the per-flight step-response measurements are (independent of how many punches fed the physics target)
    - `Setpoint Authority:` — mean setpoint level across valid windows; `LOW` indicates hover/gentle inputs that may reduce step-response quality
    - Noise level (`LOW` / `MODERATE` / `HIGH`) — HF D-term energy for the current flight; high noise limits safe P increase

- **Relationship to P:D Recommendations:**
  - P:D ratio recommendations: analyze peak overshoot → adjust D relative to P
  - Optimal P estimation: analyze response timing → adjust P magnitude
  - Both features are complementary; both appear in console output and PNG legend simultaneously

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
