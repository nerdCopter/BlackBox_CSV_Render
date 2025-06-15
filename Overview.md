## Code Overview and Step Response Calculation

The Rust program processes Betaflight Blackbox CSV logs to generate various plots. Here's a concise overview:

**Configuration:**
All analysis parameters, thresholds, plot dimensions, and algorithmic constants are centrally defined in `src/constants.rs`, making the implementation highly configurable for different analysis needs and flight controller characteristics.

**Core Functionality:**

1.  **Argument Parsing (`src/main.rs`):**
    *   Parses command-line arguments: input CSV file(s), an optional `--dps` flag (for detailed step response plots with an optional threshold determining low/high split), and an optional `--out-dir` for specifying the output directory.
    *   Additional options include `--help` and `--version` for user assistance.
    *   Handles multiple input files and determines if a directory prefix should be added to output filenames to avoid collisions when processing files from different directories.

2.  **File Processing (`src/main.rs:process_file`):**
    *   For each input CSV:
        *   **Path Setup:** Determines input and output paths.
        *   **Data Parsing (`src/data_input/log_parser.rs:parse_log_file`):** Reads the CSV, extracts log data (time, gyro, setpoint, F-term, etc.), sample rate, and checks for the presence of essential data headers.
        *   **Step Response Data Preparation:**
            *   Filters log data, excluding the start and end seconds (constants `EXCLUDE_START_S`, `EXCLUDE_END_S` from `src/constants.rs`) to create contiguous segments of time, setpoint, and gyro data for each axis. This data is stored in `contiguous_sr_input_data`.
        *   **Step Response Calculation (`src/data_analysis/calc_step_response.rs:calculate_step_response`):**
            *   This is the core of the step response analysis. It implements **non-parametric system identification** using Wiener deconvolution rather than traditional first-order or second-order curve fitting. This approach directly extracts the system's actual step response without assuming a specific mathematical model, allowing it to capture complex, higher-order dynamics and non-linearities.
            *   For each axis (Roll, Pitch, Yaw):
                *   It takes the prepared time, setpoint, and (optionally smoothed via `INITIAL_GYRO_SMOOTHING_WINDOW`) gyro data arrays and the sample rate.
                *   **Windowing:** The input signals (setpoint and gyro) are segmented into overlapping windows (`winstacker_contiguous`) of `FRAME_LENGTH_S` duration. A Tukey window (`tukeywin` with `TUKEY_ALPHA`) is applied to each segment to reduce spectral leakage.
                *   **Movement Threshold:** Windows are discarded if the maximum absolute setpoint value within them is below `MOVEMENT_THRESHOLD_DEG_S`.
                *   **Deconvolution:** For each valid window, Wiener deconvolution (`wiener_deconvolution_window`) is performed between the windowed setpoint (input) and gyro (output) signals in the frequency domain. This estimates the impulse response of the system. A regularization term (`0.0001`) helps stabilize the deconvolution.
                *   **Impulse to Step Response:** The resulting impulse response is converted to a step response by cumulative summation (`cumulative_sum`). This step response is then truncated to `RESPONSE_LENGTH_S`.
                *   **Y-Correction (Normalization):**
                    *   If `APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION` is true, each individual step response window is normalized. The mean of its steady-state portion (defined by `STEADY_STATE_START_S` and `STEADY_STATE_END_S`) is calculated. If this mean is significant (abs > `Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS`), the entire response window is divided by this mean, aiming to make its steady-state value approach 1.0.
                *   **Quality Control (QC):**
                    *   Each (potentially Y-corrected) step response window undergoes QC. The minimum and maximum values of its steady-state portion are checked against `NORMALIZED_STEADY_STATE_MIN_VAL` and `NORMALIZED_STEADY_STATE_MAX_VAL`.
                    *   Optionally (`ENABLE_NORMALIZED_STEADY_STATE_MEAN_CHECK`), the mean of the steady-state portion is also checked against `NORMALIZED_STEADY_STATE_MEAN_MIN` and `NORMALIZED_STEADY_STATE_MEAN_MAX`.
                    *   Only windows passing QC are kept.
                *   The function returns:
                    1.  A time vector for the step response plot (`response_time`).
                    2.  A 2D array (`valid_stacked_responses`) containing all step response windows that passed QC.
                    3.  A 1D array (`valid_window_max_setpoints`) of the maximum setpoint values for each corresponding valid window.
        *   **Plot Generation (various functions in `src/plot_functions/`):**
            *   `plot_step_response` (in `src/plot_functions/plot_step_response.rs`): Takes the results from `calculate_step_response`.
                *   Separates the QC'd responses into "low" and "high" setpoint groups based on the `setpoint_threshold` if the `--dps` flag (and thus `show_legend`) is active.
                *   **Averaging & Final Normalization (`plot_functions::plot_step_response::process_response`):**
                    *   The QC'd responses (either low, high, or combined) are averaged using `calc_step_response::average_responses`.
                    *   The averaged response is smoothed using a moving average (`calc_step_response::moving_average_smooth_f64`) with `POST_AVERAGING_SMOOTHING_WINDOW`.
                    *   The smoothed response is shifted to start at 0.0.
                    *   A **final normalization** step is performed: the mean of the steady-state portion of this *averaged, smoothed, and shifted* response is calculated. The entire response is then divided by this mean to ensure the plotted average response aims for a steady-state of 1.0.
                    *   The final response is only plotted if its steady-state mean (after this final normalization) is within `FINAL_NORMALIZED_STEADY_STATE_TOLERANCE` of 1.0.
                *   Calculates peak value and delay time (Td) for each plotted average response using `calc_step_response::find_peak_value` and `calc_step_response::calculate_delay_time`.
                    *   **Delay Time (Td) Calculation:** The delay time is calculated as the time for the step response to reach 50% of its final value. Linear interpolation is used for precise determination of the 50% threshold crossing. A fixed offset of -1ms is applied to the calculated time (in milliseconds) before converting to seconds, and the result is constrained to be non-negative.
                *   Generates and saves the step response plot.
            *   **Other plots generated (`src/plot_functions/`):**
                *   `plot_pidsum_error_setpoint`: PIDsum (P+I+D), PID Error (Setpoint - GyroADC), and Setpoint time-domain traces for each axis.
                *   `plot_setpoint_vs_gyro`: Setpoint and filtered gyro time-domain comparison for each axis.
                *   `plot_gyro_vs_unfilt`: Filtered vs. unfiltered gyro time-domain comparison for each axis. Includes enhanced cross-correlation filtering delay calculation.
                *   `plot_gyro_spectrums`: Frequency-domain amplitude spectrums of filtered and unfiltered gyro data with peak detection and labeling with configurable thresholds. Includes enhanced cross-correlation filtering delay calculation.
                *   `plot_psd`: Power Spectral Density plots in dB scale with peak labeling. Includes enhanced cross-correlation filtering delay calculation.
                *   `plot_psd_db_heatmap`: Spectrograms showing PSD vs. time as heatmaps using Short-Time Fourier Transform (STFT) with configurable window duration and overlap.
                *   `plot_throttle_freq_heatmap`: Heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze noise characteristics across different throttle levels.

**Filtering Delay Calculation (`src/data_analysis/filter_delay.rs`):**

### Enhanced Cross-Correlation Method (Primary Implementation)
*   **Algorithm:** For each axis (Roll, Pitch, Yaw), calculates normalized cross-correlation between filtered (`gyroADC`) and unfiltered (`gyroUnfilt`) gyro signals at different time delays using double-precision (f64) arithmetic throughout.
*   **Delay Detection:** Identifies the delay that produces the highest correlation coefficient and converts from samples to milliseconds using the sample rate.
*   **Subsample Precision:** Uses parabolic interpolation around the peak correlation to achieve subsample delay accuracy, addressing precision limitations of basic sample-rate resolution.
*   **Quality Control:** Requires correlation coefficients above configurable thresholds (`MIN_CORRELATION_THRESHOLD`, `FALLBACK_CORRELATION_THRESHOLD`) with fallback mechanisms for challenging signal conditions.
*   **Error Handling:** Provides detailed error reporting (`DelayCalculationError`) for insufficient data, low correlation, and signal mismatches.
*   **Precision Consistency:** Recent improvements ensure all correlation calculations use f64 precision, eliminating precision loss in fallback correlation paths.

### Implementation Details
*   **Data Validation:** Performs comprehensive data availability diagnostics across all axes before analysis.
*   **Averaging:** Individual axis delays are averaged to provide an overall system delay measurement when sufficient correlation is achieved.
*   **Bounds Checking:** Comprehensive bounds checking with `saturating_sub()` and explicit runtime verification prevents array access violations. Limits maximum delay search range (`MAX_DELAY_FRACTION`, `MAX_DELAY_SAMPLES`) to prevent unrealistic results and ensures robust parabolic interpolation.
*   **Configurable Thresholds:** All correlation thresholds and delay search parameters are defined as named constants in `src/constants.rs` for maintainability and tuning.
*   **Display:** Results are shown in console output with confidence metrics (as percentages), and estimates are integrated into plot legends as "Delay: X.Xms(c:XX%)" for `_GyroVsUnfilt_stacked.png`, `_Gyro_Spectrums_comparative.png`, and `_Gyro_PSD_comparative.png` outputs.
*   **Method Labeling:** Enhanced cross-correlation results are clearly labeled in plot legends to distinguish from basic correlation methods.

### Implementation Features

**Precision Design:**
*   **Double-Precision Consistency:** All correlation calculations use f64 precision throughout for maximum accuracy.
*   **Robust Error Handling:** Comprehensive bounds checking with runtime verification prevents array access violations.

**Method Clarity:**
*   **Clear Method Labeling:** Enhanced cross-correlation results are labeled as "Delay" in plot legends with confidence metrics (c:XX%).
*   **Focused Implementation:** Uses a single, well-tested enhanced cross-correlation method with reliable fallback mechanisms.

**Code Quality:**
*   **Centralized Configuration:** All analysis parameters consolidated in `src/constants.rs` for easier maintenance and tuning.
*   **Comprehensive Documentation:** Detailed inline documentation and extensive validation testing ensure reliability.

**Comparison with Other Analysis Tools:**

This implementation provides detailed and configurable analysis of flight controller performance. The modular design and centralized configuration system make it adaptable for various analysis requirements.

*   **Compared to PIDtoolbox/Matlab (`PTstepcalc.m`):**
    *   **Deconvolution Method:** Both use Wiener deconvolution with a regularization term.
    *   **Windowing:** Rust uses Tukey (`TUKEY_ALPHA`) on input/output before FFT; Matlab uses Hann.
    *   **Smoothing:** Rust has optional initial gyro smoothing (`INITIAL_GYRO_SMOOTHING_WINDOW`) and mandatory post-average smoothing (`POST_AVERAGING_SMOOTHING_WINDOW`). Matlab smooths raw gyro input upfront.
    *   **Normalization/Y-Correction:**
        *   This implementation: Optional individual response Y-correction (normalize by own steady-state mean, `APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION`, `Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS`) followed by final normalization of the averaged response to target 1.0 (within `FINAL_NORMALIZED_STEADY_STATE_TOLERANCE`).
        *   Matlab: Optional Y-correction on individual responses by calculating an offset to make the mean 1.0.
    *   **Quality Control (QC):** Both apply QC to individual responses based on steady-state characteristics. This implementation uses `NORMALIZED_STEADY_STATE_MIN_VAL`, `NORMALIZED_STEADY_STATE_MAX_VAL`, and optionally `NORMALIZED_STEADY_STATE_MEAN_MIN`, `NORMALIZED_STEADY_STATE_MEAN_MAX`. Matlab uses `min(steadyStateResp) > 0.5 && max(steadyStateResp) < 3`.
    *   **Output:** This implementation can plot low/high/combined responses based on `setpoint_threshold` if `--dps` is used. Matlab stacks all valid responses.

*   **Compared to PlasmaTree/Python (`PID-Analyzer.py`):**
    *   **Deconvolution Method:** PlasmaTree also uses Wiener deconvolution with a signal-to-noise ratio (`sn`) term in the denominator, derived from a `cutfreq` parameter and smoothed, effectively acting as frequency-dependent regularization. This implementation uses a simpler constant regularization term (`0.0001`).
    *   **Windowing:** PlasmaTree uses a Hanning window by default (or Tukey) applied to input and output segments before deconvolution. This implementation uses a Tukey window.
    *   **Input for Deconvolution:** PlasmaTree calculates an `input` signal by reconstructing the setpoint as seen by the PID loop. This implementation directly uses the logged setpoint values.
    *   **Smoothing:** PlasmaTree does not include initial gyro smoothing like this implementation's `INITIAL_GYRO_SMOOTHING_WINDOW`. For the final averaged response, PlasmaTree uses a 2D histogram and Gaussian smoothing to find the "mode" response, different from this implementation's direct moving average on the time-domain averaged response.
    *   **Normalization/Y-Correction:**
        *   This implementation: Individual responses can be normalized to their steady-state mean, followed by final normalization of the averaged response.
        *   PlasmaTree: Uses `weighted_mode_avr` to find the most common trace shape from response collections, which inherently handles variations without explicit Y-correction.
    *   **Quality Control (QC):** PlasmaTree has a `resp_quality` metric based on deviation from initial average and uses a `toolow_mask` for low input responses. This implementation uses direct steady-state value checks.
    *   **Averaging:**
        *   This implementation: `average_responses` performs weighted average of QC'd responses.
        *   PlasmaTree: `weighted_mode_avr` uses 2D histogram analysis to determine representative traces.
    *   **Output:** Both can plot low and high input responses based on configurable thresholds.

This Rust implementation offers a comprehensive and configurable analysis pipeline for flight controller performance evaluation with sophisticated signal processing techniques and detailed visualization capabilities.
