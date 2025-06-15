## Code Overview and Step Response Calculation

The Rust program processes Betaflight Blackbox CSV logs to generate various plots. Here's a concise overview:

**Configuration:**
All analysis parameters, thresholds, plot dimensions, and algorithmic constants are centrally defined in `src/constants.rs`, making the implementation highly configurable for different analysis needs and flight controller characteristics.

**Core Functionality:**

1.  **Argument Parsing (`src/main.rs`):**
    *   Parses command-line arguments: input CSV file(s), an optional `--dps` flag (for detailed step response plots with an optional threshold determining low/high split), and an optional `--out-dir` for specifying the output directory.
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
                *   `plot_gyro_vs_unfilt`: Filtered vs. unfiltered gyro time-domain comparison for each axis. Includes filtering delay calculation.
                *   `plot_gyro_spectrums`: Frequency-domain amplitude spectrums of filtered and unfiltered gyro data with peak detection and labeling with configurable thresholds. Includes filtering delay calculation.
                *   `plot_psd`: Power Spectral Density plots in dB scale with peak labeling. Includes filtering delay calculation.
                *   `plot_psd_db_heatmap`: Spectrograms showing PSD vs. time as heatmaps using Short-Time Fourier Transform (STFT) with configurable window duration and overlap.
                *   `plot_throttle_freq_heatmap`: Heatmaps showing PSD vs. throttle (Y-axis) and frequency (X-axis) to analyze noise characteristics across different throttle levels.

**Filtering Delay Calculation (`src/data_analysis/filter_delay.rs`):**

### Enhanced Cross-Correlation Method
*   **Algorithm:** For each axis (Roll, Pitch, Yaw), calculates normalized cross-correlation between filtered (`gyroADC`) and unfiltered (`gyroUnfilt`) gyro signals at different time delays.
*   **Delay Detection:** Identifies the delay that produces the highest correlation coefficient and converts from samples to milliseconds using the sample rate.
*   **Sub-sample Precision:** Uses parabolic interpolation around the peak correlation to achieve sub-sample delay accuracy, addressing precision limitations of basic sample-rate resolution.
*   **Quality Control:** Requires correlation coefficients above configurable thresholds (`MIN_CORRELATION_THRESHOLD`, `FALLBACK_CORRELATION_THRESHOLD`) with fallback mechanisms for challenging signal conditions.
*   **Error Handling:** Provides detailed error reporting (`DelayCalculationError`) for insufficient data, low correlation, and signal mismatches.

### Implementation Details
*   **Data Validation:** Performs comprehensive data availability diagnostics across all axes before analysis.
*   **Averaging:** Individual axis delays are averaged to provide an overall system delay measurement when sufficient correlation is achieved.
*   **Bounds Checking:** Comprehensive bounds checking with `saturating_sub()` and explicit runtime verification prevents array access violations. Limits maximum delay search range (`MAX_DELAY_FRACTION`, `MAX_DELAY_SAMPLES`) to prevent unrealistic results and ensures robust parabolic interpolation.
*   **Configurable Thresholds:** All correlation thresholds and delay search parameters are defined as named constants in `src/constants.rs` for maintainability and tuning.
*   **Display:** Results are shown in console output with confidence metrics (as percentages), and estimates are integrated into plot legends as "Delay: X.Xms(c:XX%)" for `_GyroVsUnfilt_stacked.png`, `_Gyro_Spectrums_comparative.png`, and `_Gyro_PSD_comparative.png` outputs.

### Function API Structure
*   **`calculate_filtering_delay`:** Core single-axis delay calculation returning `Result<f32, DelayCalculationError>`
*   **`calculate_average_filtering_delay`:** Multi-axis averaging returning `Option<f32>` with console output
*   **`calculate_average_filtering_delay_comparison`:** Enhanced analysis returning `Option<(Option<f32>, Vec<DelayResult>)>` with detailed result structures and diagnostic information

This delay measurement approach provides reliable identification of filtering phase lag characteristics in flight controller systems, with enhanced precision through interpolation techniques and robust error handling for various signal conditions. The implementation focuses on a single, well-tested cross-correlation method rather than experimental multi-method approaches.

**Step Response Differences from Other Tools:**

**System Identification Approach:**
The approach used by all three implementations (Rust, PIDtoolbox/Matlab, PlasmaTree/Python) is described in detail in the "Step Response Calculation" section. In summary, they employ **non-parametric system identification** via Wiener deconvolution to extract the system's impulse response, which is then converted to step response via integration.

*   **Compared to `PTstepcalc.m` (PIDtoolbox/Matlab):**
    *   **Deconvolution Method:** Both use Wiener deconvolution with a regularization term.
    *   **Windowing:** Rust uses Tukey (`TUKEY_ALPHA`) on input/output before FFT; Matlab uses Hann.
    *   **Smoothing:** Rust has optional initial gyro smoothing (`INITIAL_GYRO_SMOOTHING_WINDOW`) and mandatory post-average smoothing (`POST_AVERAGING_SMOOTHING_WINDOW`). Matlab smooths raw gyro input upfront.
    *   **Normalization/Y-Correction:**
        *   Rust: Optional individual response Y-correction (normalize by own steady-state mean, `APPLY_INDIVIDUAL_RESPONSE_Y_CORRECTION`, `Y_CORRECTION_MIN_UNNORMALIZED_MEAN_ABS`) *then* a final normalization of the *averaged* response to target 1.0 (within `FINAL_NORMALIZED_STEADY_STATE_TOLERANCE`).
        *   Matlab: Optional Y-correction on individual responses by calculating an offset to make the mean 1.0.
    *   **Quality Control (QC):** Both apply QC to individual responses based on steady-state characteristics. Rust uses `NORMALIZED_STEADY_STATE_MIN_VAL`, `NORMALIZED_STEADY_STATE_MAX_VAL`, and optionally `NORMALIZED_STEADY_STATE_MEAN_MIN`, `NORMALIZED_STEADY_STATE_MEAN_MAX`. Matlab uses `min(steadyStateResp) > 0.5 && max(steadyStateResp) < 3`.
    *   **Output:** Rust can plot low/high/combined responses based on `setpoint_threshold` if `--dps` is used. Matlab stacks all valid responses.

*   **Compared to `PID-Analyzer.py` (PlasmaTree/Python):**
    *   **Deconvolution Method:** PlasmaTree also uses Wiener deconvolution (`Trace.wiener_deconvolution`). It includes a signal-to-noise ratio (`sn`) term in the denominator, which is derived from a `cutfreq` parameter and smoothed, effectively acting as a frequency-dependent regularization. The Rust version uses a simpler constant regularization term (`0.0001`).
    *   **Windowing:** PlasmaTree uses a Hanning window (`np.hanning`) by default (or Tukey via `Trace.tukeywin` with `Trace.tuk_alpha`) applied to input and output segments before deconvolution. Rust uses a Tukey window.
    *   **Input for Deconvolution:** PlasmaTree calculates an `input` signal by `pid_in(data['p_err'], data['gyro'], data['P'])` which attempts to reconstruct the setpoint as seen by the PID loop. The Rust version directly uses the logged setpoint values.
    *   **Smoothing:** PlasmaTree does not explicitly mention an initial gyro smoothing step like Rust's `INITIAL_GYRO_SMOOTHING_WINDOW`. For the final averaged response, PlasmaTree's `weighted_mode_avr` uses a 2D histogram and Gaussian smoothing (`gaussian_filter1d`) on this histogram to find the "mode" response, which is different from Rust's direct moving average on the time-domain averaged response.
    *   **Normalization/Y-Correction:**
        *   PlasmaTree's `stack_response` calculates `delta_resp` (step responses) by cumulative sum of deconvolved impulse responses. There isn't an explicit "Y-correction" step for individual responses in the same way Rust or PIDtoolbox does (i.e., normalizing each to its own steady-state mean of 1). The `weighted_mode_avr` function, which produces `resp_low` and `resp_high`, aims to find the most common trace shape from a collection of responses, which inherently handles variations. The resulting averaged traces are plotted as they are, typically ranging from 0 to some peak value.
    *   **Quality Control (QC):** PlasmaTree has a `resp_quality` metric calculated based on the deviation of individual responses from an initial average. It also uses a `toolow_mask` (based on input magnitude `max_in < 20`) to discard responses from very low inputs. These are used as weights or masks in `thr_response` and `weighted_mode_avr`. This is different from Rust's direct steady-state value checks.
    *   **Averaging:**
        *   Rust: `average_responses` performs a weighted average of QC'd responses.
        *   PlasmaTree: `weighted_mode_avr` uses a 2D histogram of all response traces over time vs. amplitude, smooths this histogram, and then calculates a weighted average based on the smoothed histogram's "mode" to determine the final average step response. This is a more complex way to find a representative trace.
    *   **Output:** Both can plot low and high input responses based on a threshold (`Trace.threshold` in PlasmaTree, `setpoint_threshold` in Rust).

In summary, this Rust implementation offers a detailed and configurable analysis pipeline for flight controller performance evaluation. It draws conceptual parallels with existing tools but provides its own specific algorithms and parameterization. The modular design and centralized configuration system make it adaptable for various analysis requirements. PlasmaTree's approach is also sophisticated, particularly in its use of histogram-based averaging for the final step response.
