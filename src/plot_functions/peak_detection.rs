// src/plot_functions/peak_detection.rs

use crate::constants::{
    ENABLE_FILTERED_D_TERM_PEAK_DETECTION, ENABLE_WINDOW_PEAK_DETECTION, MAX_PEAKS_TO_LABEL,
    MIN_PEAK_SEPARATION_HZ, MIN_SECONDARY_PEAK_RATIO, PEAK_DETECTION_WINDOW_RADIUS,
    PEAK_LABEL_MIN_AMPLITUDE, SPECTRUM_NOISE_FLOOR_HZ,
};

/// Determines if spectrum data is too flat for meaningful peak detection
/// This is primarily used for filtered D-term data which can be very flat after filtering
fn is_data_too_flat(series_data: &[(f64, f64)], amplitude_threshold: f64) -> bool {
    if series_data.len() < 10 {
        return true; // Too little data
    }

    // Calculate dynamic range: difference between max and min values
    let amplitudes: Vec<f64> = series_data.iter().map(|(_, amp)| *amp).collect();
    let max_amp = amplitudes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_amp = amplitudes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let dynamic_range = max_amp - min_amp;

    // For filtered D-term data, be more aggressive about detecting flatness
    // For dB scale (PSD plots), consider data flat if dynamic range < 30 dB
    // For linear scale (spectrum plots), consider data flat if dynamic range < 5x amplitude threshold
    let flatness_threshold = if amplitude_threshold < 0.0 {
        // dB scale (negative threshold like -60 dB) - be more aggressive for filtered D-term
        30.0 // Less than 30 dB dynamic range is considered flat for filtered D-term
    } else {
        // Linear scale (positive threshold like 1000.0) - be more aggressive for filtered D-term
        amplitude_threshold * 5.0 // Dynamic range must be at least 5x the threshold (was 10x)
    };

    let is_flat = dynamic_range < flatness_threshold;

    if is_flat {
        println!(
            "    Dynamic range: {:.2}, threshold: {:.2} - flagged as too flat",
            dynamic_range, flatness_threshold
        );
    }

    is_flat
}

/// Detects and sorts peaks in spectrum data for labeling (D-term plots)
/// Returns a vector of (frequency, amplitude) tuples for peaks that should be labeled
/// Uses the exact same logic as gyro plots, including noise floor filtering to avoid low-frequency artifacts
pub fn find_and_sort_peaks(
    series_data: &[(f64, f64)],
    primary_peak_info: Option<(f64, f64)>,
    axis_name_str: &str,
    spectrum_type_str: &str,
) -> Vec<(f64, f64)> {
    find_and_sort_peaks_with_threshold(
        series_data,
        primary_peak_info,
        axis_name_str,
        spectrum_type_str,
        PEAK_LABEL_MIN_AMPLITUDE,
    )
}

/// Detects and sorts peaks with configurable amplitude threshold
/// For PSD plots (dB scale), use PSD_PEAK_LABEL_MIN_VALUE_DB
/// For linear spectrum plots, use PEAK_LABEL_MIN_AMPLITUDE
/// Includes flatness detection to skip peak finding on near flat-line filtered D-term data
pub fn find_and_sort_peaks_with_threshold(
    series_data: &[(f64, f64)],
    primary_peak_info: Option<(f64, f64)>,
    axis_name_str: &str,
    spectrum_type_str: &str,
    amplitude_threshold: f64,
) -> Vec<(f64, f64)> {
    // Check if filtered D-term peak detection is disabled globally
    if spectrum_type_str.contains("Filtered D-term") && !ENABLE_FILTERED_D_TERM_PEAK_DETECTION {
        println!("  {axis_name_str} {spectrum_type_str}: Peak detection disabled for filtered D-term data.");
        return Vec::new();
    }

    // Check if this is filtered D-term data and if it's too flat for meaningful peak detection
    if spectrum_type_str.contains("Filtered D-term")
        && is_data_too_flat(series_data, amplitude_threshold)
    {
        println!(
            "  {axis_name_str} {spectrum_type_str}: Data too flat for meaningful peak detection."
        );
        return Vec::new();
    }

    let mut peaks_to_plot: Vec<(f64, f64)> = Vec::new();

    if let Some((peak_freq, peak_amp)) = primary_peak_info {
        if peak_amp > amplitude_threshold {
            peaks_to_plot.push((peak_freq, peak_amp));
        }
    }

    if series_data.len() > 2 && peaks_to_plot.len() < MAX_PEAKS_TO_LABEL {
        let mut candidate_secondary_peaks: Vec<(f64, f64)> = Vec::new();
        // Iterate from the second point to the second-to-last point,
        // as peak detection logic needs at least one point on each side.
        for j in 1..(series_data.len() - 1) {
            let (freq, amp) = series_data[j];

            let is_potential_peak = {
                // Assign directly from the block's result
                if ENABLE_WINDOW_PEAK_DETECTION {
                    let w = PEAK_DETECTION_WINDOW_RADIUS;
                    // Check if a full window can be formed around j.
                    // j must be at least w points from the start,
                    // and j must be at least w points from the end (so j+w is a valid index).
                    if j >= w && j + w < series_data.len() {
                        // Full window logic
                        let mut ge_left_in_window = true;
                        for k_offset in 1..=w {
                            // series_data[j - k_offset] is valid because j >= w >= k_offset
                            if amp < series_data[j - k_offset].1 {
                                ge_left_in_window = false;
                                break;
                            }
                        }

                        let mut gt_right_in_window = true;
                        if ge_left_in_window {
                            // Optimization: only check right if left is good
                            for k_offset in 1..=w {
                                // series_data[j + k_offset] is valid because j + w < series_data.len()
                                // and k_offset <= w
                                if amp <= series_data[j + k_offset].1 {
                                    gt_right_in_window = false;
                                    break;
                                }
                            }
                        }
                        ge_left_in_window && gt_right_in_window // Return this value for this path
                    } else {
                        // Fallback for edges where a full window isn't possible.
                        // The loop for j ensures j-1 and j+1 are always valid.
                        let prev_amp = series_data[j - 1].1;
                        let next_amp = series_data[j + 1].1;
                        // Using rightmost point of plateau for consistency with window logic's tendency
                        amp >= prev_amp && amp > next_amp // Return this value for this path
                    }
                } else {
                    // Original 3-point logic (leftmost point of plateau or sharp peak).
                    // The loop for j ensures j-1 and j+1 are always valid.
                    let prev_amp = series_data[j - 1].1;
                    let next_amp = series_data[j + 1].1;
                    amp > prev_amp && amp >= next_amp // Return this value for this path
                }
            }; // End of block assignment to is_potential_peak

            // Apply noise floor filtering to avoid low-frequency artifacts (like 1Hz peaks)
            if freq >= SPECTRUM_NOISE_FLOOR_HZ && is_potential_peak && amp > amplitude_threshold {
                let mut is_valid_for_secondary_consideration = true;
                if let Some((primary_freq, primary_amp_val)) = primary_peak_info {
                    if freq == primary_freq && amp == primary_amp_val {
                        // Don't re-add the primary peak
                        is_valid_for_secondary_consideration = false;
                    } else {
                        is_valid_for_secondary_consideration = (amp
                            >= primary_amp_val * MIN_SECONDARY_PEAK_RATIO)
                            && ((freq - primary_freq).abs() > MIN_PEAK_SEPARATION_HZ);
                    }
                }
                // If no primary_peak_info, is_valid_for_secondary_consideration remains true (as long as it's a potential peak and above min amplitude)
                if is_valid_for_secondary_consideration {
                    candidate_secondary_peaks.push((freq, amp));
                }
            }
        }

        candidate_secondary_peaks
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (s_freq, s_amp) in candidate_secondary_peaks {
            if peaks_to_plot.len() >= MAX_PEAKS_TO_LABEL {
                break;
            }
            let mut too_close_to_existing = false;
            for (p_freq, _) in &peaks_to_plot {
                if (s_freq - *p_freq).abs() < MIN_PEAK_SEPARATION_HZ {
                    too_close_to_existing = true;
                    break;
                }
            }
            if !too_close_to_existing && s_amp > amplitude_threshold {
                // Ensure it's still above min amp
                peaks_to_plot.push((s_freq, s_amp));
            }
        }
    }

    peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !peaks_to_plot.is_empty() {
        let (main_freq, main_amp) = peaks_to_plot[0];
        println!(
            "  {axis_name_str} {spectrum_type_str}: Primary Peak value {main_amp:.2} at {main_freq} Hz"
        );
        for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
            println!("    Subordinate Peak {}: {:.2} at {freq} Hz", idx + 1, amp);
        }
    } else {
        println!("  {axis_name_str} {spectrum_type_str}: No significant peaks found.");
    }
    peaks_to_plot
}
