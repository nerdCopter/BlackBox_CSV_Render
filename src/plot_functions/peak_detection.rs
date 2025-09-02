// src/plot_functions/peak_detection.rs

use crate::constants::{
    ENABLE_WINDOW_PEAK_DETECTION, MAX_PEAKS_TO_LABEL, MIN_PEAK_SEPARATION_HZ,
    MIN_SECONDARY_PEAK_RATIO, PEAK_DETECTION_WINDOW_RADIUS, PEAK_LABEL_MIN_AMPLITUDE,
    SPECTRUM_NOISE_FLOOR_HZ,
};

/// Detects and sorts peaks in spectrum data for labeling (D-term plots)
/// Returns a vector of (frequency, amplitude) tuples for peaks that should be labeled
/// Uses the exact same logic as gyro plots, including noise floor filtering to avoid low-frequency artifacts
pub fn find_and_sort_peaks(
    series_data: &[(f64, f64)],
    primary_peak_info: Option<(f64, f64)>,
    axis_name_str: &str,
    spectrum_type_str: &str,
) -> Vec<(f64, f64)> {
    let mut peaks_to_plot: Vec<(f64, f64)> = Vec::new();

    if let Some((peak_freq, peak_amp)) = primary_peak_info {
        if peak_amp > PEAK_LABEL_MIN_AMPLITUDE {
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
            if freq >= SPECTRUM_NOISE_FLOOR_HZ
                && is_potential_peak
                && amp > PEAK_LABEL_MIN_AMPLITUDE
            {
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
            if !too_close_to_existing && s_amp > PEAK_LABEL_MIN_AMPLITUDE {
                // Ensure it's still above min amp
                peaks_to_plot.push((s_freq, s_amp));
            }
        }
    }

    peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !peaks_to_plot.is_empty() {
        let (main_freq, main_amp) = peaks_to_plot[0];
        println!(
            "  {axis_name_str} {spectrum_type_str} D-term Spectrum: Primary Peak value {main_amp:.2} at {main_freq} Hz"
        );
        for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
            println!("    Subordinate Peak {}: {:.2} at {freq} Hz", idx + 1, amp);
        }
    } else {
        println!(
            "  {axis_name_str} {spectrum_type_str} D-term Spectrum: No significant peaks found."
        );
    }
    peaks_to_plot
}
