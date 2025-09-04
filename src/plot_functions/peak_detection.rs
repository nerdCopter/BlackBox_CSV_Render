// src/plot_functions/peak_detection.rs

use crate::constants::{
    ENABLE_WINDOW_PEAK_DETECTION, FILTERED_D_TERM_MIN_THRESHOLD, MAX_PEAKS_TO_LABEL,
    MIN_PEAK_SEPARATION_HZ, MIN_SECONDARY_PEAK_DB, MIN_SECONDARY_PEAK_RATIO,
    PEAK_DETECTION_WINDOW_RADIUS, SPECTRUM_NOISE_FLOOR_HZ,
};

/// Formats large numbers with "k" and "M" notation for better readability.
///
/// # Arguments
/// * `value` - The numeric value to format
///
/// # Returns
/// A formatted string with appropriate unit suffixes:
/// * Values >= 1,000,000: formatted as "X.XM" (e.g., "1.5M")  
/// * Values >= 1,000: formatted as "X.Xk" (e.g., "30.1k")
/// * Values < 1,000: formatted as "X.X" (e.g., "123.4")
fn format_value_with_k(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value >= 1000.0 {
        format!("{:.1}k", value / 1000.0)
    } else {
        format!("{:.1}", value)
    }
}

/// Detects and sorts peaks with configurable amplitude threshold and intelligent filtering.
///
/// This function implements scale-aware peak detection that adapts thresholds based on data type
/// and range. For filtered D-term data, it uses intelligent threshold checking to avoid
/// reporting peaks that are below meaningful amplitudes for the specific data type.
///
/// # Arguments
/// * `series_data` - The frequency-amplitude data points to analyze
/// * `primary_peak_info` - Optional primary peak information for threshold validation  
/// * `axis_name_str` - Name of the axis for logging (e.g., "Roll", "Pitch", "Yaw")
/// * `spectrum_type_str` - Type of spectrum for adaptive thresholding (e.g., "Filtered D-term")
/// * `amplitude_threshold` - Base amplitude threshold (adapted based on data type)
///
/// # Returns
/// Vector of (frequency, amplitude) tuples for peaks above the threshold, sorted by amplitude
///
/// # Threshold Adaptation
/// * PSD plots (dB scale, negative values): Uses provided dB threshold directly
/// * D-term spectrums: Uses `FILTERED_D_TERM_MIN_THRESHOLD` for realistic scale-aware filtering
/// * Other spectrum types: Uses the original `amplitude_threshold` parameter
///
/// # Notes
/// Includes flatness detection to skip peak finding on near flat-line filtered D-term data
/// where peak detection would not be meaningful.
pub fn find_and_sort_peaks_with_threshold(
    series_data: &[(f64, f64)],
    primary_peak_info: Option<(f64, f64)>,
    axis_name_str: &str,
    spectrum_type_str: &str,
    amplitude_threshold: f64,
) -> Vec<(f64, f64)> {
    // Input validation
    if series_data.is_empty() {
        return Vec::new();
    }

    if axis_name_str.is_empty() || spectrum_type_str.is_empty() {
        return Vec::new();
    }

    if !amplitude_threshold.is_finite() {
        return Vec::new();
    }

    // For filtered D-term data, use intelligent scale-aware threshold checking
    // Instead of fixed thresholds, use percentage-based logic that adapts to data scale
    if spectrum_type_str.contains("Filtered D-term") {
        if let Some((_, peak_amp)) = primary_peak_info {
            // Calculate scale-aware threshold based on the data type and range
            let intelligent_threshold = if amplitude_threshold < 0.0 {
                // PSD plots (dB scale): Use the provided dB threshold
                amplitude_threshold
            } else {
                // Spectrum plots (linear scale): Use percentage-based threshold
                // For D-term data (typically 20k-500k range), require at least 10% of reasonable scale
                // This means ~5k minimum for D-term, but adapts if unfiltered peak is very high
                if spectrum_type_str.contains("D-term") {
                    // D-term spectrums: Use a realistic threshold based on typical D-term scales
                    // Unfiltered D-terms are typically 10M-100M, so filtered peaks below 100k are usually noise
                    FILTERED_D_TERM_MIN_THRESHOLD.max(amplitude_threshold)
                } else {
                    // Other spectrum types: Use the original threshold
                    amplitude_threshold
                }
            };

            if peak_amp <= intelligent_threshold {
                let formatted_peak = if amplitude_threshold < 0.0 {
                    format!("{:.2}", peak_amp)
                } else {
                    format_value_with_k(peak_amp)
                };
                let formatted_threshold = if amplitude_threshold < 0.0 {
                    format!("{:.2}", intelligent_threshold)
                } else {
                    format_value_with_k(intelligent_threshold)
                };
                println!("  {axis_name_str} {spectrum_type_str}: Primary peak ({}) below intelligent threshold ({}) - skipping peak detection.", formatted_peak, formatted_threshold);
                return Vec::new();
            }
        } else {
            // No primary peak found at all
            println!("  {axis_name_str} {spectrum_type_str}: No peaks above threshold - skipping peak detection.");
            return Vec::new();
        }
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
                        // Check if amplitudes are in dB (negative values or spectrum_type contains "dB")
                        let is_db_scale =
                            amplitude_threshold < 0.0 || spectrum_type_str.contains("dB");

                        let amplitude_check = if is_db_scale {
                            // For dB scale: compare using dB difference
                            amp - primary_amp_val >= -MIN_SECONDARY_PEAK_DB
                        } else {
                            // For linear scale: compare using ratio
                            amp >= primary_amp_val * MIN_SECONDARY_PEAK_RATIO
                        };

                        is_valid_for_secondary_consideration = amplitude_check
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
        // Use "k" notation for large spectrum values (but not for dB values)
        let formatted_amp = if amplitude_threshold < 0.0 {
            // dB scale - don't use k notation
            format!("{:.2}", main_amp)
        } else {
            // Linear scale - use k notation for readability
            format_value_with_k(main_amp)
        };
        println!(
            "  {axis_name_str} {spectrum_type_str}: Primary Peak value {} at {:.1} Hz",
            formatted_amp, main_freq
        );
        for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
            let formatted_sub_amp = if amplitude_threshold < 0.0 {
                format!("{:.2}", amp)
            } else {
                format_value_with_k(*amp)
            };
            println!(
                "    Subordinate Peak {}: {} at {:.1} Hz",
                idx + 1,
                formatted_sub_amp,
                freq
            );
        }
    } else {
        println!("  {axis_name_str} {spectrum_type_str}: No significant peaks found.");
    }
    peaks_to_plot
}
