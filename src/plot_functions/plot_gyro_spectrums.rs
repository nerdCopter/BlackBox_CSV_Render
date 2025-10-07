// src/plot_functions/plot_gyro_spectrums.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT, ENABLE_WINDOW_PEAK_DETECTION,
    FILTERED_GYRO_MIN_THRESHOLD, LINE_WIDTH_PLOT, MAX_PEAKS_TO_LABEL, MIN_PEAK_SEPARATION_HZ,
    MIN_SECONDARY_PEAK_RATIO, PEAK_DETECTION_WINDOW_RADIUS, PEAK_LABEL_MIN_AMPLITUDE,
    SPECTRUM_NOISE_FLOOR_HZ, SPECTRUM_Y_AXIS_FLOOR, SPECTRUM_Y_AXIS_HEADROOM_FACTOR, TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_analysis::filter_delay;
use crate::data_analysis::filter_response;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{
    draw_dual_spectrum_plot, AxisSpectrum, PlotConfig, PlotSeries, CUTOFF_LINE_DOTTED_PREFIX,
    CUTOFF_LINE_PREFIX,
};
use crate::types::AllFFTData;
use plotters::style::RGBColor;

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro spectrums.
/// Now includes filter response curve overlays based on header metadata.
pub fn plot_gyro_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    header_metadata: Option<&[(String, String)]>,
    show_butterworth: bool,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_Gyro_Spectrums_comparative.png");
    let plot_type_name = "Gyro Spectrums";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro Spectrum Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Calculate filtering delay using enhanced cross-correlation
    let delay_analysis =
        filter_delay::calculate_average_filtering_delay_comparison(log_data, sr_value);
    let delay_comparison_results = if !delay_analysis.results.is_empty() {
        Some(delay_analysis.results)
    } else {
        None
    };

    let filter_config = header_metadata.map(filter_response::parse_filter_config);

    // Extract gyro rate once for proper Nyquist calculation
    let gyro_rate_hz = filter_response::extract_gyro_rate(header_metadata).unwrap_or(8000.0); // Default 8kHz

    // Extract dynamic notch range for graphical visualization
    let dynamic_notch_range = filter_response::extract_dynamic_notch_range(header_metadata);

    let mut all_fft_raw_data: AllFFTData = Default::default();
    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;
    let mut overall_max_y_amplitude = 0.0f64;

    fn find_and_sort_peaks(
        series_data: &[(f64, f64)],
        primary_peak_info: Option<(f64, f64)>,
        axis_name_str: &str,
        spectrum_type_str: &str,
    ) -> Vec<(f64, f64)> {
        // For filtered gyro data, use intelligent threshold checking
        // Based on user feedback: 4k, 2.1k peaks are reasonable, but <1.9k peaks are not meaningful
        let is_filtered = spectrum_type_str == "Filtered";
        let amplitude_threshold = if is_filtered {
            FILTERED_GYRO_MIN_THRESHOLD // Use constant from constants.rs
        } else {
            PEAK_LABEL_MIN_AMPLITUDE // 1k threshold for unfiltered gyro
        };

        if is_filtered {
            if let Some((_, peak_amp)) = primary_peak_info {
                if peak_amp <= amplitude_threshold {
                    let formatted_peak = if peak_amp >= 1000.0 {
                        format!("{:.1}k", peak_amp / 1000.0)
                    } else {
                        format!("{:.1}", peak_amp)
                    };
                    let formatted_threshold = format!("{:.1}k", amplitude_threshold / 1000.0);
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

                if freq >= SPECTRUM_NOISE_FLOOR_HZ && is_potential_peak && amp > amplitude_threshold
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
                if !too_close_to_existing && s_amp > amplitude_threshold {
                    // Ensure it's still above the intelligent threshold for filtered data or min amp for unfiltered
                    peaks_to_plot.push((s_freq, s_amp));
                }
            }
        }

        peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !peaks_to_plot.is_empty() {
            let (main_freq, main_amp) = peaks_to_plot[0];
            println!(
                "  {axis_name_str} {spectrum_type_str} Gyro Spectrum: Primary Peak amplitude {main_amp:.0} at {:.1} Hz",
                main_freq
            );
            for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
                println!(
                    "    Subordinate Peak: {}: {:.0} at {:.1} Hz",
                    idx + 1,
                    amp,
                    freq
                );
            }
        } else {
            println!(
                "  {axis_name_str} {spectrum_type_str} Gyro Spectrum: No significant peaks found."
            );
        }
        peaks_to_plot
    }

    // Iterate safely over the minimum of AXIS_NAMES.len() and the fixed array size
    let axis_count = AXIS_NAMES.len().min(3); // gyro arrays are [Option<f64>; 3]
    for axis_idx in 0..axis_count {
        let axis_name = AXIS_NAMES[axis_idx];
        let mut unfilt_samples: Vec<f32> = Vec::new();
        let mut filt_samples: Vec<f32> = Vec::new();

        for row in log_data {
            if let (Some(unfilt_val), Some(filt_val)) =
                (row.gyro_unfilt[axis_idx], row.gyro[axis_idx])
            {
                unfilt_samples.push(unfilt_val as f32);
                filt_samples.push(filt_val as f32);
            }
        }

        if unfilt_samples.is_empty() || filt_samples.is_empty() {
            println!("  No unfiltered or filtered gyro data for {axis_name} axis. Skipping spectrum peak analysis.");
            continue;
        }

        let min_len = unfilt_samples.len().min(filt_samples.len());
        if min_len == 0 {
            println!("  Not enough common gyro data for {axis_name} axis. Skipping spectrum peak analysis.");
            continue;
        }

        let unfilt_samples_slice = &unfilt_samples[0..min_len];
        let filt_samples_slice = &filt_samples[0..min_len];
        let window_func = calc_step_response::tukeywin(min_len, TUKEY_ALPHA);

        let fft_padded_len = min_len.next_power_of_two();
        let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
        padded_unfilt
            .slice_mut(s![0..min_len])
            .assign(&(&Array1::from_vec(unfilt_samples_slice.to_vec()) * &window_func));
        let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
        padded_filt
            .slice_mut(s![0..min_len])
            .assign(&(&Array1::from_vec(filt_samples_slice.to_vec()) * &window_func));

        let unfilt_spec = fft_utils::fft_forward(&padded_unfilt);
        let filt_spec = fft_utils::fft_forward(&padded_filt);

        if unfilt_spec.is_empty() || filt_spec.is_empty() {
            println!("  FFT computation failed or resulted in empty spectrums for {axis_name} axis. Skipping spectrum peak analysis.");
            continue;
        }

        let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
        let mut filt_series_data: Vec<(f64, f64)> = Vec::new();
        let freq_step = sr_value / fft_padded_len as f64;
        let num_unique_freqs = if fft_padded_len % 2 == 0 {
            fft_padded_len / 2 + 1
        } else {
            fft_padded_len.div_ceil(2)
        };

        let mut primary_peak_unfilt: Option<(f64, f64)> = None;
        let mut primary_peak_filt: Option<(f64, f64)> = None;

        for i in 0..num_unique_freqs {
            let freq_val = i as f64 * freq_step;
            let amp_unfilt = unfilt_spec[i].norm() as f64;
            let amp_filt = filt_spec[i].norm() as f64;
            unfilt_series_data.push((freq_val, amp_unfilt));
            filt_series_data.push((freq_val, amp_filt));

            if freq_val >= SPECTRUM_NOISE_FLOOR_HZ {
                if amp_unfilt > primary_peak_unfilt.map_or(0.0, |(_, amp)| amp) {
                    primary_peak_unfilt = Some((freq_val, amp_unfilt));
                }
                if amp_filt > primary_peak_filt.map_or(0.0, |(_, amp)| amp) {
                    primary_peak_filt = Some((freq_val, amp_filt));
                }
            }
        }

        let unfilt_peaks_for_plot = find_and_sort_peaks(
            &unfilt_series_data,
            primary_peak_unfilt,
            axis_name,
            "Unfiltered",
        );
        let filt_peaks_for_plot =
            find_and_sort_peaks(&filt_series_data, primary_peak_filt, axis_name, "Filtered");

        let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step).max(0.0) as usize;
        let max_amp_after_noise_floor_unfilt = unfilt_series_data
            .get(noise_floor_sample_idx..)
            .map_or(0.0, |data_slice| {
                data_slice
                    .iter()
                    .map(|&(_, amp)| amp)
                    .fold(0.0f64, |max_val, amp| max_val.max(amp))
            });
        let max_amp_after_noise_floor_filt =
            filt_series_data
                .get(noise_floor_sample_idx..)
                .map_or(0.0, |data_slice| {
                    data_slice
                        .iter()
                        .map(|&(_, amp)| amp)
                        .fold(0.0f64, |max_val, amp| max_val.max(amp))
                });

        let y_max_unfilt_for_range = SPECTRUM_Y_AXIS_FLOOR
            .max(max_amp_after_noise_floor_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);
        let y_max_filt_for_range = SPECTRUM_Y_AXIS_FLOOR
            .max(max_amp_after_noise_floor_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);

        all_fft_raw_data[axis_idx] = Some((
            unfilt_series_data,
            unfilt_peaks_for_plot,
            filt_series_data,
            filt_peaks_for_plot,
        ));
        global_max_y_unfilt = global_max_y_unfilt.max(y_max_unfilt_for_range);
        global_max_y_filt = global_max_y_filt.max(y_max_filt_for_range);
    }

    overall_max_y_amplitude = overall_max_y_amplitude
        .max(global_max_y_unfilt)
        .max(global_max_y_filt);
    if overall_max_y_amplitude < SPECTRUM_Y_AXIS_FLOOR {
        overall_max_y_amplitude = SPECTRUM_Y_AXIS_FLOOR;
    }

    draw_dual_spectrum_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        if let Some((unfilt_series_data, unfilt_peaks, filt_series_data, filt_peaks)) =
            all_fft_raw_data[axis_index].as_ref().cloned()
        {
            let max_freq_val = sr_value / 2.0;
            let x_range = 0.0..max_freq_val * 1.05;
            let y_range_for_all_clone = 0.0..overall_max_y_amplitude;

            // Build series in Betaflight signal path order:
            // 1. Unfiltered Gyro (raw)
            // 2. Dynamic Notch (first applied filter)
            // 3. RPM Filter (if configured - not yet implemented)
            // 4. Gyro LPF1 (static or dynamic)
            // 5. Gyro LPF2 (static)
            // 6. IMUF (if configured)
            let mut unfilt_plot_series = vec![];

            // 1. Unfiltered Gyro (raw data)
            unfilt_plot_series.push(PlotSeries {
                data: unfilt_series_data,
                label: {
                    // Check if dynamic LPF is being used to enhance the legend
                    if let Some(ref config) = filter_config {
                        let (has_dynamic, min_cutoff, max_cutoff) =
                            filter_response::check_gyro_dynamic_lpf_usage(config);
                        if has_dynamic {
                            format!(
                                "Unfiltered Gyro (Dynamic LPF {:.0}-{:.0}Hz)",
                                min_cutoff, max_cutoff
                            )
                        } else {
                            "Unfiltered Gyro".to_string()
                        }
                    } else {
                        "Unfiltered Gyro".to_string()
                    }
                },
                color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                stroke_width: LINE_WIDTH_PLOT,
            });

            // 2. Dynamic Notch (second in signal path - if configured)
            // Check if dynamic notch applies to this axis (Emuflight can exclude Yaw)
            let dynamic_notch_config = dynamic_notch_range.as_ref();
            let show_dynamic_notch = if let Some(config) = dynamic_notch_config {
                // axis_index: 0=Roll, 1=Pitch, 2=Yaw
                if axis_index == 2 && !config.applies_to_yaw {
                    false // Skip Yaw if dynamic notch is RP-only
                } else {
                    true
                }
            } else {
                false
            };

            // 3. RPM Filter would go here (not yet implemented)

            // 4-6. Gyro LPF1, LPF2, IMUF filters
            // Add filter response curves to unfiltered plot if available
            if let Some(ref config) = filter_config {
                // Use gyro rate for Nyquist, not logging rate - filters operate at gyro frequency
                let max_freq = gyro_rate_hz / 2.0; // Proper gyro Nyquist frequency
                let num_points = 1000; // More points for smooth curves

                // Generate individual filter response curves for this axis
                let filter_curves = filter_response::generate_individual_filter_curves(
                    &config.gyro[axis_index],
                    max_freq,
                    num_points,
                    show_butterworth,
                );

                // Filter colors matching Betaflight signal path order:
                // LPF1 (4th filter) - Crimson/Red
                // LPF2 (5th filter) - Orange
                // IMUF (6th filter) - Brown/Dark Red
                let filter_colors = [
                    RGBColor(220, 20, 60), // Crimson for LPF1 (first in filter chain after notches)
                    RGBColor(255, 140, 0), // Dark orange for LPF2
                    RGBColor(139, 69, 19), // Saddle brown for IMUF
                ];

                for (curve_idx, (label, curve_data, cutoff_hz_ref)) in
                    filter_curves.iter().enumerate()
                {
                    if !curve_data.is_empty() {
                        // Show filter response as a normalized curve overlaid on the spectrum
                        // Use a fixed amplitude scale that makes the cutoff frequency visible
                        let filter_curve_amplitude = overall_max_y_amplitude * 0.3; // 30% of max spectrum height
                        let filter_curve_offset = overall_max_y_amplitude * 0.05; // Offset from bottom

                        let scaled_response: Vec<(f64, f64)> = curve_data
                            .iter()
                            // Keep overlay within the plotted spectrum range
                            .filter(|(freq, _)| *freq <= max_freq_val)
                            .map(|(freq, response)| {
                                // Scale response from [0,1] to [offset, offset + amplitude]
                                let scaled_amplitude =
                                    filter_curve_offset + (response * filter_curve_amplitude);
                                (*freq, scaled_amplitude)
                            })
                            .collect();

                        // Create filter response series - use gray for per-stage curves
                        let curve_color = if label.contains("per-stage") {
                            RGBColor(128, 128, 128) // Gray for per-stage PT1 cutoffs
                        } else {
                            filter_colors[curve_idx % filter_colors.len()] // Standard colors for combined response curves
                        };

                        unfilt_plot_series.push(PlotSeries {
                            data: scaled_response,
                            label: label.clone(),
                            color: curve_color,
                            stroke_width: 2,
                        });

                        // Add vertical cutoff indicator line (no legend entry)
                        let cutoff_hz = *cutoff_hz_ref;
                        if !cutoff_hz.is_finite() {
                            continue;
                        }

                        // Use dotted line and different color for IMUF per-stage cutoffs
                        let (cutoff_prefix, cutoff_color) = if label.contains("per-stage") {
                            // Per-stage cutoffs: dotted line with muted gray color to show Butterworth correction
                            (CUTOFF_LINE_DOTTED_PREFIX, RGBColor(128, 128, 128))
                        // Gray for per-stage values
                        } else {
                            // Combined cutoffs: solid line with filter color to show effective response
                            (
                                CUTOFF_LINE_PREFIX,
                                filter_colors[curve_idx % filter_colors.len()],
                            )
                        };

                        unfilt_plot_series.push(PlotSeries {
                            data: vec![(cutoff_hz, 0.0), (cutoff_hz, overall_max_y_amplitude)],
                            label: format!("{}{}", cutoff_prefix, cutoff_hz), // Special prefix to avoid legend
                            color: cutoff_color,
                            stroke_width: 1,
                        });
                    }
                }
            }

            // Add dynamic notch legend entry AFTER filter curves (correct signal path order)
            // Dynamic notch comes after unfiltered gyro but before LPF filters in the legend
            if show_dynamic_notch {
                if let Some(config) = dynamic_notch_config {
                    unfilt_plot_series.push(PlotSeries {
                        data: vec![], // No data - just for legend with matching color
                        label: format!(
                            "Dynamic Notch: {} notch{}, Q: {:.0}, range: {:.0}-{:.0}Hz{}",
                            config.notch_count,
                            if config.notch_count > 1 { "es" } else { "" },
                            config.q_factor,
                            config.min_hz,
                            config.max_hz,
                            if !config.applies_to_yaw {
                                " (RP only)"
                            } else {
                                ""
                            }
                        ),
                        color: RGBColor(147, 112, 219), // Medium purple - matches shading
                        stroke_width: LINE_WIDTH_PLOT,  // Same as other series
                    });
                }
            }

            let filt_plot_series = vec![PlotSeries {
                data: filt_series_data,
                label: if let Some(ref results) = delay_comparison_results {
                    // Show comparison of both methods if available - NO AVERAGING
                    let mut method_strings = Vec::new();
                    for result in results.iter() {
                        if let Some(freq) = result.frequency_hz {
                            method_strings.push(format!(
                                "{}: {:.1}ms@{:.0}Hz(c:{:.0}%)",
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown",
                                },
                                result.delay_ms,
                                freq,
                                result.confidence * 100.0
                            ));
                        } else {
                            method_strings.push(format!(
                                "{}: {:.1}ms(c:{:.0}%)",
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown",
                                },
                                result.delay_ms,
                                result.confidence * 100.0
                            ));
                        }
                    }
                    if method_strings.is_empty() {
                        "Filtered Gyro".to_string()
                    } else {
                        format!("Filtered Gyro - {}", method_strings.join(" vs "))
                    }
                } else {
                    "Filtered Gyro".to_string()
                },
                color: *COLOR_GYRO_VS_UNFILT_FILT,
                stroke_width: LINE_WIDTH_PLOT,
            }];

            // Create dynamic notch frequency range visualization if configured
            // Only show on axes where dynamic notch applies (respect RP-only setting)
            let frequency_ranges = if show_dynamic_notch {
                if let Some(config) = dynamic_notch_config {
                    use crate::plot_framework::FrequencyRange;
                    use plotters::style::RGBColor;

                    let label = format!(
                        "Dynamic Notch: {} notch{}, Q: {:.0}, range: {:.0}-{:.0}Hz{}",
                        config.notch_count,
                        if config.notch_count > 1 { "es" } else { "" },
                        config.q_factor,
                        config.min_hz,
                        config.max_hz,
                        if !config.applies_to_yaw {
                            " (RP only)"
                        } else {
                            ""
                        }
                    );

                    Some(vec![FrequencyRange {
                        min_hz: config.min_hz,
                        max_hz: config.max_hz,
                        color: RGBColor(147, 112, 219), // Medium purple
                        opacity: 0.15,                  // Semi-transparent
                        label,
                    }])
                } else {
                    None
                }
            } else {
                None
            };

            let unfiltered_plot_config = Some(PlotConfig {
                title: format!("{} Unfiltered Gyro Spectrum", AXIS_NAMES[axis_index]),
                x_range: x_range.clone(),
                y_range: y_range_for_all_clone.clone(),
                series: unfilt_plot_series,
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: unfilt_peaks,
                // MINIMAL CHANGE: Initialize new fields to Some for linear amplitude plots
                peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                peak_label_format_string: Some("{:.0}".to_string()),
                frequency_ranges, // Dynamic notch only on unfiltered plot
            });

            let filtered_plot_config = Some(PlotConfig {
                title: format!("{} Filtered Gyro Spectrum", AXIS_NAMES[axis_index]),
                x_range,
                y_range: y_range_for_all_clone,
                series: filt_plot_series,
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: filt_peaks,
                // MINIMAL CHANGE: Initialize new fields to Some for linear amplitude plots
                peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                peak_label_format_string: Some("{:.0}".to_string()),
                frequency_ranges: None, // No dynamic notch on filtered plot (already applied)
            });

            Some(AxisSpectrum {
                unfiltered: unfiltered_plot_config,
                filtered: filtered_plot_config,
            })
        } else {
            Some(AxisSpectrum {
                unfiltered: None,
                filtered: None,
            })
        }
    })
}

// src/plot_functions/plot_gyro_spectrums.rs
