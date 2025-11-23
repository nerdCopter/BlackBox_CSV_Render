// src/plot_functions/plot_d_term_spectrums.rs

use ndarray::Array1;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_D_TERM_FILT, COLOR_D_TERM_UNFILT, PEAK_LABEL_MIN_AMPLITUDE, SPECTRUM_NOISE_FLOOR_HZ,
    SPECTRUM_Y_AXIS_HEADROOM_FACTOR, TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::d_term_delay;
use crate::data_analysis::derivative::calculate_derivative;
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_analysis::filter_response;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{
    draw_dual_spectrum_plot, AxisSpectrum, PlotConfig, PlotSeries, CUTOFF_LINE_DOTTED_PREFIX,
    CUTOFF_LINE_PREFIX,
};
use crate::plot_functions::peak_detection::find_and_sort_peaks_with_threshold;
use plotters::style::RGBColor;

/// Generates a stacked plot with two columns per axis, showing Unfiltered D-term and Filtered D-term spectrums (linear amplitude).
/// Now includes filter response curve overlays based on header metadata.
/// Unfiltered D-term is calculated as the derivative of gyroUnfilt.
/// Filtered D-term uses the flight controller's processed D-term output.
pub fn plot_d_term_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    header_metadata: Option<&[(String, String)]>,
    show_butterworth: bool,
    using_debug_fallback: bool,
    debug_mode_name: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Clone debug mode name to move into closures
    let debug_mode_name_owned = debug_mode_name.map(|s| s.to_string());
    // Input validation
    if log_data.is_empty() {
        return Ok(()); // No data to process
    }

    if root_name.is_empty() {
        return Err("Root name cannot be empty".into());
    }

    let output_file = format!("{root_name}_D_Term_Spectrums_comparative.png");

    let sample_rate_value = if let Some(sr) = sample_rate {
        if sr.is_finite() && sr > 0.0 {
            sr
        } else {
            println!("\nINFO: Skipping D-Term Spectrum Plot: Invalid sample rate provided.");
            return Ok(());
        }
    } else {
        println!("\nINFO: Skipping D-Term Spectrum Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Calculate filtering delay using enhanced cross-correlation on D-terms
    // This returns Vec<Option<DelayResult>> with per-axis alignment
    let delay_by_axis =
        d_term_delay::calculate_d_term_filtering_delay_comparison(log_data, sample_rate_value);

    let filter_config = header_metadata.map(filter_response::parse_filter_config);

    // Extract gyro rate once for proper Nyquist calculation
    let gyro_rate_hz = filter_response::extract_gyro_rate(header_metadata).unwrap_or(8000.0); // Default 8kHz

    // Extract D-term dynamic notch range for Emuflight (if enabled)
    let dterm_dynamic_notch_range =
        filter_response::extract_dterm_dynamic_notch_range(header_metadata);

    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;

    let mut max_freq_for_auto_scale = 0.0f64;

    // Store axis spectrum data
    let mut axis_spectrums: Vec<AxisSpectrum> = Vec::new();

    // Iterate safely over the minimum of AXIS_NAMES.len() and the fixed array size
    let axis_count = AXIS_NAMES.len().min(3);
    for (axis_idx, &axis_name) in AXIS_NAMES.iter().enumerate().take(axis_count) {
        // Extract gyro_unfilt data for derivative calculation
        let mut gyro_unfilt_series: Vec<f32> = Vec::new();
        for row in log_data {
            if let Some(unfilt_val) = row.gyro_unfilt[axis_idx] {
                gyro_unfilt_series.push(unfilt_val as f32);
            }
        }

        // Calculate unfiltered D-term (derivative of gyroUnfilt)
        let unfilt_d_term_series = if gyro_unfilt_series.len() >= 2 {
            calculate_derivative(&gyro_unfilt_series, sample_rate_value)
        } else {
            println!("  Not enough unfiltered gyro data for {axis_name} axis. Skipping unfiltered D-term spectrum.");
            Vec::new()
        };

        // Extract filtered D-term data (flight controller D-term output)
        let mut filt_d_term_series: Vec<f32> = Vec::new();
        for row in log_data {
            if let Some(d_term_val) = row.d_term[axis_idx] {
                filt_d_term_series.push(d_term_val as f32);
            }
        }

        if unfilt_d_term_series.is_empty() && filt_d_term_series.is_empty() {
            println!("  No D-term data for {axis_name} axis. Skipping D-term spectrum.");
            axis_spectrums.push(AxisSpectrum {
                unfiltered: None,
                filtered: None,
            });
            continue;
        }

        // Determine common length for synchronization
        let min_common_length =
            if !unfilt_d_term_series.is_empty() && !filt_d_term_series.is_empty() {
                unfilt_d_term_series.len().min(filt_d_term_series.len())
            } else if !unfilt_d_term_series.is_empty() {
                unfilt_d_term_series.len()
            } else {
                filt_d_term_series.len()
            };

        if min_common_length < 32 {
            println!("  Not enough common D-term data for {axis_name} axis. Skipping spectrum peak analysis.");
            axis_spectrums.push(AxisSpectrum {
                unfiltered: None,
                filtered: None,
            });
            continue;
        }

        // Prepare data arrays for FFT
        let unfilt_data_array = if !unfilt_d_term_series.is_empty() {
            let truncated: Vec<f32> = unfilt_d_term_series
                .into_iter()
                .take(min_common_length)
                .collect();
            Array1::from_vec(truncated)
        } else {
            Array1::zeros(0)
        };

        let filt_data_array = if !filt_d_term_series.is_empty() {
            let truncated: Vec<f32> = filt_d_term_series
                .into_iter()
                .take(min_common_length)
                .collect();
            Array1::from_vec(truncated)
        } else {
            Array1::zeros(0)
        };

        // Apply windowing and compute FFT
        let window_func = calc_step_response::tukeywin(min_common_length, TUKEY_ALPHA);

        let (unfilt_spectrum, unfilt_freqs) = if !unfilt_data_array.is_empty() {
            let windowed_unfilt = &unfilt_data_array * &window_func;
            let spectrum = fft_utils::fft_forward(&windowed_unfilt);

            if !spectrum.is_empty() {
                // number of unique (one‐sided) FFT bins
                let n_unique = spectrum.len();
                // use original input length to compute true frequency step
                let freq_step = sample_rate_value / (min_common_length as f64);
                let freqs: Vec<f64> = (0..n_unique).map(|i| i as f64 * freq_step).collect();
                (spectrum, freqs)
            } else {
                (Array1::zeros(0), Vec::new())
            }
        } else {
            (Array1::zeros(0), Vec::new())
        };

        let (filt_spectrum, filt_freqs) = if !filt_data_array.is_empty() {
            let windowed_filt = &filt_data_array * &window_func;
            let spectrum = fft_utils::fft_forward(&windowed_filt);

            if !spectrum.is_empty() {
                let n_unique = spectrum.len();
                let freq_step = sample_rate_value / (min_common_length as f64);
                let freqs: Vec<f64> = (0..n_unique).map(|i| i as f64 * freq_step).collect();
                (spectrum, freqs)
            } else {
                (Array1::zeros(0), Vec::new())
            }
        } else {
            (Array1::zeros(0), Vec::new())
        };

        if unfilt_spectrum.is_empty() && filt_spectrum.is_empty() {
            println!("  FFT computation failed or resulted in empty spectrums for {axis_name} axis D-terms.");
            axis_spectrums.push(AxisSpectrum {
                unfiltered: None,
                filtered: None,
            });
            continue;
        }

        // Convert to dB (power spectral density) and create plot data
        let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
        let mut filt_series_data: Vec<(f64, f64)> = Vec::new();

        if !unfilt_spectrum.is_empty() {
            for (i, &freq) in unfilt_freqs.iter().enumerate() {
                if freq <= sample_rate_value / 2.0 {
                    let amplitude = unfilt_spectrum[i].norm() as f64;
                    // Use linear amplitude (not dB)
                    unfilt_series_data.push((freq, amplitude));
                    global_max_y_unfilt = global_max_y_unfilt.max(amplitude);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        if !filt_spectrum.is_empty() {
            for (i, &freq) in filt_freqs.iter().enumerate() {
                if freq <= sample_rate_value / 2.0 {
                    let amplitude = filt_spectrum[i].norm() as f64;
                    // Use linear amplitude (not dB)
                    filt_series_data.push((freq, amplitude));
                    global_max_y_filt = global_max_y_filt.max(amplitude);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        // Find peaks for labeling (apply noise floor filtering like gyro plots)
        let unfilt_primary_peak = if !unfilt_series_data.is_empty() {
            unfilt_series_data
                .iter()
                .filter(|(freq, _)| *freq >= SPECTRUM_NOISE_FLOOR_HZ)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };

        let filt_primary_peak = if !filt_series_data.is_empty() {
            filt_series_data
                .iter()
                .filter(|(freq, _)| *freq >= SPECTRUM_NOISE_FLOOR_HZ)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };

        let unfilt_peaks = find_and_sort_peaks_with_threshold(
            &unfilt_series_data,
            unfilt_primary_peak,
            axis_name,
            "Unfiltered D-term Spectrum",
            PEAK_LABEL_MIN_AMPLITUDE,
        );
        // Per issue #92: Skip peak detection on filtered plots entirely - they won't be rendered
        let _filt_primary_peak = filt_primary_peak;
        let filt_peaks = Vec::new();

        // Get delay string for this axis for legend display
        let delay_str = if let Some(result) = delay_by_axis.get(axis_idx).and_then(|r| r.as_ref()) {
            format!(
                "Delay: {:.1}ms(c:{:.0}%)",
                result.delay_ms,
                result.confidence * 100.0
            )
        } else {
            // Don't show delay information for this axis if calculation failed
            "".to_string()
        };

        // Calculate linear amplitude Y-axis range with intelligent floor for D-terms
        // D-terms naturally have lower amplitudes, especially filtered ones
        // Use a much lower floor to prevent compressing low-amplitude signals
        let d_term_floor_unfilt = (global_max_y_unfilt * 0.001).max(1.0); // 0.1% of max or 1.0 minimum
        let d_term_floor_filt = (global_max_y_filt * 0.001).max(1.0); // 0.1% of max or 1.0 minimum

        let max_y_unfilt = global_max_y_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR;
        let y_max_unfilt = max_y_unfilt.max(d_term_floor_unfilt * 100.0); // Ensure reasonable range

        let max_y_filt = global_max_y_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR;
        let y_max_filt = max_y_filt.max(d_term_floor_filt * 100.0); // Ensure reasonable range

        // Create plot configurations
        let unfiltered_config = if !unfilt_series_data.is_empty() {
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sample_rate_value / 2.0)
            } else {
                sample_rate_value / 2.0
            };

            let mut unfilt_plot_series = vec![PlotSeries {
                data: unfilt_series_data,
                label: {
                    let label_base = if delay_str.is_empty() {
                        "Unfiltered D-term".to_string()
                    } else {
                        format!("Unfiltered D-term | {}", delay_str)
                    };
                    super::format_debug_suffix(
                        &label_base,
                        using_debug_fallback,
                        debug_mode_name_owned.as_deref(),
                    )
                },
                color: *COLOR_D_TERM_UNFILT,
                stroke_width: 2,
            }];

            // Add filter response curves to unfiltered plot if available
            if let Some(ref config) = filter_config {
                // Use gyro rate for Nyquist, not logging rate - filters operate at gyro frequency
                let max_freq = gyro_rate_hz / 2.0; // Proper gyro Nyquist frequency
                let num_points = 1000; // More points for smooth curves

                // Generate individual filter response curves for this axis using D-term config
                let filter_curves = filter_response::generate_individual_filter_curves(
                    &config.dterm[axis_idx],
                    max_freq,
                    num_points,
                    show_butterworth,
                );

                // Add each filter curve as a separate series
                let filter_colors = [
                    RGBColor(220, 20, 60), // Crimson for first filter
                    RGBColor(178, 34, 34), // Fire brick for second filter
                    RGBColor(255, 69, 0),  // Red-orange for third filter
                ];

                for (curve_idx, (label, curve_data, cutoff_hz_ref)) in
                    filter_curves.iter().enumerate()
                {
                    if !curve_data.is_empty() {
                        // Show filter response as a normalized curve overlaid on the spectrum
                        // Use a fixed amplitude scale that makes the cutoff frequency visible
                        let overall_max_y_amplitude = global_max_y_unfilt.max(global_max_y_filt);
                        let filter_curve_amplitude = overall_max_y_amplitude * 0.3; // 30% of max spectrum height
                        let filter_curve_offset = overall_max_y_amplitude * 0.05; // Offset from bottom

                        let scaled_response: Vec<(f64, f64)> = curve_data
                            .iter()
                            // Keep overlay within the plotted spectrum range
                            .filter(|(freq, _)| *freq <= max_freq_display)
                            .map(|(freq, response)| {
                                // Scale response from [0,1] to [offset, offset + amplitude]
                                let scaled_amplitude =
                                    filter_curve_offset + (response * filter_curve_amplitude);
                                (*freq, scaled_amplitude)
                            })
                            .collect();

                        // Create filter response series - use gray for effective curves
                        let curve_color = if label.contains("IMUF v256 Effective") {
                            RGBColor(128, 128, 128) // Gray for calculated effective curves
                        } else {
                            filter_colors[curve_idx % filter_colors.len()] // Standard colors for user-configured curves
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

                        // Use dotted line and different color for IMUF effective cutoffs
                        let (cutoff_prefix, cutoff_color) = if label.contains("IMUF v256 Effective")
                        {
                            // Effective cutoffs: dotted line with muted gray color to show they're calculated
                            (CUTOFF_LINE_DOTTED_PREFIX, RGBColor(128, 128, 128))
                        // Gray for calculated values
                        } else {
                            // Header cutoffs: solid line with filter color to show user configuration
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

                // Add IMUF parameters to legend (Q-factor and window size)
                if let Some(ref imuf) = config.dterm[axis_idx].imuf {
                    let imuf_label = if imuf.lowpass_cutoff_hz > 0.0 {
                        // HELIOSPRING: Show full PTn configuration with Q-factor
                        format!(
                            "IMUF Config: Q={:.1}, W={:.0}",
                            imuf.q_factor,
                            imuf.pseudo_kalman_w.unwrap_or(0.0)
                        )
                    } else {
                        // Non-HELIO: Show pseudo-Kalman parameters
                        format!(
                            "Pseudo-Kalman: Q={:.1}, W={:.0}",
                            imuf.q_factor,
                            imuf.pseudo_kalman_w.unwrap_or(0.0)
                        )
                    };
                    unfilt_plot_series.push(PlotSeries {
                        data: vec![], // No data - just for legend
                        label: imuf_label,
                        color: RGBColor(200, 200, 200), // Light gray - no visible line
                        stroke_width: 0,
                    });
                }
            }

            // Create dynamic notch frequency range visualization if configured (Emuflight only)
            // Only show on axes where dynamic notch applies (respect RP-only setting)
            let dterm_dynamic_notch_config = dterm_dynamic_notch_range.as_ref();
            let show_dynamic_notch = if let Some(config) = dterm_dynamic_notch_config {
                // axis_idx: 0=Roll, 1=Pitch, 2=Yaw
                if axis_idx == 2 && !config.applies_to_yaw {
                    false // Skip Yaw if dynamic notch is RP-only
                } else {
                    true
                }
            } else {
                false
            };

            let frequency_ranges = if show_dynamic_notch {
                if let Some(config) = dterm_dynamic_notch_config {
                    use crate::plot_framework::FrequencyRange;

                    let label = format!(
                        "D-term Dynamic Notch: {} notch{}, Q: {:.0}, range: {:.0}-{:.0}Hz{}",
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
                        color: RGBColor(147, 112, 219), // Medium purple - matches gyro dynamic notch
                        opacity: 0.15,                  // Semi-transparent
                        label,
                    }])
                } else {
                    None
                }
            } else {
                None
            };

            Some(PlotConfig {
                title: format!("{} Unfiltered D-term (derivative of gyroUnfilt)", axis_name),
                x_range: 0.0..max_freq_display,
                y_range: d_term_floor_unfilt..y_max_unfilt,
                series: unfilt_plot_series,
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: unfilt_peaks,
                peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                peak_label_format_string: Some("{:.0}".to_string()),
                frequency_ranges, // D-term dynamic notch only on unfiltered plot (Emuflight only)
            })
        } else {
            None
        };

        let filtered_config = if !filt_series_data.is_empty() {
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sample_rate_value / 2.0)
            } else {
                sample_rate_value / 2.0
            };

            Some(PlotConfig {
                title: format!("{} Filtered D-term (flight controller output)", axis_name),
                x_range: 0.0..max_freq_display,
                y_range: d_term_floor_filt..y_max_filt,
                series: vec![PlotSeries {
                    data: filt_series_data,
                    label: if delay_str.is_empty() {
                        "Filtered D-term".to_string()
                    } else {
                        format!("Filtered D-term | {}", delay_str)
                    },
                    color: *COLOR_D_TERM_FILT,
                    stroke_width: 2,
                }],
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: filt_peaks,
                peak_label_threshold: None, // No peak labels on filtered plot per issue #92
                peak_label_format_string: None,
                frequency_ranges: None,
            })
        } else {
            None
        };

        axis_spectrums.push(AxisSpectrum {
            unfiltered: unfiltered_config,
            filtered: filtered_config,
        });
    }

    let overall_max_y_amplitude = global_max_y_unfilt.max(global_max_y_filt);

    if overall_max_y_amplitude <= 0.0 {
        println!("  No valid D-term spectrum data found. Skipping D-term spectrum plot.");
        return Ok(());
    }

    draw_dual_spectrum_plot(
        &output_file,
        root_name,
        "D-Term Spectrums",
        move |axis_index| {
            if axis_index < axis_spectrums.len() {
                Some(axis_spectrums[axis_index].clone())
            } else {
                None
            }
        },
    )?;

    println!("  D-term spectrum plot saved as '{}'", output_file);
    Ok(())
}

// Removed duplicate function: calculate_d_term_filtering_delay_comparison
// Now using the shared implementation from crate::data_analysis::d_term_delay instead
