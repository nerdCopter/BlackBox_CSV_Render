// src/plot_functions/plot_d_term_spectrums.rs

use ndarray::Array1;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT, ENABLE_WINDOW_PEAK_DETECTION,
    MAX_PEAKS_TO_LABEL, MIN_PEAK_SEPARATION_HZ, MIN_SECONDARY_PEAK_RATIO,
    PEAK_DETECTION_WINDOW_RADIUS, PEAK_LABEL_MIN_AMPLITUDE, SPECTRUM_NOISE_FLOOR_HZ,
    SPECTRUM_Y_AXIS_FLOOR, SPECTRUM_Y_AXIS_HEADROOM_FACTOR, TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_analysis::filter_delay;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, AxisSpectrum, PlotConfig, PlotSeries};

/// Calculates discrete derivative of a time series
/// For D-term analysis, this represents the rate of change of gyro signal
fn calculate_derivative(data: &[f32], sample_rate: f64) -> Vec<f32> {
    if data.len() < 2 {
        return Vec::new();
    }

    let dt = 1.0 / sample_rate;
    let mut derivative = Vec::with_capacity(data.len());

    // Use forward difference for first point
    derivative.push((data[1] - data[0]) / dt as f32);

    // Use central difference for middle points
    for i in 1..data.len() - 1 {
        derivative.push((data[i + 1] - data[i - 1]) / (2.0 * dt as f32));
    }

    // Use backward difference for last point
    let n = data.len() - 1;
    derivative.push((data[n] - data[n - 1]) / dt as f32);

    derivative
}

/// Generates a stacked plot with two columns per axis, showing Unfiltered D-term and Filtered D-term spectrums.
/// Unfiltered D-term is calculated as the derivative of gyroUnfilt.
/// Filtered D-term uses the axisD values directly from the flight controller.
pub fn plot_d_term_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    _header_metadata: Option<&[(String, String)]>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_D_Term_Spectrums_comparative.png");

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping D-Term Spectrum Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Calculate filtering delay using enhanced cross-correlation on D-terms
    let delay_analysis = calculate_d_term_filtering_delay_comparison(log_data, sr_value);
    let delay_comparison_results = if !delay_analysis.results.is_empty() {
        Some(delay_analysis.results)
    } else {
        None
    };

    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;

    fn find_and_sort_peaks(
        series_data: &[(f64, f64)],
        primary_peak_info: Option<(f64, f64)>,
        _axis_name_str: &str,
        _spectrum_type_str: &str,
    ) -> Vec<(f64, f64)> {
        let mut peaks_to_plot: Vec<(f64, f64)> = Vec::new();

        if let Some((peak_freq, peak_amp)) = primary_peak_info {
            if peak_amp > PEAK_LABEL_MIN_AMPLITUDE {
                peaks_to_plot.push((peak_freq, peak_amp));
            }
        }

        if series_data.len() > 2 && peaks_to_plot.len() < MAX_PEAKS_TO_LABEL {
            let mut candidate_secondary_peaks: Vec<(f64, f64)> = Vec::new();
            for j in 1..(series_data.len() - 1) {
                let (freq, amp) = series_data[j];

                let is_potential_peak = {
                    if ENABLE_WINDOW_PEAK_DETECTION {
                        let w = PEAK_DETECTION_WINDOW_RADIUS;
                        if j >= w && j + w < series_data.len() {
                            let mut ge_left_in_window = true;
                            for k_offset in 1..=w {
                                if amp < series_data[j - k_offset].1 {
                                    ge_left_in_window = false;
                                    break;
                                }
                            }

                            let mut gt_right_in_window = true;
                            if ge_left_in_window {
                                for k_offset in 1..=w {
                                    if amp <= series_data[j + k_offset].1 {
                                        gt_right_in_window = false;
                                        break;
                                    }
                                }
                            }

                            ge_left_in_window && gt_right_in_window
                        } else {
                            // Simple peak detection for edge cases
                            amp > series_data[j - 1].1 && amp > series_data[j + 1].1
                        }
                    } else {
                        // Simple peak detection: check immediate neighbors
                        amp > series_data[j - 1].1 && amp > series_data[j + 1].1
                    }
                };

                if is_potential_peak && amp > PEAK_LABEL_MIN_AMPLITUDE {
                    if let Some((primary_freq, primary_amp)) = primary_peak_info {
                        if (freq - primary_freq).abs() >= MIN_PEAK_SEPARATION_HZ
                            && amp >= primary_amp * MIN_SECONDARY_PEAK_RATIO
                        {
                            candidate_secondary_peaks.push((freq, amp));
                        }
                    } else {
                        candidate_secondary_peaks.push((freq, amp));
                    }
                }
            }

            candidate_secondary_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let remaining_slots = MAX_PEAKS_TO_LABEL - peaks_to_plot.len();
            peaks_to_plot.extend(candidate_secondary_peaks.into_iter().take(remaining_slots));
        }

        peaks_to_plot.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        peaks_to_plot
    }

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
            calculate_derivative(&gyro_unfilt_series, sr_value)
        } else {
            println!("  Not enough unfiltered gyro data for {axis_name} axis. Skipping unfiltered D-term spectrum.");
            Vec::new()
        };

        // Extract filtered D-term data (axisD from flight controller)
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
                let freqs: Vec<f64> = (0..spectrum.len())
                    .map(|i| (i as f64 * sr_value) / (min_common_length as f64))
                    .collect();
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
                let freqs: Vec<f64> = (0..spectrum.len())
                    .map(|i| (i as f64 * sr_value) / (min_common_length as f64))
                    .collect();
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

        // Convert to amplitude and create plot data
        let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
        let mut filt_series_data: Vec<(f64, f64)> = Vec::new();

        if !unfilt_spectrum.is_empty() {
            for (i, &freq) in unfilt_freqs.iter().enumerate() {
                if freq <= sr_value / 2.0 && freq >= SPECTRUM_NOISE_FLOOR_HZ {
                    let amplitude = unfilt_spectrum[i].norm() as f64;
                    unfilt_series_data.push((freq, amplitude));
                    global_max_y_unfilt = global_max_y_unfilt.max(amplitude);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        if !filt_spectrum.is_empty() {
            for (i, &freq) in filt_freqs.iter().enumerate() {
                if freq <= sr_value / 2.0 && freq >= SPECTRUM_NOISE_FLOOR_HZ {
                    let amplitude = filt_spectrum[i].norm() as f64;
                    filt_series_data.push((freq, amplitude));
                    global_max_y_filt = global_max_y_filt.max(amplitude);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        // Find peaks for labeling
        let unfilt_primary_peak = if !unfilt_series_data.is_empty() {
            unfilt_series_data
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .copied()
        } else {
            None
        };

        let filt_primary_peak = if !filt_series_data.is_empty() {
            filt_series_data
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .copied()
        } else {
            None
        };

        let unfilt_peaks = find_and_sort_peaks(
            &unfilt_series_data,
            unfilt_primary_peak,
            axis_name,
            "Unfiltered D-term",
        );
        let filt_peaks = find_and_sort_peaks(
            &filt_series_data,
            filt_primary_peak,
            axis_name,
            "Filtered D-term",
        );

        // Get delay string for this axis
        let delay_str = if let Some(ref results) = delay_comparison_results {
            if let Some(result) = results.get(axis_idx) {
                format!(
                    "{}: {:.1}ms(c:{:.2})",
                    if result.method == "Cross-Correlation" {
                        "XCorr"
                    } else {
                        "TFunc"
                    },
                    result.delay_ms,
                    result.confidence
                )
            } else {
                "No delay data".to_string()
            }
        } else {
            "No delay data".to_string()
        };

        // Create plot configurations
        let unfiltered_config = if !unfilt_series_data.is_empty() {
            let max_y_unfilt = global_max_y_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR;
            let y_max_unfilt = max_y_unfilt.max(SPECTRUM_Y_AXIS_FLOOR * 2.0);
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sr_value / 2.0)
            } else {
                sr_value / 2.0
            };

            Some(PlotConfig {
                title: format!(
                    "Unfiltered D-term (derivative of gyroUnfilt) - {} - {}",
                    axis_name, delay_str
                ),
                x_range: 0.0..max_freq_display,
                y_range: SPECTRUM_Y_AXIS_FLOOR..y_max_unfilt,
                series: vec![PlotSeries {
                    data: unfilt_series_data,
                    label: "Unfiltered D-term".to_string(),
                    color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                    stroke_width: 2,
                }],
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: unfilt_peaks,
                peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                peak_label_format_string: Some("{:.0}Hz".to_string()),
            })
        } else {
            None
        };

        let filtered_config = if !filt_series_data.is_empty() {
            let max_y_filt = global_max_y_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR;
            let y_max_filt = max_y_filt.max(SPECTRUM_Y_AXIS_FLOOR * 2.0);
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sr_value / 2.0)
            } else {
                sr_value / 2.0
            };

            Some(PlotConfig {
                title: format!("Filtered D-term (axisD) - {} - {}", axis_name, delay_str),
                x_range: 0.0..max_freq_display,
                y_range: SPECTRUM_Y_AXIS_FLOOR..y_max_filt,
                series: vec![PlotSeries {
                    data: filt_series_data,
                    label: "Filtered D-term".to_string(),
                    color: *COLOR_GYRO_VS_UNFILT_FILT,
                    stroke_width: 2,
                }],
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Amplitude".to_string(),
                peaks: filt_peaks,
                peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                peak_label_format_string: Some("{:.0}Hz".to_string()),
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

/// Calculate D-term filtering delay using cross-correlation
/// Similar to gyro filtering delay but for D-term data
fn calculate_d_term_filtering_delay_comparison(
    log_data: &[LogRowData],
    sample_rate: f64,
) -> filter_delay::DelayAnalysisResult {
    let mut results = Vec::new();

    for axis_idx in 0..3 {
        // Extract data for this axis
        let mut gyro_unfilt_data: Vec<f32> = Vec::new();
        let mut d_term_filtered_data: Vec<f32> = Vec::new();

        for row in log_data {
            if let (Some(unfilt_val), Some(d_term_val)) =
                (row.gyro_unfilt[axis_idx], row.d_term[axis_idx])
            {
                gyro_unfilt_data.push(unfilt_val as f32);
                d_term_filtered_data.push(d_term_val as f32);
            }
        }

        if gyro_unfilt_data.len() < 100 || d_term_filtered_data.len() < 100 {
            continue;
        }

        // Calculate derivative of unfiltered gyro for comparison with filtered D-term
        let unfilt_d_term = calculate_derivative(&gyro_unfilt_data, sample_rate);

        if unfilt_d_term.len() != d_term_filtered_data.len() {
            let min_len = unfilt_d_term.len().min(d_term_filtered_data.len());
            let unfilt_truncated = &unfilt_d_term[..min_len];
            let filt_truncated = &d_term_filtered_data[..min_len];

            // Use cross-correlation to find delay
            if let Some(result) = filter_delay::calculate_filtering_delay_enhanced_xcorr(
                &Array1::from_vec(unfilt_truncated.to_vec()),
                &Array1::from_vec(filt_truncated.to_vec()),
                sample_rate,
            ) {
                results.push(result);
            }
        } else if let Some(result) = filter_delay::calculate_filtering_delay_enhanced_xcorr(
            &Array1::from_vec(unfilt_d_term),
            &Array1::from_vec(d_term_filtered_data),
            sample_rate,
        ) {
            results.push(result);
        }
    }

    filter_delay::DelayAnalysisResult {
        average_delay: None,
        results,
    }
}
