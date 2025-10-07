// src/plot_functions/plot_psd.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::types::AllPSDData;

use crate::constants::{
    COLOR_GYRO_VS_UNFILT_FILT,
    COLOR_GYRO_VS_UNFILT_UNFILT,
    ENABLE_WINDOW_PEAK_DETECTION,
    LINE_WIDTH_PLOT,
    MAX_PEAKS_TO_LABEL,
    MIN_PEAK_SEPARATION_HZ,
    MIN_SECONDARY_PEAK_RATIO,
    PEAK_DETECTION_WINDOW_RADIUS,
    PSD_PEAK_LABEL_MIN_VALUE_DB,
    PSD_Y_AXIS_FLOOR_DB,
    PSD_Y_AXIS_HEADROOM_FACTOR_DB,
    SPECTRUM_NOISE_FLOOR_HZ,
    TUKEY_ALPHA, // For the window function
};
use crate::data_analysis::calc_step_response;
use crate::data_analysis::fft_utils;
use crate::data_analysis::filter_delay;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, AxisSpectrum, PlotConfig, PlotSeries};

/// Detects and sorts peaks in PSD data (in dB scale)
fn linear_to_db_for_plot(value: f64) -> f64 {
    if value <= 0.0 {
        PSD_Y_AXIS_FLOOR_DB // Clamp very small or zero values to the minimum dB for plotting
    } else {
        10.0 * value.log10()
    }
}

// Helper function to find and sort peaks for PSD data (now in dB)
fn find_and_sort_peaks(
    series_data: &[(f64, f64)],            // Data is now in dB
    primary_peak_info: Option<(f64, f64)>, // Info is now in dB
    axis_name_str: &str,
    spectrum_type_str: &str,
) -> Vec<(f64, f64)> {
    let mut peaks_to_plot: Vec<(f64, f64)> = Vec::new();

    if let Some((peak_freq, peak_amp_db)) = primary_peak_info {
        if peak_amp_db > PSD_PEAK_LABEL_MIN_VALUE_DB {
            // Use new dB threshold
            peaks_to_plot.push((peak_freq, peak_amp_db));
        }
    }

    if series_data.len() > 2 && peaks_to_plot.len() < MAX_PEAKS_TO_LABEL {
        let mut candidate_secondary_peaks: Vec<(f64, f64)> = Vec::new();
        // Iterate from the second point to the second-to-last point,
        // as peak detection logic needs at least one point on each side.
        for j in 1..(series_data.len() - 1) {
            let (freq, amp_db) = series_data[j];

            let is_potential_peak = {
                if ENABLE_WINDOW_PEAK_DETECTION {
                    let w = PEAK_DETECTION_WINDOW_RADIUS;
                    // Check if a full window can be formed around j.
                    if j >= w && j + w < series_data.len() {
                        let mut ge_left_in_window = true;
                        for k_offset in 1..=w {
                            if amp_db < series_data[j - k_offset].1 {
                                ge_left_in_window = false;
                                break;
                            }
                        }

                        let mut gt_right_in_window = true;
                        if ge_left_in_window {
                            for k_offset in 1..=w {
                                if amp_db <= series_data[j + k_offset].1 {
                                    gt_right_in_window = false;
                                    break;
                                }
                            }
                        }
                        ge_left_in_window && gt_right_in_window
                    } else {
                        // Fallback for edges where a full window isn't possible.
                        let prev_amp_db = series_data[j - 1].1;
                        let next_amp_db = series_data[j + 1].1;
                        amp_db >= prev_amp_db && amp_db > next_amp_db
                    }
                } else {
                    // Original 3-point logic
                    let prev_amp_db = series_data[j - 1].1;
                    let next_amp_db = series_data[j + 1].1;
                    amp_db > prev_amp_db && amp_db >= next_amp_db
                }
            };

            if freq >= SPECTRUM_NOISE_FLOOR_HZ
                && is_potential_peak
                && amp_db > PSD_PEAK_LABEL_MIN_VALUE_DB
            {
                // Use new dB threshold
                let mut is_valid_for_secondary_consideration = true;
                if let Some((primary_freq, primary_amp_val_db)) = primary_peak_info {
                    // For dB values, MIN_SECONDARY_PEAK_RATIO (a linear ratio) needs to be converted to a dB difference.
                    // A ratio of 0.05 corresponds to 10 * log10(0.05) = -13.01 dB.
                    // Convert linear threshold to dB: 10 * log10(MIN_SECONDARY_PEAK_RATIO) gives the dB difference
                    let min_secondary_db_relative_to_primary =
                        primary_amp_val_db + 10.0 * MIN_SECONDARY_PEAK_RATIO.log10();

                    if freq == primary_freq && amp_db == primary_amp_val_db {
                        is_valid_for_secondary_consideration = false;
                    } else {
                        is_valid_for_secondary_consideration = (amp_db
                            >= min_secondary_db_relative_to_primary)
                            && ((freq - primary_freq).abs() > MIN_PEAK_SEPARATION_HZ);
                    }
                }
                if is_valid_for_secondary_consideration {
                    candidate_secondary_peaks.push((freq, amp_db));
                }
            }
        }

        candidate_secondary_peaks
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (s_freq, s_amp_db) in candidate_secondary_peaks {
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
            if !too_close_to_existing && s_amp_db > PSD_PEAK_LABEL_MIN_VALUE_DB {
                // Ensure it's still above new min value
                peaks_to_plot.push((s_freq, s_amp_db));
            }
        }
    }

    peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !peaks_to_plot.is_empty() {
        let (main_freq, main_amp_db) = peaks_to_plot[0];
        println!(
            "  {axis_name_str} {spectrum_type_str} Gyro PSD: Primary Peak value {main_amp_db:.2} dB at {main_freq:.0} Hz"
        );
        for (idx, (freq, amp_db)) in peaks_to_plot.iter().skip(1).enumerate() {
            println!(
                "    Subordinate Peak {}: {:.2} dB at {:.0} Hz",
                idx + 1,
                amp_db,
                freq
            );
        }
    } else {
        println!("  {axis_name_str} {spectrum_type_str} Gyro PSD: No significant peaks found.");
    }
    peaks_to_plot
}

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD).
pub fn plot_psd(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_Gyro_PSD_comparative.png");
    let plot_type_name = "Gyro PSD";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro PSD Plot: Sample rate could not be determined.");
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

    let mut all_psd_raw_data: AllPSDData = Default::default();
    let mut global_max_y_unfilt_db = f64::NEG_INFINITY; // Initialize with negative infinity for max
    let mut global_max_y_filt_db = f64::NEG_INFINITY;
    let mut overall_max_y_value_db = f64::NEG_INFINITY;

    // Iterate safely over the minimum of AXIS_NAMES.len() and the fixed array size
    let axis_count = AXIS_NAMES.len().min(all_psd_raw_data.len());
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
            println!("  No unfiltered or filtered gyro data for {axis_name} axis. Skipping PSD analysis.");
            continue;
        }

        let min_len = unfilt_samples.len().min(filt_samples.len());
        if min_len == 0 {
            println!("  Not enough common gyro data for {axis_name} axis. Skipping PSD analysis.");
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
            println!("  FFT computation failed or resulted in empty spectrums for {axis_name} axis. Skipping PSD analysis.");
            continue;
        }

        let mut unfilt_psd_data: Vec<(f64, f64)> = Vec::new();
        let mut filt_psd_data: Vec<(f64, f64)> = Vec::new();
        let freq_step = sr_value / fft_padded_len as f64;
        let num_unique_freqs = if fft_padded_len % 2 == 0 {
            fft_padded_len / 2 + 1
        } else {
            fft_padded_len.div_ceil(2)
        };

        let mut primary_peak_unfilt_db: Option<(f64, f64)> = None;
        let mut primary_peak_filt_db: Option<(f64, f64)> = None;

        let psd_scale = 1.0 / (min_len as f64 * sr_value);

        for i in 0..num_unique_freqs {
            let freq_val = i as f64 * freq_step;

            let mut amp_unfilt_linear_psd = unfilt_spec[i].norm_sqr() as f64 * psd_scale;
            let mut amp_filt_linear_psd = filt_spec[i].norm_sqr() as f64 * psd_scale;

            let is_nyquist = fft_padded_len % 2 == 0 && i == num_unique_freqs - 1;

            if i > 0 && !is_nyquist {
                amp_unfilt_linear_psd *= 2.0;
                amp_filt_linear_psd *= 2.0;
            }

            // Convert to dB for plotting
            let amp_unfilt_db = linear_to_db_for_plot(amp_unfilt_linear_psd);
            let amp_filt_db = linear_to_db_for_plot(amp_filt_linear_psd);

            unfilt_psd_data.push((freq_val, amp_unfilt_db));
            filt_psd_data.push((freq_val, amp_filt_db));

            if freq_val >= SPECTRUM_NOISE_FLOOR_HZ {
                if amp_unfilt_db > primary_peak_unfilt_db.map_or(f64::NEG_INFINITY, |(_, amp)| amp)
                {
                    primary_peak_unfilt_db = Some((freq_val, amp_unfilt_db));
                }
                if amp_filt_db > primary_peak_filt_db.map_or(f64::NEG_INFINITY, |(_, amp)| amp) {
                    primary_peak_filt_db = Some((freq_val, amp_filt_db));
                }
            }
        }

        let unfilt_peaks_for_plot = find_and_sort_peaks(
            &unfilt_psd_data,
            primary_peak_unfilt_db,
            axis_name,
            "Unfiltered",
        );
        let filt_peaks_for_plot =
            find_and_sort_peaks(&filt_psd_data, primary_peak_filt_db, axis_name, "Filtered");

        let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step).max(0.0) as usize;
        let max_val_after_noise_floor_unfilt_db = unfilt_psd_data
            .get(noise_floor_sample_idx..)
            .map_or(f64::NEG_INFINITY, |data_slice| {
                data_slice
                    .iter()
                    .map(|&(_, val)| val)
                    .fold(f64::NEG_INFINITY, |max_val, val| max_val.max(val))
            });
        let max_val_after_noise_floor_filt_db =
            filt_psd_data
                .get(noise_floor_sample_idx..)
                .map_or(f64::NEG_INFINITY, |data_slice| {
                    data_slice
                        .iter()
                        .map(|&(_, val)| val)
                        .fold(f64::NEG_INFINITY, |max_val, val| max_val.max(val))
                });

        // Determine Y-axis range based on actual PSD values, with a configurable floor and headroom.
        let y_max_unfilt_for_range_db = PSD_Y_AXIS_FLOOR_DB
            .max(max_val_after_noise_floor_unfilt_db + PSD_Y_AXIS_HEADROOM_FACTOR_DB);
        let y_max_filt_for_range_db = PSD_Y_AXIS_FLOOR_DB
            .max(max_val_after_noise_floor_filt_db + PSD_Y_AXIS_HEADROOM_FACTOR_DB);

        all_psd_raw_data[axis_idx] = Some((
            unfilt_psd_data,
            unfilt_peaks_for_plot,
            filt_psd_data,
            filt_peaks_for_plot,
        ));
        global_max_y_unfilt_db = global_max_y_unfilt_db.max(y_max_unfilt_for_range_db);
        global_max_y_filt_db = global_max_y_filt_db.max(y_max_filt_for_range_db);
    }

    overall_max_y_value_db = overall_max_y_value_db
        .max(global_max_y_unfilt_db)
        .max(global_max_y_filt_db);
    // Ensure the overall max is at least the floor if no significant peaks push it higher
    if overall_max_y_value_db < PSD_Y_AXIS_FLOOR_DB {
        overall_max_y_value_db = PSD_Y_AXIS_FLOOR_DB;
    }

    draw_dual_spectrum_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        if let Some((unfilt_psd_data, unfilt_peaks, filt_psd_data, filt_peaks)) =
            all_psd_raw_data[axis_index].as_ref().cloned()
        {
            let max_freq_val = sr_value / 2.0;
            let x_range = 0.0..max_freq_val * 1.05;
            // Use the dB-scaled floor and overall max for the Y-axis range
            let y_range_for_all_clone = PSD_Y_AXIS_FLOOR_DB..overall_max_y_value_db;

            let unfilt_plot_series = vec![PlotSeries {
                data: unfilt_psd_data,
                label: "Unfiltered Gyro PSD".to_string(),
                color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                stroke_width: LINE_WIDTH_PLOT,
            }];
            let filt_plot_series = vec![PlotSeries {
                data: filt_psd_data,
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
                        "Filtered Gyro PSD".to_string()
                    } else {
                        format!("Filtered Gyro PSD - {}", method_strings.join(" vs "))
                    }
                } else {
                    "Filtered Gyro PSD".to_string()
                },
                color: *COLOR_GYRO_VS_UNFILT_FILT,
                stroke_width: LINE_WIDTH_PLOT,
            }];

            let unfiltered_plot_config = Some(PlotConfig {
                title: format!(
                    "{} Unfiltered Gyro Power Spectral Density",
                    AXIS_NAMES.get(axis_index).unwrap_or(&"Unknown")
                ),
                x_range: x_range.clone(),
                y_range: y_range_for_all_clone.clone(),
                series: unfilt_plot_series,
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Power/Frequency (dB)".to_string(),
                peaks: unfilt_peaks,
                peak_label_threshold: Some(PSD_PEAK_LABEL_MIN_VALUE_DB),
                peak_label_format_string: Some("{:.2} dB".to_string()),
                frequency_ranges: None,
            });

            let filtered_plot_config = Some(PlotConfig {
                title: format!(
                    "{} Filtered Gyro Power Spectral Density",
                    AXIS_NAMES.get(axis_index).unwrap_or(&"Unknown")
                ),
                x_range,
                y_range: y_range_for_all_clone,
                series: filt_plot_series,
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Power/Frequency (dB)".to_string(),
                peaks: filt_peaks,
                peak_label_threshold: Some(PSD_PEAK_LABEL_MIN_VALUE_DB),
                peak_label_format_string: Some("{:.2} dB".to_string()),
                frequency_ranges: None,
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

// src/plot_functions/plot_psd.rs
