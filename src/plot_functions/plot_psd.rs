// src/plot_functions/plot_psd.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::plot_functions::peak_detection::find_and_sort_peaks_with_threshold;
use crate::types::AllPSDData;

use crate::constants::{
    COLOR_GYRO_VS_UNFILT_FILT,
    COLOR_GYRO_VS_UNFILT_UNFILT,
    LINE_WIDTH_PLOT,
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

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD).
pub fn plot_psd(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    using_debug_fallback: bool,
    debug_mode_name: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Clone debug mode name to move into closures
    let debug_mode_name_owned = debug_mode_name.map(|s| s.to_string());
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

        let unfilt_peaks_for_plot = find_and_sort_peaks_with_threshold(
            &unfilt_psd_data,
            primary_peak_unfilt_db,
            axis_name,
            "Unfiltered",
            PSD_PEAK_LABEL_MIN_VALUE_DB,
        );
        let filt_peaks_for_plot = find_and_sort_peaks_with_threshold(
            &filt_psd_data,
            primary_peak_filt_db,
            axis_name,
            "Filtered",
            PSD_PEAK_LABEL_MIN_VALUE_DB,
        );

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
                label: super::format_debug_suffix(
                    "Unfiltered Gyro PSD",
                    using_debug_fallback,
                    debug_mode_name_owned.as_deref(),
                ),
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
