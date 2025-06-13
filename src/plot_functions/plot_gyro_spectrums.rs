// src/plot_functions/plot_gyro_spectrums.rs

use std::error::Error;
use ndarray::{Array1, s};

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, PlotSeries, PlotConfig, AxisSpectrum};
use crate::constants::{
    SPECTRUM_Y_AXIS_FLOOR, SPECTRUM_NOISE_FLOOR_HZ, SPECTRUM_Y_AXIS_HEADROOM_FACTOR,
    COLOR_GYRO_VS_UNFILT_UNFILT, COLOR_GYRO_VS_UNFILT_FILT, LINE_WIDTH_PLOT, PEAK_LABEL_MIN_AMPLITUDE,
    MAX_PEAKS_TO_LABEL, MIN_SECONDARY_PEAK_RATIO, MIN_PEAK_SEPARATION_HZ,
    ENABLE_WINDOW_PEAK_DETECTION, PEAK_DETECTION_WINDOW_RADIUS,
    TUKEY_ALPHA,
};
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::filter_delay;

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro spectrums.
pub fn plot_gyro_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{}_Gyro_Spectrums_comparative.png", root_name);
    let plot_type_name = "Gyro Spectrums";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro Spectrum Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Calculate filtering delay using both methods for comparison
    let (_average_delay_ms, delay_comparison_results) = if let Some((avg_delay, results)) = filter_delay::calculate_average_filtering_delay_comparison(log_data, sr_value) {
        (Some(avg_delay), Some(results))
    } else {
        // Fallback to original method
        (filter_delay::calculate_average_filtering_delay(log_data, sr_value), None)
    };

    let mut all_fft_raw_data: [Option<(Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>)>; 3] = Default::default();
    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;
    let mut overall_max_y_amplitude = 0.0f64;

    let axis_names = ["Roll", "Pitch", "Yaw"];

    fn find_and_sort_peaks(
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
                
                let is_potential_peak = { // Assign directly from the block's result
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
                            if ge_left_in_window {  // Optimization: only check right if left is good
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
                            let prev_amp = series_data[j-1].1;
                            let next_amp = series_data[j+1].1;
                            // Using rightmost point of plateau for consistency with window logic's tendency
                            amp >= prev_amp && amp > next_amp // Return this value for this path
                        }
                    } else {
                        // Original 3-point logic (leftmost point of plateau or sharp peak).
                        // The loop for j ensures j-1 and j+1 are always valid.
                        let prev_amp = series_data[j-1].1;
                        let next_amp = series_data[j+1].1;
                        amp > prev_amp && amp >= next_amp // Return this value for this path
                    }
                }; // End of block assignment to is_potential_peak
                
                if freq >= SPECTRUM_NOISE_FLOOR_HZ && is_potential_peak && amp > PEAK_LABEL_MIN_AMPLITUDE {
                    let mut is_valid_for_secondary_consideration = true;
                    if let Some((primary_freq, primary_amp_val)) = primary_peak_info {
                        if freq == primary_freq && amp == primary_amp_val { // Don't re-add the primary peak
                            is_valid_for_secondary_consideration = false;
                        } else {
                            is_valid_for_secondary_consideration = (amp >= primary_amp_val * MIN_SECONDARY_PEAK_RATIO) &&
                                                                    ((freq - primary_freq).abs() > MIN_PEAK_SEPARATION_HZ);
                        }
                    }
                    // If no primary_peak_info, is_valid_for_secondary_consideration remains true (as long as it's a potential peak and above min amplitude)
                    if is_valid_for_secondary_consideration {
                        candidate_secondary_peaks.push((freq, amp));
                    }
                }
            }
            
            candidate_secondary_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (s_freq, s_amp) in candidate_secondary_peaks {
                if peaks_to_plot.len() >= MAX_PEAKS_TO_LABEL { break; }
                let mut too_close_to_existing = false;
                for (p_freq, _) in &peaks_to_plot {
                    if (s_freq - *p_freq).abs() < MIN_PEAK_SEPARATION_HZ {
                        too_close_to_existing = true;
                        break;
                    }
                }
                if !too_close_to_existing && s_amp > PEAK_LABEL_MIN_AMPLITUDE { // Ensure it's still above min amp
                    peaks_to_plot.push((s_freq, s_amp));
                }
            }
        }
        
        peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !peaks_to_plot.is_empty() {
            let (main_freq, main_amp) = peaks_to_plot[0];
            println!("  {} {} Gyro Spectrum: Primary Peak amplitude {:.0} at {:.0} Hz", axis_name_str, spectrum_type_str, main_amp, main_freq);
            for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
                println!("    Subordinate Peak: {}: {:.0} at {:.0} Hz", idx + 1, amp, freq);
            }
        } else {
            println!("  {} {} Gyro Spectrum: No significant peaks found.", axis_name_str, spectrum_type_str);
        }
        peaks_to_plot
    }

    for axis_idx in 0..3 {
            let axis_name = axis_names[axis_idx];
            let mut unfilt_samples: Vec<f32> = Vec::new();
            let mut filt_samples: Vec<f32> = Vec::new();

            for row in log_data {
                if let (Some(unfilt_val), Some(filt_val)) = (row.gyro_unfilt[axis_idx], row.gyro[axis_idx]) {
                    unfilt_samples.push(unfilt_val as f32);
                    filt_samples.push(filt_val as f32);
                }
            }

            if unfilt_samples.is_empty() || filt_samples.is_empty() {
                println!("  No unfiltered or filtered gyro data for {} axis. Skipping spectrum peak analysis.", axis_name);
                continue;
            }

            let min_len = unfilt_samples.len().min(filt_samples.len());
            if min_len == 0 {
                println!("  Not enough common gyro data for {} axis. Skipping spectrum peak analysis.", axis_name);
                continue;
            }

            let unfilt_samples_slice = &unfilt_samples[0..min_len];
            let filt_samples_slice = &filt_samples[0..min_len];
            let window_func = calc_step_response::tukeywin(min_len, TUKEY_ALPHA);

            let fft_padded_len = min_len.next_power_of_two();
            let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
            padded_unfilt.slice_mut(s![0..min_len]).assign(&(&Array1::from_vec(unfilt_samples_slice.to_vec()) * &window_func));
            let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
            padded_filt.slice_mut(s![0..min_len]).assign(&(&Array1::from_vec(filt_samples_slice.to_vec()) * &window_func));

            let unfilt_spec = fft_utils::fft_forward(&padded_unfilt);
            let filt_spec = fft_utils::fft_forward(&padded_filt);

            if unfilt_spec.is_empty() || filt_spec.is_empty() {
                println!("  FFT computation failed or resulted in empty spectrums for {} axis. Skipping spectrum peak analysis.", axis_name);
                continue;
            }

            let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
            let mut filt_series_data: Vec<(f64, f64)> = Vec::new();
            let freq_step = sr_value / fft_padded_len as f64;
            let num_unique_freqs = if fft_padded_len % 2 == 0 { fft_padded_len / 2 + 1 } else { (fft_padded_len + 1) / 2 };
            
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

            let unfilt_peaks_for_plot = find_and_sort_peaks(&unfilt_series_data, primary_peak_unfilt, axis_name, "Unfiltered");
            let filt_peaks_for_plot = find_and_sort_peaks(&filt_series_data, primary_peak_filt, axis_name, "Filtered");

            let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step).max(0.0) as usize;
            let max_amp_after_noise_floor_unfilt = unfilt_series_data.get(noise_floor_sample_idx..)
                .map_or(0.0, |data_slice| data_slice.iter().map(|&(_, amp)| amp).fold(0.0f64, |max_val, amp| max_val.max(amp)));
            let max_amp_after_noise_floor_filt = filt_series_data.get(noise_floor_sample_idx..)
                .map_or(0.0, |data_slice| data_slice.iter().map(|&(_, amp)| amp).fold(0.0f64, |max_val, amp| max_val.max(amp)));

            let y_max_unfilt_for_range = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);
            let y_max_filt_for_range = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);

            all_fft_raw_data[axis_idx] = Some((unfilt_series_data, unfilt_peaks_for_plot, filt_series_data, filt_peaks_for_plot));
            global_max_y_unfilt = global_max_y_unfilt.max(y_max_unfilt_for_range);
            global_max_y_filt = global_max_y_filt.max(y_max_filt_for_range);
    }

    overall_max_y_amplitude = overall_max_y_amplitude.max(global_max_y_unfilt).max(global_max_y_filt);
    if overall_max_y_amplitude < SPECTRUM_Y_AXIS_FLOOR { 
        overall_max_y_amplitude = SPECTRUM_Y_AXIS_FLOOR;
    }

    draw_dual_spectrum_plot(
        &output_file,
        root_name,
        plot_type_name,
        move |axis_index| {
            if let Some((unfilt_series_data, unfilt_peaks, filt_series_data, filt_peaks)) = all_fft_raw_data[axis_index].take() {
                let max_freq_val = sr_value / 2.0;
                let x_range = 0.0..max_freq_val * 1.05; 
                let y_range_for_all_clone = 0.0..overall_max_y_amplitude;

                let unfilt_plot_series = vec![
                    PlotSeries {
                        data: unfilt_series_data,
                        label: "Unfiltered Gyro".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];
                let filt_plot_series = vec![
                    PlotSeries {
                        data: filt_series_data,
                        label: if let Some(ref results) = delay_comparison_results {
                            // Show comparison of both methods if available - NO AVERAGING
                            let mut method_strings = Vec::new();
                            for result in results.iter() {
                                if let Some(freq) = result.frequency_hz {
                                    method_strings.push(format!("{}: {:.1}ms@{:.0}Hz(c:{:.2})", 
                                        match result.method.as_str() {
                                            "Enhanced Cross-Correlation" => "XCorr+",
                                            "Cross-Correlation" => "XCorr",
                                            "Transfer Function" => "TFunc",
                                            _ => "Unknown"
                                        },
                                        result.delay_ms, freq, result.confidence));
                                } else {
                                    method_strings.push(format!("{}: {:.1}ms(c:{:.2})", 
                                        match result.method.as_str() {
                                            "Enhanced Cross-Correlation" => "XCorr+",
                                            "Cross-Correlation" => "XCorr",
                                            "Transfer Function" => "TFunc",
                                            _ => "Unknown"
                                        },
                                        result.delay_ms, result.confidence));
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
                    }
                ];

                let unfiltered_plot_config = Some(PlotConfig {
                    title: format!("{} Unfiltered Gyro Spectrum", axis_names[axis_index]),
                    x_range: x_range.clone(),
                    y_range: y_range_for_all_clone.clone(),
                    series: unfilt_plot_series,
                    x_label: "Frequency (Hz)".to_string(),
                    y_label: "Amplitude".to_string(),
                    peaks: unfilt_peaks,
                    // MINIMAL CHANGE: Initialize new fields to Some for linear amplitude plots
                    peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                    peak_label_format_string: Some("{:.0}".to_string()),
                });

                let filtered_plot_config = Some(PlotConfig {
                    title: format!("{} Filtered Gyro Spectrum", axis_names[axis_index]),
                    x_range,
                    y_range: y_range_for_all_clone,
                    series: filt_plot_series,
                    x_label: "Frequency (Hz)".to_string(),
                    y_label: "Amplitude".to_string(),
                    peaks: filt_peaks,
                    // MINIMAL CHANGE: Initialize new fields to Some for linear amplitude plots
                    peak_label_threshold: Some(PEAK_LABEL_MIN_AMPLITUDE),
                    peak_label_format_string: Some("{:.0}".to_string()),
                });

                Some(AxisSpectrum {
                    unfiltered: unfiltered_plot_config,
                    filtered: filtered_plot_config,
                })
            } else {
                Some(AxisSpectrum { unfiltered: None, filtered: None })
            }
        },
    )
}

// src/plot_functions/plot_gyro_spectrums.rs
