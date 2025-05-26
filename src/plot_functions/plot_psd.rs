// src/plot_functions/plot_psd.rs

use std::error::Error;
use ndarray::{Array1, s};

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, PlotSeries, PlotConfig, AxisSpectrum};
use crate::constants::{
    // Reusing frequency-related spectrum constants
    SPECTRUM_NOISE_FLOOR_HZ, 
    // Reusing plot style constants
    COLOR_GYRO_VS_UNFILT_UNFILT, COLOR_GYRO_VS_UNFILT_FILT, LINE_WIDTH_PLOT,
    // Reusing peak labeling logic constants, but values might need tuning for PSD
    MAX_PEAKS_TO_LABEL, MIN_SECONDARY_PEAK_FACTOR, MIN_PEAK_SEPARATION_HZ,
    ENABLE_WINDOW_PEAK_DETECTION, PEAK_DETECTION_WINDOW_RADIUS,
    PSD_Y_AXIS_FLOOR, PSD_Y_AXIS_HEADROOM_FACTOR, PSD_PEAK_LABEL_MIN_VALUE,
};
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::constants::TUKEY_ALPHA;

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD).
pub fn plot_psd(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{}_Gyro_PSD_comparative.png", root_name);
    let plot_type_name = "Gyro PSD";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro PSD Plot: Sample rate could not be determined.");
        return Ok(());
    };

    let mut all_psd_raw_data: [Option<(Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>)>; 3] = Default::default();
    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;
    let mut overall_max_y_value = 0.0f64;

    let axis_names = ["Roll", "Pitch", "Yaw"];

    // Helper function to find and sort peaks for PSD data
    fn find_and_sort_peaks(
        series_data: &[(f64, f64)],
        primary_peak_info: Option<(f64, f64)>,
        axis_name_str: &str, 
        spectrum_type_str: &str,
    ) -> Vec<(f64, f64)> {
        let mut peaks_to_plot: Vec<(f64, f64)> = Vec::new();

        if let Some((peak_freq, peak_amp)) = primary_peak_info {
            if peak_amp > PSD_PEAK_LABEL_MIN_VALUE { // Use new PSD threshold
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
                            if ge_left_in_window { // Optimization: only check right if left is good
                                for k_offset in 1..=w {
                                    // series_data[j + k_offset] is valid because j + w < series_data.len()
                                    // and k_offset <= w
                                    if amp <= series_data[j + k_offset].1 {
                                        gt_right_in_window = false;
                                        break;
                                    }
                                }
                            }
                            ge_left_in_window && gt_right_in_window 
                        } else {
                            // Fallback for edges where a full window isn't possible.
                            // The loop for j ensures j-1 and j+1 are always valid.
                            let prev_amp = series_data[j-1].1;
                            let next_amp = series_data[j+1].1;
                            // Using rightmost point of plateau for consistency with window logic's tendency
                            amp >= prev_amp && amp > next_amp 
                        }
                    } else {
                        // Original 3-point logic (leftmost point of plateau or sharp peak).
                        // The loop for j ensures j-1 and j+1 are always valid.
                        let prev_amp = series_data[j-1].1;
                        let next_amp = series_data[j+1].1;
                        amp > prev_amp && amp >= next_amp 
                    }
                };
                
                if freq >= SPECTRUM_NOISE_FLOOR_HZ && is_potential_peak && amp > PSD_PEAK_LABEL_MIN_VALUE { // Use new PSD threshold
                    let mut is_valid_for_secondary_consideration = true;
                    if let Some((primary_freq, primary_amp_val)) = primary_peak_info {
                        if freq == primary_freq && amp == primary_amp_val { // Don't re-add the primary peak
                            is_valid_for_secondary_consideration = false;
                        } else {
                            is_valid_for_secondary_consideration = (amp >= primary_amp_val * MIN_SECONDARY_PEAK_FACTOR) &&
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
                if !too_close_to_existing && s_amp > PSD_PEAK_LABEL_MIN_VALUE { // Ensure it's still above new min value
                    peaks_to_plot.push((s_freq, s_amp));
                }
            }
        }
        
        peaks_to_plot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !peaks_to_plot.is_empty() {
            let (main_freq, main_amp) = peaks_to_plot[0];
            println!("  {} {} Gyro PSD: Primary Peak value {:.4e} at {:.0} Hz", axis_name_str, spectrum_type_str, main_amp, main_freq); // Print PSD value
            for (idx, (freq, amp)) in peaks_to_plot.iter().skip(1).enumerate() {
                println!("    Subordinate Peak {}: {:.4e} at {:.0} Hz", idx + 1, amp, freq); // Print PSD value
            }
        } else {
            println!("  {} {} Gyro PSD: No significant peaks found.", axis_name_str, spectrum_type_str);
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
                println!("  No unfiltered or filtered gyro data for {} axis. Skipping PSD analysis.", axis_name);
                continue;
            }

            let min_len = unfilt_samples.len().min(filt_samples.len());
            if min_len == 0 {
                println!("  Not enough common gyro data for {} axis. Skipping PSD analysis.", axis_name);
                continue;
            }

            let unfilt_samples_slice = &unfilt_samples[0..min_len];
            let filt_samples_slice = &filt_samples[0..min_len];
            let window_func = calc_step_response::tukeywin(min_len, TUKEY_ALPHA); // Hanning window

            // Apply window and pad to next power of two for FFT
            let fft_padded_len = min_len.next_power_of_two();
            let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
            let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
            
            padded_unfilt.slice_mut(s![0..min_len]).assign(&(&Array1::from_vec(unfilt_samples_slice.to_vec()) * &window_func));
            padded_filt.slice_mut(s![0..min_len]).assign(&(&Array1::from_vec(filt_samples_slice.to_vec()) * &window_func));
            
            let unfilt_spec = fft_utils::fft_forward(&padded_unfilt);
            let filt_spec = fft_utils::fft_forward(&padded_filt);

            if unfilt_spec.is_empty() || filt_spec.is_empty() {
                println!("  FFT computation failed or resulted in empty spectrums for {} axis. Skipping PSD analysis.", axis_name);
                continue;
            }

            let mut unfilt_psd_data: Vec<(f64, f64)> = Vec::new();
            let mut filt_psd_data: Vec<(f64, f64)> = Vec::new();
            let freq_step = sr_value / fft_padded_len as f64;
            let num_unique_freqs = if fft_padded_len % 2 == 0 { fft_padded_len / 2 + 1 } else { (fft_padded_len + 1) / 2 };
            
            let mut primary_peak_unfilt: Option<(f64, f64)> = None;
            let mut primary_peak_filt: Option<(f64, f64)> = None;

            // Normalization for one-sided Power Spectral Density (PSD) for real signals:
            // Pxx[k] = (2 * |X[k]|^2) / (Fs * N) for k > 0 and k < N/2 (non-DC, non-Nyquist)
            // Pxx[0] = (|X[0]|^2) / (Fs * N) for k = 0 (DC component)
            // Pxx[N/2] = (|X[N/2]|^2) / (Fs * N) for k = N/2 (Nyquist component, if N is even)
            let psd_scale = 1.0 / (min_len as f64 * sr_value); 
            
            for i in 0..num_unique_freqs {
                let freq_val = i as f64 * freq_step;
                
                let mut amp_unfilt_psd = unfilt_spec[i].norm_sqr() as f64 * psd_scale;
                let mut amp_filt_psd = filt_spec[i].norm_sqr() as f64 * psd_scale;

                // Apply factor of 2 for non-DC and non-Nyquist components
                let is_nyquist = fft_padded_len % 2 == 0 && i == num_unique_freqs - 1;

                if i > 0 && !is_nyquist {
                    amp_unfilt_psd *= 2.0;
                    amp_filt_psd *= 2.0;
                }

                unfilt_psd_data.push((freq_val, amp_unfilt_psd));
                filt_psd_data.push((freq_val, amp_filt_psd));

                if freq_val >= SPECTRUM_NOISE_FLOOR_HZ {
                    if amp_unfilt_psd > primary_peak_unfilt.map_or(0.0, |(_, amp)| amp) {
                        primary_peak_unfilt = Some((freq_val, amp_unfilt_psd));
                    }
                    if amp_filt_psd > primary_peak_filt.map_or(0.0, |(_, amp)| amp) {
                        primary_peak_filt = Some((freq_val, amp_filt_psd));
                    }
                }
            }

            let unfilt_peaks_for_plot = find_and_sort_peaks(&unfilt_psd_data, primary_peak_unfilt, axis_name, "Unfiltered");
            let filt_peaks_for_plot = find_and_sort_peaks(&filt_psd_data, primary_peak_filt, axis_name, "Filtered");

            let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step).max(0.0) as usize;
            let max_val_after_noise_floor_unfilt = unfilt_psd_data.get(noise_floor_sample_idx..)
                .map_or(0.0, |data_slice| data_slice.iter().map(|&(_, val)| val).fold(0.0f64, |max_val, val| max_val.max(val)));
            let max_val_after_noise_floor_filt = filt_psd_data.get(noise_floor_sample_idx..)
                .map_or(0.0, |data_slice| data_slice.iter().map(|&(_, val)| val).fold(0.0f64, |max_val, val| max_val.max(val)));

            // Determine Y-axis range based on actual PSD values, with a configurable floor and headroom.
            let y_max_unfilt_for_range = PSD_Y_AXIS_FLOOR.max(max_val_after_noise_floor_unfilt * PSD_Y_AXIS_HEADROOM_FACTOR);
            let y_max_filt_for_range = PSD_Y_AXIS_FLOOR.max(max_val_after_noise_floor_filt * PSD_Y_AXIS_HEADROOM_FACTOR);

            all_psd_raw_data[axis_idx] = Some((unfilt_psd_data, unfilt_peaks_for_plot, filt_psd_data, filt_peaks_for_plot));
            global_max_y_unfilt = global_max_y_unfilt.max(y_max_unfilt_for_range);
            global_max_y_filt = global_max_y_filt.max(y_max_filt_for_range);
    }

    // Ensure overall max Y value for consistent scaling across all plots
    overall_max_y_value = overall_max_y_value.max(global_max_y_unfilt).max(global_max_y_filt);
    if overall_max_y_value < PSD_Y_AXIS_FLOOR { 
        overall_max_y_value = PSD_Y_AXIS_FLOOR;
    }

    // Use the generic `draw_dual_spectrum_plot` to render the PSD charts
    draw_dual_spectrum_plot(
        &output_file,
        root_name,
        plot_type_name,
        move |axis_index| {
            if let Some((unfilt_psd_data, unfilt_peaks, filt_psd_data, filt_peaks)) = all_psd_raw_data[axis_index].take() {
                let max_freq_val = sr_value / 2.0;
                let x_range = 0.0..max_freq_val * 1.05; 
                let y_range_for_all_clone = PSD_Y_AXIS_FLOOR..overall_max_y_value;

                let unfilt_plot_series = vec![
                    PlotSeries {
                        data: unfilt_psd_data,
                        label: "Unfiltered Gyro (PSD)".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];
                let filt_plot_series = vec![
                    PlotSeries {
                        data: filt_psd_data,
                        label: "Filtered Gyro (PSD)".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_FILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];

                let unfiltered_plot_config = Some(PlotConfig {
                    title: format!("{} Unfiltered Gyro Power Spectral Density", axis_names[axis_index]),
                    x_range: x_range.clone(),
                    y_range: y_range_for_all_clone.clone(),
                    series: unfilt_plot_series,
                    x_label: "Frequency (Hz)".to_string(),
                    y_label: "Power/Frequency ((deg/s)^2/Hz)".to_string(), // Correct PSD units
                    peaks: unfilt_peaks,
                });

                let filtered_plot_config = Some(PlotConfig {
                    title: format!("{} Filtered Gyro Power Spectral Density", axis_names[axis_index]),
                    x_range,
                    y_range: y_range_for_all_clone,
                    series: filt_plot_series,
                    x_label: "Frequency (Hz)".to_string(),
                    y_label: "Power/Frequency ((deg/s)^2/Hz)".to_string(), // Correct PSD units
                    peaks: filt_peaks,
                });

                Some(AxisSpectrum {
                    unfiltered: unfiltered_plot_config,
                    filtered: filtered_plot_config,
                })
            } else {
                // If no data was available for this axis, return None for both unfiltered and filtered
                // to let the framework draw the "Data Not Available" message.
                Some(AxisSpectrum { unfiltered: None, filtered: None })
            }
        },
    )
}

// src/plot_functions/plot_psd.rs
