// src/plot_functions/plot_throttle_freq_heatmap.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::constants::{
    HEATMAP_MIN_PSD_DB, // Removed HEATMAP_MAX_PSD_DB from here
    STFT_OVERLAP_FACTOR,
    STFT_WINDOW_DURATION_S,
    THROTTLE_Y_BINS_COUNT,
    THROTTLE_Y_MAX_VALUE,
    THROTTLE_Y_MIN_VALUE,
    TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response;
use crate::data_analysis::fft_utils;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{
    draw_dual_heatmap_plot, AxisHeatmapSpectrum, HeatmapData, HeatmapPlotConfig,
};

/// Helper to convert linear PSD to dB. Clamps values to prevent log(0) and provide a floor.
fn linear_to_db_for_heatmap(value: f64) -> f64 {
    if value <= 0.0 {
        HEATMAP_MIN_PSD_DB // Clamp very small or zero values to the minimum dB for plotting
    } else {
        10.0 * value.log10()
    }
}

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD) as heatmaps (spectrograms)
/// with Throttle on the Y-axis and Frequency on the X-axis.
pub fn plot_throttle_freq_heatmap(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{}_Throttle_Freq_Heatmap_comparative.png", root_name);
    let plot_type_name = "Throttle-Frequency Heatmap";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Throttle-Frequency Heatmap Plot: Sample rate could not be determined.");
        return Ok(());
    };

    let axis_names = ["Roll", "Pitch", "Yaw"];

    let window_size_samples = (STFT_WINDOW_DURATION_S * sr_value) as usize;
    let hop_size_samples = (window_size_samples as f64 * (1.0 - STFT_OVERLAP_FACTOR)) as usize;
    if hop_size_samples == 0 {
        eprintln!("Error: Hop size is zero, cannot perform STFT. Adjust STFT_OVERLAP_FACTOR or STFT_WINDOW_DURATION_S.");
        return Ok(());
    }
    if window_size_samples == 0 {
        eprintln!(
            "Error: Window size is zero, cannot perform STFT. Adjust STFT_WINDOW_DURATION_S."
        );
        return Ok(());
    }

    let fft_padded_len = window_size_samples.next_power_of_two();
    let freq_step = sr_value / fft_padded_len as f64;
    let num_unique_freqs = if fft_padded_len % 2 == 0 {
        fft_padded_len / 2 + 1
    } else {
        fft_padded_len.div_ceil(2)
    };

    // Frequencies to display on X-axis (up to Nyquist frequency)
    let max_freq_to_plot = sr_value / 2.0;
    let frequencies_x_bins: Vec<f64> = (0..num_unique_freqs)
        .map(|i| i as f64 * freq_step) // Calculate frequency for each bin
        .filter(|&f| f <= max_freq_to_plot * 1.05) // Add a small buffer for plotting range
        .collect();
    let num_freq_bins_to_plot = frequencies_x_bins.len();

    // Throttle bins for Y-axis
    let throttle_range = THROTTLE_Y_MAX_VALUE - THROTTLE_Y_MIN_VALUE;
    let throttle_bin_size = throttle_range / THROTTLE_Y_BINS_COUNT as f64;
    let throttle_y_bins: Vec<f64> = (0..THROTTLE_Y_BINS_COUNT)
        .map(|i| THROTTLE_Y_MIN_VALUE + (i as f64 + 0.5) * throttle_bin_size) // Center of the bin
        .collect();

    let mut all_heatmap_data: [Option<(HeatmapPlotConfig, HeatmapPlotConfig)>; 3] =
        Default::default();

    for axis_idx in 0..3 {
        let axis_name = axis_names[axis_idx];
        let mut unfilt_time_series: Vec<f32> = Vec::new();
        let mut filt_time_series: Vec<f32> = Vec::new();
        let mut throttle_values: Vec<f64> = Vec::new(); // Store throttle for each time point

        for row in log_data {
            if let (Some(unfilt_val), Some(filt_val), Some(throttle_val)) = (
                row.gyro_unfilt[axis_idx],
                row.gyro[axis_idx],
                row.setpoint[3],
            ) {
                unfilt_time_series.push(unfilt_val as f32);
                filt_time_series.push(filt_val as f32);
                throttle_values.push(throttle_val);
            }
        }

        if unfilt_time_series.len() < window_size_samples
            || filt_time_series.len() < window_size_samples
        {
            println!("  Not enough data for {} axis to perform STFT. Skipping Throttle-Frequency heatmap.", axis_name);
            continue;
        }

        // Initialize aggregation matrices: [frequency_bin_idx][throttle_bin_idx]
        let mut unfilt_psd_sums: Vec<Vec<f64>> =
            vec![vec![0.0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];
        let mut unfilt_psd_counts: Vec<Vec<usize>> =
            vec![vec![0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];
        let mut filt_psd_sums: Vec<Vec<f64>> =
            vec![vec![0.0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];
        let mut filt_psd_counts: Vec<Vec<usize>> =
            vec![vec![0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];

        let window_func = calc_step_response::tukeywin(window_size_samples, TUKEY_ALPHA);

        let mut current_start_sample = 0;
        while current_start_sample + window_size_samples <= unfilt_time_series.len() {
            let end_sample = current_start_sample + window_size_samples;

            let unfilt_window_slice = &unfilt_time_series[current_start_sample..end_sample];
            let filt_window_slice = &filt_time_series[current_start_sample..end_sample];

            // Get throttle value at the center of the window
            let window_throttle_val =
                throttle_values[current_start_sample + window_size_samples / 2];

            // Map throttle value to a Y-axis bin
            let mut throttle_y_bin_idx =
                ((window_throttle_val - THROTTLE_Y_MIN_VALUE) / throttle_bin_size).floor() as usize;
            throttle_y_bin_idx = throttle_y_bin_idx.clamp(0, THROTTLE_Y_BINS_COUNT - 1);

            let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
            padded_unfilt
                .slice_mut(s![0..window_size_samples])
                .assign(&(&Array1::from_vec(unfilt_window_slice.to_vec()) * &window_func));
            let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
            padded_filt
                .slice_mut(s![0..window_size_samples])
                .assign(&(&Array1::from_vec(filt_window_slice.to_vec()) * &window_func));

            let unfilt_spec = fft_utils::fft_forward(&padded_unfilt);
            let filt_spec = fft_utils::fft_forward(&padded_filt);

            if unfilt_spec.is_empty() || filt_spec.is_empty() {
                current_start_sample += hop_size_samples;
                continue;
            }

            // Normalization for one-sided Power Spectral Density (PSD) for real signals:
            let psd_scale = 1.0 / (window_size_samples as f64 * sr_value);

            for i in 0..num_unique_freqs {
                if i >= num_freq_bins_to_plot {
                    break;
                } // Only process frequencies we intend to plot

                let mut amp_unfilt_linear_psd = unfilt_spec[i].norm_sqr() as f64 * psd_scale;
                let mut amp_filt_linear_psd = filt_spec[i].norm_sqr() as f64 * psd_scale;

                let is_nyquist = fft_padded_len % 2 == 0 && i == num_unique_freqs - 1;

                if i > 0 && !is_nyquist {
                    amp_unfilt_linear_psd *= 2.0;
                    amp_filt_linear_psd *= 2.0;
                }

                unfilt_psd_sums[i][throttle_y_bin_idx] +=
                    linear_to_db_for_heatmap(amp_unfilt_linear_psd);
                unfilt_psd_counts[i][throttle_y_bin_idx] += 1;
                filt_psd_sums[i][throttle_y_bin_idx] +=
                    linear_to_db_for_heatmap(amp_filt_linear_psd);
                filt_psd_counts[i][throttle_y_bin_idx] += 1;
            }
            current_start_sample += hop_size_samples;
        }

        // Calculate averaged PSDs for the heatmap
        let mut final_unfilt_psd_matrix: Vec<Vec<f64>> =
            vec![vec![0.0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];
        let mut final_filt_psd_matrix: Vec<Vec<f64>> =
            vec![vec![0.0; THROTTLE_Y_BINS_COUNT]; num_freq_bins_to_plot];

        for f_idx in 0..num_freq_bins_to_plot {
            for t_idx in 0..THROTTLE_Y_BINS_COUNT {
                if unfilt_psd_counts[f_idx][t_idx] > 0 {
                    final_unfilt_psd_matrix[f_idx][t_idx] =
                        unfilt_psd_sums[f_idx][t_idx] / unfilt_psd_counts[f_idx][t_idx] as f64;
                } else {
                    final_unfilt_psd_matrix[f_idx][t_idx] = HEATMAP_MIN_PSD_DB; // No data for this bin
                }
                if filt_psd_counts[f_idx][t_idx] > 0 {
                    final_filt_psd_matrix[f_idx][t_idx] =
                        filt_psd_sums[f_idx][t_idx] / filt_psd_counts[f_idx][t_idx] as f64;
                } else {
                    final_filt_psd_matrix[f_idx][t_idx] = HEATMAP_MIN_PSD_DB; // No data for this bin
                }
            }
        }

        if final_unfilt_psd_matrix.is_empty() {
            println!("  No valid STFT windows generated for {} axis. Skipping Throttle-Frequency heatmap.", axis_name);
            continue;
        }

        // Determine X-axis range for plotting (frequency)
        let x_range_plot = if frequencies_x_bins.len() > 1 {
            *frequencies_x_bins.first().unwrap()..*frequencies_x_bins.last().unwrap()
        } else if !frequencies_x_bins.is_empty() {
            frequencies_x_bins[0]..frequencies_x_bins[0] + freq_step
        } else {
            0.0..max_freq_to_plot
        };

        // Determine Y-axis range for plotting (throttle)
        let y_range_plot = THROTTLE_Y_MIN_VALUE..THROTTLE_Y_MAX_VALUE;

        let unfiltered_heatmap_config = HeatmapPlotConfig {
            title: format!("{} Unfiltered Gyro Frequency vs. Throttle", axis_name),
            x_range: x_range_plot.clone(),
            y_range: y_range_plot.clone(),
            heatmap_data: HeatmapData {
                x_bins: frequencies_x_bins.clone(),
                y_bins: throttle_y_bins.clone(),
                values: final_unfilt_psd_matrix,
            },
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle (setpoint)".to_string(),
        };

        let filtered_heatmap_config = HeatmapPlotConfig {
            title: format!("{} Filtered Gyro Frequency vs. Throttle", axis_name),
            x_range: x_range_plot.clone(),
            y_range: y_range_plot.clone(),
            heatmap_data: HeatmapData {
                x_bins: frequencies_x_bins.clone(),
                y_bins: throttle_y_bins.clone(),
                values: final_filt_psd_matrix,
            },
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle (setpoint)".to_string(),
        };

        all_heatmap_data[axis_idx] = Some((unfiltered_heatmap_config, filtered_heatmap_config));
    }

    draw_dual_heatmap_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        if let Some((unfiltered_config, filtered_config)) = all_heatmap_data[axis_index].take() {
            Some(AxisHeatmapSpectrum {
                unfiltered: Some(unfiltered_config),
                filtered: Some(filtered_config),
            })
        } else {
            Some(AxisHeatmapSpectrum {
                unfiltered: None,
                filtered: None,
            })
        }
    })
}

// src/plot_functions/plot_throttle_freq_heatmap.rs
