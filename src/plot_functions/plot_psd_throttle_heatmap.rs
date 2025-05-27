// src/plot_functions/plot_psd_throttle_heatmap.rs

use std::error::Error;
use ndarray::{Array1, s};
use std::collections::HashMap;

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_heatmap_plot, HeatmapPlotConfig, AxisHeatmapSpectrum, HeatmapData};
use crate::constants::{
    STFT_WINDOW_DURATION_S,
    HEATMAP_MIN_PSD_DB,
    TUKEY_ALPHA,
    SETPOINT_MIN_BIN_VALUE, SETPOINT_MAX_BIN_VALUE, SETPOINT_BIN_COUNT, // New setpoint binning constants
};
use crate::data_analysis::fft_utils;
use crate::data_analysis::calc_step_response;

/// Helper to convert linear PSD to dB. Clamps values to prevent log(0) and provide a floor.
fn linear_to_db_for_heatmap(value: f64) -> f64 {
    if value <= 0.0 {
        HEATMAP_MIN_PSD_DB // Clamp very small or zero values to the minimum dB for plotting
    } else {
        10.0 * value.log10()
    }
}

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD)
/// as heatmaps, with Setpoint Value (Throttle) on the Y-axis and Frequency on the X-axis.
///
/// This plot visualizes how the frequency content (PSD) of each gyro axis changes
/// as the throttle value (setpoint[3]) changes.
pub fn plot_psd_throttle_heatmap(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{}_Gyro_PSD_Throttle_Heatmap_comparative.png", root_name);
    let plot_type_name = "Gyro PSD vs Throttle Heatmap";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro PSD vs Throttle Heatmap Plot: Sample rate could not be determined.");
        return Ok(());
    };

    let axis_names = ["Roll", "Pitch", "Yaw"];

    let mut all_heatmap_data: [Option<(HeatmapPlotConfig, HeatmapPlotConfig)>; 3] = Default::default();

    // Calculate STFT parameters (reusing for windowing individual samples)
    let window_size_samples = (STFT_WINDOW_DURATION_S * sr_value) as usize;
    if window_size_samples == 0 {
        eprintln!("Error: Window size is zero, cannot perform STFT for setpoint heatmap. Adjust STFT_WINDOW_DURATION_S.");
        return Ok(());
    }

    let fft_padded_len = window_size_samples.next_power_of_two();
    let freq_step = sr_value / fft_padded_len as f64;
    let num_unique_freqs = if fft_padded_len % 2 == 0 { fft_padded_len / 2 + 1 } else { (fft_padded_len + 1) / 2 };
    
    // Frequencies to display on X-axis
    let max_freq_to_plot = sr_value / 2.0;
    let frequencies_x_bins: Vec<f64> = (0..num_unique_freqs)
        .map(|i| i as f64 * freq_step) // Calculate frequency for each bin
        .filter(|&f| f <= max_freq_to_plot * 1.05) // Add a small buffer for plotting range
        .collect();
    let num_freq_bins_to_plot = frequencies_x_bins.len();

    // Setpoint bins for Y-axis (Throttle)
    let setpoint_bin_width = (SETPOINT_MAX_BIN_VALUE - SETPOINT_MIN_BIN_VALUE) / SETPOINT_BIN_COUNT as f64;
    let setpoint_y_bins: Vec<f64> = (0..SETPOINT_BIN_COUNT)
        .map(|i| SETPOINT_MIN_BIN_VALUE + i as f64 * setpoint_bin_width)
        .collect();

    for axis_idx in 0..3 {
        let axis_name = axis_names[axis_idx];
        
        // Store samples grouped by setpoint (throttle) bin
        // HashMap<throttle_bin_index, Vec<gyro_sample_value>>
        let mut unfilt_binned_samples: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut filt_binned_samples: HashMap<usize, Vec<f32>> = HashMap::new();

        for row in log_data {
            // Use setpoint[3] (throttle) for binning, regardless of gyro axis
            if let (Some(throttle_val), Some(unfilt_val), Some(filt_val)) = (row.setpoint[3], row.gyro_unfilt[axis_idx], row.gyro[axis_idx]) {
                let bin_idx = ((throttle_val - SETPOINT_MIN_BIN_VALUE) / setpoint_bin_width) as usize;
                if bin_idx < SETPOINT_BIN_COUNT {
                    unfilt_binned_samples.entry(bin_idx).or_insert_with(Vec::new).push(unfilt_val as f32);
                    filt_binned_samples.entry(bin_idx).or_insert_with(Vec::new).push(filt_val as f32);
                }
            }
        }

        // Initialize PSD matrices with zeros, dimensions: [setpoint_bins][frequency_bins]
        let mut unfilt_psd_matrix: Vec<Vec<f64>> = vec![vec![0.0; num_freq_bins_to_plot]; SETPOINT_BIN_COUNT];
        let mut filt_psd_matrix: Vec<Vec<f64>> = vec![vec![0.0; num_freq_bins_to_plot]; SETPOINT_BIN_COUNT];
        
        // Keep track of how many windows contributed to each setpoint bin, for averaging
        let mut unfilt_bin_window_counts: Vec<usize> = vec![0; SETPOINT_BIN_COUNT];
        let mut filt_bin_window_counts: Vec<usize> = vec![0; SETPOINT_BIN_COUNT];

        let window_func = calc_step_response::tukeywin(window_size_samples, TUKEY_ALPHA);
        let psd_scale = 1.0 / (window_size_samples as f64 * sr_value);

        for bin_idx in 0..SETPOINT_BIN_COUNT {
            // Process unfiltered data for this setpoint bin
            if let Some(unfilt_samples_in_bin) = unfilt_binned_samples.get(&bin_idx) {
                if unfilt_samples_in_bin.len() >= window_size_samples {
                    for chunk in unfilt_samples_in_bin.chunks(window_size_samples) {
                        unfilt_bin_window_counts[bin_idx] += 1; // Count each window processed
                        let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
                        // Ensure window_func slice matches chunk length if chunk is smaller than full window_size_samples
                        padded_unfilt.slice_mut(s![0..chunk.len()]).assign(&(&Array1::from_vec(chunk.to_vec()) * &window_func.slice(s![0..chunk.len()])));
                        let unfilt_spec = fft_utils::fft_forward(&padded_unfilt);

                        if !unfilt_spec.is_empty() {
                            for i in 0..num_unique_freqs {
                                if i >= num_freq_bins_to_plot { break; }
                                let mut amp_unfilt_linear_psd = unfilt_spec[i].norm_sqr() as f64 * psd_scale;
                                let is_nyquist = fft_padded_len % 2 == 0 && i == num_unique_freqs - 1;
                                if i > 0 && !is_nyquist { amp_unfilt_linear_psd *= 2.0; }
                                unfilt_psd_matrix[bin_idx][i] += linear_to_db_for_heatmap(amp_unfilt_linear_psd);
                            }
                        }
                    }
                }
            }

            // Process filtered data for this setpoint bin
            if let Some(filt_samples_in_bin) = filt_binned_samples.get(&bin_idx) {
                if filt_samples_in_bin.len() >= window_size_samples {
                    for chunk in filt_samples_in_bin.chunks(window_size_samples) {
                        filt_bin_window_counts[bin_idx] += 1; // Count each window processed
                        let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
                        // Ensure window_func slice matches chunk length
                        padded_filt.slice_mut(s![0..chunk.len()]).assign(&(&Array1::from_vec(chunk.to_vec()) * &window_func.slice(s![0..chunk.len()])));
                        let filt_spec = fft_utils::fft_forward(&padded_filt);

                        if !filt_spec.is_empty() {
                            for i in 0..num_unique_freqs {
                                if i >= num_freq_bins_to_plot { break; }
                                let mut amp_filt_linear_psd = filt_spec[i].norm_sqr() as f64 * psd_scale;
                                let is_nyquist = fft_padded_len % 2 == 0 && i == num_unique_freqs - 1;
                                if i > 0 && !is_nyquist { amp_filt_linear_psd *= 2.0; }
                                filt_psd_matrix[bin_idx][i] += linear_to_db_for_heatmap(amp_filt_linear_psd);
                            }
                        }
                    }
                }
            }

            // Average the PSD values for each bin by the number of windows processed
            for i in 0..num_freq_bins_to_plot {
                if unfilt_bin_window_counts[bin_idx] > 0 {
                    unfilt_psd_matrix[bin_idx][i] /= unfilt_bin_window_counts[bin_idx] as f64;
                } else {
                    unfilt_psd_matrix[bin_idx][i] = HEATMAP_MIN_PSD_DB; // No data for this bin, set to floor
                }
                if filt_bin_window_counts[bin_idx] > 0 {
                    filt_psd_matrix[bin_idx][i] /= filt_bin_window_counts[bin_idx] as f64;
                } else {
                    filt_psd_matrix[bin_idx][i] = HEATMAP_MIN_PSD_DB; // No data for this bin, set to floor
                }
            }
        }

        // Determine X-axis range for plotting (Frequency)
        let x_range_plot = if frequencies_x_bins.len() > 1 {
            frequencies_x_bins.first().unwrap().clone()..frequencies_x_bins.last().unwrap().clone()
        } else if !frequencies_x_bins.is_empty() {
            0.0..frequencies_x_bins[0] + freq_step
        } else {
            0.0..max_freq_to_plot
        };
        
        // Determine Y-axis range for plotting (Setpoint/Throttle)
        let y_range_plot = SETPOINT_MIN_BIN_VALUE..SETPOINT_MAX_BIN_VALUE;

        let unfiltered_heatmap_config = HeatmapPlotConfig {
            title: format!("{} Unfiltered Gyro PSD vs Throttle", axis_name),
            x_range: x_range_plot.clone(),
            y_range: y_range_plot.clone(),
            heatmap_data: HeatmapData {
                x_bins: frequencies_x_bins.clone(), // Frequencies are X
                y_bins: setpoint_y_bins.clone(),    // Setpoints (Throttle) are Y
                values: unfilt_psd_matrix,          // values[y_bin_idx][x_bin_idx]
            },
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle (deg/s)".to_string(), // Specific label for throttle
        };

        let filtered_heatmap_config = HeatmapPlotConfig {
            title: format!("{} Filtered Gyro PSD vs Throttle", axis_name),
            x_range: x_range_plot,
            y_range: y_range_plot,
            heatmap_data: HeatmapData {
                x_bins: frequencies_x_bins.clone(),
                y_bins: setpoint_y_bins.clone(),
                values: filt_psd_matrix,
            },
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle (deg/s)".to_string(), // Specific label for throttle
        };

        all_heatmap_data[axis_idx] = Some((unfiltered_heatmap_config, filtered_heatmap_config));
    }

    draw_dual_heatmap_plot(
        &output_file,
        root_name,
        plot_type_name,
        move |axis_index| {
            if let Some((unfiltered_config, filtered_config)) = all_heatmap_data[axis_index].take() {
                Some(AxisHeatmapSpectrum {
                    unfiltered: Some(unfiltered_config),
                    filtered: Some(filtered_config),
                })
            } else {
                Some(AxisHeatmapSpectrum { unfiltered: None, filtered: None })
            }
        },
    )
}

// src/plot_functions/plot_psd_throttle_heatmap.rs