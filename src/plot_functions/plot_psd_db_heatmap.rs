// src/plot_functions/plot_psd_db_heatmap.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    HEATMAP_MIN_PSD_DB, STFT_OVERLAP_FACTOR, STFT_WINDOW_DURATION_S, TUKEY_ALPHA,
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

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro Power Spectral Density (PSD) as heatmaps (spectrograms).
pub fn plot_psd_db_heatmap(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_Gyro_PSD_Spectrogram_comparative.png");
    let plot_type_name = "Gyro PSD Spectrogram";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!(
            "\nINFO: Skipping Gyro PSD Spectrogram Plot: Sample rate could not be determined."
        );
        return Ok(());
    };

    let mut all_heatmap_data: [Option<(HeatmapPlotConfig, HeatmapPlotConfig)>; 3] =
        Default::default();

    // Determine common time range for X-axis of spectrograms
    let first_time = log_data.first().and_then(|row| row.time_sec).unwrap_or(0.0);
    let last_time = log_data.last().and_then(|row| row.time_sec).unwrap_or(0.0);
    let total_duration = last_time - first_time;

    // Calculate STFT parameters
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

    // Frequencies to display on Y-axis (up to Nyquist frequency)
    let max_freq_to_plot = sr_value / 2.0;
    let frequencies_y_bins: Vec<f64> = (0..num_unique_freqs)
        .map(|i| i as f64 * freq_step) // Calculate frequency for each bin
        .filter(|&f| f <= max_freq_to_plot * 1.05) // Add a small buffer for plotting range
        .collect();
    let num_freq_bins_to_plot = frequencies_y_bins.len();

    // Iterate safely over the minimum of AXIS_NAMES.len() and the fixed array size
    let axis_count = AXIS_NAMES.len().min(3); // gyro arrays are [Option<f64>; 3]
    for axis_idx in 0..axis_count {
        let axis_name = AXIS_NAMES[axis_idx];
        let mut unfilt_time_series: Vec<f32> = Vec::new();
        let mut filt_time_series: Vec<f32> = Vec::new();
        let mut time_stamps: Vec<f64> = Vec::new();

        for row in log_data {
            if let (Some(time), Some(unfilt_val), Some(filt_val)) =
                (row.time_sec, row.gyro_unfilt[axis_idx], row.gyro[axis_idx])
            {
                time_stamps.push(time);
                unfilt_time_series.push(unfilt_val as f32);
                filt_time_series.push(filt_val as f32);
            }
        }

        if unfilt_time_series.len() < window_size_samples
            || filt_time_series.len() < window_size_samples
        {
            println!(
                "  Not enough data for {axis_name} axis to perform STFT. Skipping PSD heatmap."
            );
            continue;
        }

        let mut unfilt_psd_matrix: Vec<Vec<f64>> = Vec::new(); // Stores PSD dB values for heatmap
        let mut filt_psd_matrix: Vec<Vec<f64>> = Vec::new();
        let mut time_x_bins: Vec<f64> = Vec::new(); // Stores time centers for X-axis

        let window_func = calc_step_response::tukeywin(window_size_samples, TUKEY_ALPHA); // Use TUKEY_ALPHA

        let mut current_start_sample = 0;
        while current_start_sample + window_size_samples <= unfilt_time_series.len() {
            let end_sample = current_start_sample + window_size_samples;

            let unfilt_window_slice = &unfilt_time_series[current_start_sample..end_sample];
            let filt_window_slice = &filt_time_series[current_start_sample..end_sample];

            // Calculate time center for this window
            let window_time_center =
                time_stamps[current_start_sample] + STFT_WINDOW_DURATION_S / 2.0;

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

            let mut current_unfilt_psd_row: Vec<f64> = Vec::with_capacity(num_freq_bins_to_plot);
            let mut current_filt_psd_row: Vec<f64> = Vec::with_capacity(num_freq_bins_to_plot);

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

                current_unfilt_psd_row.push(linear_to_db_for_heatmap(amp_unfilt_linear_psd));
                current_filt_psd_row.push(linear_to_db_for_heatmap(amp_filt_linear_psd));
            }
            unfilt_psd_matrix.push(current_unfilt_psd_row);
            filt_psd_matrix.push(current_filt_psd_row);
            time_x_bins.push(window_time_center);

            current_start_sample += hop_size_samples;
        }

        if unfilt_psd_matrix.is_empty() {
            println!(
                "  No valid STFT windows generated for {axis_name} axis. Skipping PSD heatmap."
            );
            continue;
        }

        // Determine X-axis range for plotting (time)
        let x_range_plot = if time_x_bins.len() > 1 {
            *time_x_bins.first().unwrap()..*time_x_bins.last().unwrap()
        } else if !time_x_bins.is_empty() {
            time_x_bins[0]..time_x_bins[0] + STFT_WINDOW_DURATION_S
        } else {
            0.0..total_duration
        };

        // Determine Y-axis range for plotting (frequency)
        let y_range_plot = if frequencies_y_bins.len() > 1 {
            *frequencies_y_bins.first().unwrap()..*frequencies_y_bins.last().unwrap()
        } else if !frequencies_y_bins.is_empty() {
            0.0..frequencies_y_bins[0] + freq_step
        } else {
            0.0..max_freq_to_plot
        };

        let unfiltered_heatmap_config = HeatmapPlotConfig {
            title: format!("{axis_name} Unfiltered Gyro PSD Spectrogram"),
            x_range: x_range_plot.clone(),
            y_range: y_range_plot.clone(),
            heatmap_data: HeatmapData {
                x_bins: time_x_bins.clone(),
                y_bins: frequencies_y_bins.clone(),
                values: unfilt_psd_matrix,
            },
            x_label: "Time (s)".to_string(),
            y_label: "Frequency (Hz)".to_string(),
        };

        let filtered_heatmap_config = HeatmapPlotConfig {
            title: format!("{axis_name} Filtered Gyro PSD Spectrogram"), // axis_name is Roll, Pitch, Yaw
            x_range: x_range_plot.clone(),
            y_range: y_range_plot.clone(),
            heatmap_data: HeatmapData {
                x_bins: time_x_bins.clone(),
                y_bins: frequencies_y_bins.clone(),
                values: filt_psd_matrix,
            },
            x_label: "Time (s)".to_string(),
            y_label: "Frequency (Hz)".to_string(),
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

// src/plot_functions/plot_psd_db_heatmap.rs
