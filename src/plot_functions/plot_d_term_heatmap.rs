// src/plot_functions/plot_d_term_heatmap.rs

use ndarray::{s, Array1};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    HEATMAP_MIN_PSD_DB, STFT_OVERLAP_FACTOR, STFT_WINDOW_DURATION_S, THROTTLE_Y_BINS_COUNT,
    THROTTLE_Y_MAX_VALUE, THROTTLE_Y_MIN_VALUE, TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response;
use crate::data_analysis::derivative::calculate_derivative;
use crate::data_analysis::fft_utils;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{
    draw_dual_heatmap_plot, AxisHeatmapSpectrum, HeatmapData, HeatmapPlotConfig,
};

/// Helper to convert linear power values to dB. Clamps values to prevent log(0) and provide a floor.
fn linear_to_db_for_heatmap(value: f64) -> f64 {
    if value <= 0.0 {
        HEATMAP_MIN_PSD_DB // Clamp very small or zero values to the minimum dB for plotting
    } else {
        10.0 * value.log10() // Use 10*log10 for power (PSD) to dB conversion
    }
}

/// Generates a stacked plot with two columns per axis, showing Unfiltered D-term and Filtered D-term Power Spectral Density (PSD) as heatmaps (spectrograms)
/// with Throttle on the Y-axis and Frequency on the X-axis.
/// Unfiltered D-term is calculated as the derivative of gyroUnfilt.
/// Filtered D-term uses the flight controller's processed D-term output.
pub fn plot_d_term_heatmap(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_D_Term_Heatmap_comparative.png");
    let plot_type_name = "D-Term Throttle-Frequency Heatmap";

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping D-Term Throttle-Frequency Heatmap Plot: Sample rate could not be determined.");
        return Ok(());
    };

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

    // Store axis spectrum data
    let mut axis_heatmap_spectrums: Vec<AxisHeatmapSpectrum> = Vec::new();

    // Iterate safely over the minimum of AXIS_NAMES.len() and the fixed array size
    let axis_count = AXIS_NAMES.len().min(3);
    for (axis_idx, &axis_name) in AXIS_NAMES.iter().enumerate().take(axis_count) {
        // Extract gyro_unfilt data for derivative calculation
        let mut gyro_unfilt_series: Vec<f32> = Vec::new();
        let mut d_term_filt_series: Vec<f32> = Vec::new();
        let mut throttle_values: Vec<f64> = Vec::new(); // Store throttle for each time point

        for row in log_data {
            if let (Some(unfilt_val), Some(d_term_val), Some(throttle_val)) = (
                row.gyro_unfilt[axis_idx],
                row.d_term[axis_idx],
                row.setpoint[3],
            ) {
                gyro_unfilt_series.push(unfilt_val as f32);
                d_term_filt_series.push(d_term_val as f32);
                throttle_values.push(throttle_val);
            }
        }

        // Calculate unfiltered D-term (derivative of gyroUnfilt)
        let unfilt_d_term_series = if gyro_unfilt_series.len() >= 2 {
            calculate_derivative(&gyro_unfilt_series, sr_value)
        } else {
            Vec::new()
        };

        if unfilt_d_term_series.len() < window_size_samples
            || d_term_filt_series.len() < window_size_samples
        {
            println!("  Not enough D-term data for {axis_name} axis to perform STFT. Skipping D-term heatmap.");
            axis_heatmap_spectrums.push(AxisHeatmapSpectrum {
                unfiltered: None,
                filtered: None,
            });
            continue;
        }

        // Ensure all data series have the same length
        let min_length = unfilt_d_term_series
            .len()
            .min(d_term_filt_series.len())
            .min(throttle_values.len());
        let unfilt_d_term_truncated = &unfilt_d_term_series[..min_length];
        let d_term_filt_truncated = &d_term_filt_series[..min_length];
        let throttle_truncated = &throttle_values[..min_length];

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
        while current_start_sample + window_size_samples <= min_length {
            let end_sample = current_start_sample + window_size_samples;

            let unfilt_window_slice = &unfilt_d_term_truncated[current_start_sample..end_sample];
            let filt_window_slice = &d_term_filt_truncated[current_start_sample..end_sample];

            // Get throttle value at the center of the window
            let window_throttle_val =
                throttle_truncated[current_start_sample + window_size_samples / 2];

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

        let mut unfilt_max_psd = HEATMAP_MIN_PSD_DB;
        let mut filt_max_psd = HEATMAP_MIN_PSD_DB;

        for f_idx in 0..num_freq_bins_to_plot {
            for t_idx in 0..THROTTLE_Y_BINS_COUNT {
                if unfilt_psd_counts[f_idx][t_idx] > 0 {
                    final_unfilt_psd_matrix[f_idx][t_idx] =
                        unfilt_psd_sums[f_idx][t_idx] / unfilt_psd_counts[f_idx][t_idx] as f64;
                    unfilt_max_psd = unfilt_max_psd.max(final_unfilt_psd_matrix[f_idx][t_idx]);
                } else {
                    final_unfilt_psd_matrix[f_idx][t_idx] = HEATMAP_MIN_PSD_DB; // No data for this bin
                }
                if filt_psd_counts[f_idx][t_idx] > 0 {
                    final_filt_psd_matrix[f_idx][t_idx] =
                        filt_psd_sums[f_idx][t_idx] / filt_psd_counts[f_idx][t_idx] as f64;
                    filt_max_psd = filt_max_psd.max(final_filt_psd_matrix[f_idx][t_idx]);
                } else {
                    final_filt_psd_matrix[f_idx][t_idx] = HEATMAP_MIN_PSD_DB; // No data for this bin
                }
            }
        }

        // Use the same scale for both unfiltered and filtered for direct comparison
        let common_max_db = unfilt_max_psd.max(filt_max_psd);

        // Debug output for PSD ranges
        println!(
            "  {axis_name} axis D-term PSD ranges: Unfiltered: {:.1} to {:.1} dB, Filtered: {:.1} to {:.1} dB, Common scale: {:.1} dB",
            HEATMAP_MIN_PSD_DB, unfilt_max_psd, HEATMAP_MIN_PSD_DB, filt_max_psd, common_max_db
        );

        // Create HeatmapData structures
        let unfilt_heatmap_data = HeatmapData {
            values: final_unfilt_psd_matrix,
            x_bins: frequencies_x_bins.clone(),
            y_bins: throttle_y_bins.clone(),
        };
        let filt_heatmap_data = HeatmapData {
            values: final_filt_psd_matrix,
            x_bins: frequencies_x_bins.clone(),
            y_bins: throttle_y_bins.clone(),
        };

        let unfilt_config = HeatmapPlotConfig {
            title: format!("{axis_name} Unfiltered D-term (derivative of gyroUnfilt)"),
            x_range: 0.0..max_freq_to_plot,
            y_range: THROTTLE_Y_MIN_VALUE..THROTTLE_Y_MAX_VALUE,
            heatmap_data: unfilt_heatmap_data,
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle %".to_string(),
            max_db: common_max_db,
        };
        let filt_config = HeatmapPlotConfig {
            title: format!("{axis_name} Filtered D-term (flight controller output)"),
            x_range: 0.0..max_freq_to_plot,
            y_range: THROTTLE_Y_MIN_VALUE..THROTTLE_Y_MAX_VALUE,
            heatmap_data: filt_heatmap_data,
            x_label: "Frequency (Hz)".to_string(),
            y_label: "Throttle %".to_string(),
            max_db: common_max_db,
        };

        axis_heatmap_spectrums.push(AxisHeatmapSpectrum {
            unfiltered: Some(unfilt_config),
            filtered: Some(filt_config),
        });
    }

    // Check if we have any heatmap data to plot
    let has_data = axis_heatmap_spectrums
        .iter()
        .any(|axis| axis.unfiltered.is_some() || axis.filtered.is_some());
    if !has_data {
        println!("  No valid D-term heatmap data found. Skipping D-term heatmap plot.");
        return Ok(());
    }

    // Draw the heatmap plot
    draw_dual_heatmap_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        if axis_index < axis_heatmap_spectrums.len() {
            Some(axis_heatmap_spectrums[axis_index].clone())
        } else {
            None
        }
    })?;

    println!("  D-term heatmap plot saved as '{}'", output_file);
    Ok(())
}
