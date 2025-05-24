// src/plot_functions/plot_gyro_spectrums.rs

use std::error::Error;
use ndarray::{Array1, s};

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, PlotSeries};
use crate::constants::{
    SPECTRUM_Y_AXIS_FLOOR, SPECTRUM_NOISE_FLOOR_HZ, SPECTRUM_Y_AXIS_HEADROOM_FACTOR,
    COLOR_GYRO_VS_UNFILT_UNFILT, COLOR_GYRO_VS_UNFILT_FILT,
    LINE_WIDTH_PLOT,
};
use crate::data_analysis::fft_utils; // For fft_forward
use crate::calc_step_response; // For tukeywin

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

    let mut all_fft_data: [Option<(Vec<(f64, f64)>, Option<(f64, f64)>, Vec<(f64, f64)>, Option<(f64, f64)>)>; 3] = Default::default();
    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;
    let mut overall_max_y_amplitude = 0.0f64;

    let axis_names = ["Roll", "Pitch", "Yaw"];

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
            let window_func = calc_step_response::tukeywin(min_len, 1.0); // Use from step_response module
            let unfilt_windowed: Array1<f32> = Array1::from_vec(unfilt_samples_slice.to_vec()) * &window_func;
            let filt_windowed: Array1<f32> = Array1::from_vec(filt_samples_slice.to_vec()) * &window_func;

            let fft_padded_len = min_len.next_power_of_two();
            let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
            padded_unfilt.slice_mut(s![0..min_len]).assign(&unfilt_windowed);
            let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
            padded_filt.slice_mut(s![0..min_len]).assign(&filt_windowed);

            let unfilt_spec = fft_utils::fft_forward(&padded_unfilt); // Use from fft_utils module
            let filt_spec = fft_utils::fft_forward(&padded_filt); // Use from fft_utils module

            if unfilt_spec.is_empty() || filt_spec.is_empty() {
                println!("  FFT computation failed or resulted in empty spectrums for {} axis. Skipping spectrum peak analysis.", axis_name);
                continue;
            }

            let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
            let mut filt_series_data: Vec<(f64, f64)> = Vec::new();
            let freq_step = sr_value / fft_padded_len as f64;
            let num_unique_freqs = if fft_padded_len % 2 == 0 { fft_padded_len / 2 + 1 } else { (fft_padded_len + 1) / 2 };
            let mut max_amp_unfilt = 0.0f64;
            let mut peak_unfilt_freq = f64::NAN;
            let mut max_amp_filt = 0.0f64;
            let mut peak_filt_freq = f64::NAN;

            for i in 0..num_unique_freqs {
                let freq_val = i as f64 * freq_step;
                let amp_unfilt = unfilt_spec[i].norm() as f64;
                let amp_filt = filt_spec[i].norm() as f64;
                unfilt_series_data.push((freq_val, amp_unfilt));
                filt_series_data.push((freq_val, amp_filt));
                if freq_val >= SPECTRUM_NOISE_FLOOR_HZ {
                    if amp_unfilt > max_amp_unfilt {
                        max_amp_unfilt = amp_unfilt;
                        peak_unfilt_freq = freq_val;
                    }
                }
                if freq_val >= SPECTRUM_NOISE_FLOOR_HZ {
                    if amp_filt > max_amp_filt {
                        max_amp_filt = amp_filt;
                        peak_filt_freq = freq_val;
                    }
                }
            }

            let unfilt_peak_info_for_plot = if max_amp_unfilt > 0.0 && !peak_unfilt_freq.is_nan() {
                println!("  {} Unfiltered Gyro Spectrum: Peak amplitude {:.0} at {:.0} Hz", axis_name, max_amp_unfilt, peak_unfilt_freq);
                Some((peak_unfilt_freq, max_amp_unfilt))
            } else {
                println!("  {} Unfiltered Gyro Spectrum: No significant peak found above noise floor.", axis_name);
                None
            };
            let filt_peak_info_for_plot = if max_amp_filt > 0.0 && !peak_filt_freq.is_nan() {
                println!("  {} Filtered Gyro Spectrum: Peak amplitude {:.0} at {:.0} Hz", axis_name, max_amp_filt, peak_filt_freq);
                Some((peak_filt_freq, max_amp_filt))
            } else {
                println!("  {} Filtered Gyro Spectrum: No significant peak found above noise floor.", axis_name);
                None
            };

            let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step) as usize;
            let max_amp_after_noise_floor_unfilt = unfilt_series_data[noise_floor_sample_idx..]
                .iter().map(|&(_, amp)| amp).fold(0.0f64, |max_val, amp| max_val.max(amp));
            let max_amp_after_noise_floor_filt = filt_series_data[noise_floor_sample_idx..]
                .iter().map(|&(_, amp)| amp).fold(0.0f64, |max_val, amp| max_val.max(amp));
            let y_max_unfilt_for_range = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);
            let y_max_filt_for_range = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);

            all_fft_data[axis_idx] = Some((unfilt_series_data, unfilt_peak_info_for_plot, filt_series_data, filt_peak_info_for_plot));
            global_max_y_unfilt = global_max_y_unfilt.max(y_max_unfilt_for_range);
            global_max_y_filt = global_max_y_filt.max(y_max_filt_for_range);
    }

    overall_max_y_amplitude = overall_max_y_amplitude.max(global_max_y_unfilt).max(global_max_y_filt);

    draw_dual_spectrum_plot(
        &output_file,
        root_name,
        plot_type_name,
        move |axis_index| {
            if let Some((unfilt_series_data, unfilt_peak_info, filt_series_data, filt_peak_info)) = all_fft_data[axis_index].take() {
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
                        label: "Filtered Gyro".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_FILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];

                Some([
                    Some((
                        format!("{} Unfiltered Gyro Spectrum", axis_names[axis_index]),
                        x_range.clone(),
                        y_range_for_all_clone.clone(),
                        unfilt_plot_series,
                        "Frequency (Hz)".to_string(),
                        "Amplitude".to_string(),
                        unfilt_peak_info,
                    )),
                    Some((
                        format!("{} Filtered Gyro Spectrum", axis_names[axis_index]),
                        x_range,
                        y_range_for_all_clone,
                        filt_plot_series,
                        "Frequency (Hz)".to_string(),
                        "Amplitude".to_string(),
                        filt_peak_info,
                    )),
                ])
            } else {
                Some([None, None])
            }
        },
    )
}

// src/plot_functions/plot_gyro_spectrums.rs