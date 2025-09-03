// src/plot_functions/plot_d_term_psd.rs

use ndarray::Array1;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_D_TERM_FILT, COLOR_D_TERM_UNFILT, PSD_PEAK_LABEL_MIN_VALUE_DB, SPECTRUM_NOISE_FLOOR_HZ,
    TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::d_term_delay;
use crate::data_analysis::derivative::calculate_derivative;
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_dual_spectrum_plot, AxisSpectrum, PlotConfig, PlotSeries};
use crate::plot_functions::peak_detection::find_and_sort_peaks_with_threshold;

/// Generates a stacked plot with two columns per axis, showing Unfiltered D-term and Filtered D-term PSDs.
/// Unfiltered D-term is calculated as the derivative of gyroUnfilt.
/// Filtered D-term uses the flight controller's processed D-term output.
pub fn plot_d_term_psd(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    _header_metadata: Option<&[(String, String)]>,
    debug_mode: bool,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_D_Term_PSD_comparative.png");

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping D-Term Power Spectral Density (PSD) Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Calculate filtering delay using enhanced cross-correlation on D-terms
    // This returns Vec<Option<DelayResult>> with per-axis alignment
    let delay_by_axis =
        d_term_delay::calculate_d_term_filtering_delay_comparison(log_data, sr_value);

    let mut global_max_y_unfilt = 0.0f64;
    let mut global_max_y_filt = 0.0f64;

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

        // Extract filtered D-term data (flight controller D-term output)
        let mut filt_d_term_series: Vec<f32> = Vec::new();
        for row in log_data {
            if let Some(d_term_val) = row.d_term[axis_idx] {
                filt_d_term_series.push(d_term_val as f32);
            }
        }

        // Debug: Check D-term value ranges
        if debug_mode {
            if !unfilt_d_term_series.is_empty() {
                let unfilt_min = unfilt_d_term_series
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                let unfilt_max = unfilt_d_term_series
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let unfilt_rms = (unfilt_d_term_series.iter().map(|&x| x * x).sum::<f32>()
                    / unfilt_d_term_series.len() as f32)
                    .sqrt();
                println!(
                    "  {axis_name} Unfiltered D-term (derivative): min={:.3}, max={:.3}, rms={:.3}",
                    unfilt_min, unfilt_max, unfilt_rms
                );
            }

            if !filt_d_term_series.is_empty() {
                let filt_min = filt_d_term_series
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                let filt_max = filt_d_term_series
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let filt_rms = (filt_d_term_series.iter().map(|&x| x * x).sum::<f32>()
                    / filt_d_term_series.len() as f32)
                    .sqrt();
                println!(
                    "  {axis_name} Filtered D-term (FC output): min={:.3}, max={:.3}, rms={:.3}",
                    filt_min, filt_max, filt_rms
                );
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

        // Calculate window power for proper PSD normalization
        let window_power: f64 =
            window_func.iter().map(|&w| w * w).sum::<f32>() as f64 / min_common_length as f64;

        // Convert to linear amplitude and create plot data with proper PSD normalization
        let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
        let mut filt_series_data: Vec<(f64, f64)> = Vec::new();

        if !unfilt_spectrum.is_empty() {
            for (i, &freq) in unfilt_freqs.iter().enumerate() {
                if freq <= sr_value / 2.0 {
                    // Calculate power spectral density with proper normalization
                    let magnitude_sqr = unfilt_spectrum[i].norm_sqr() as f64;

                    // Proper PSD calculation: divide by (Fs * N) and correct for window power
                    let mut psd =
                        magnitude_sqr / (sr_value * min_common_length as f64 * window_power);

                    // Apply one-sided doubling for positive frequencies (except DC and Nyquist if present)
                    let is_nyquist =
                        (min_common_length % 2 == 0) && (i == unfilt_spectrum.len() - 1);
                    if i > 0 && !is_nyquist {
                        psd *= 2.0;
                    }

                    // Convert to dB/Hz
                    let psd_db = if psd > 0.0 {
                        10.0 * psd.log10()
                    } else {
                        -100.0 // Use -100 dB as floor for zero/negative values
                    };

                    unfilt_series_data.push((freq, psd_db));
                    global_max_y_unfilt = global_max_y_unfilt.max(psd_db);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        if !filt_spectrum.is_empty() {
            for (i, &freq) in filt_freqs.iter().enumerate() {
                if freq <= sr_value / 2.0 {
                    // Calculate power spectral density with proper normalization
                    let magnitude_sqr = filt_spectrum[i].norm_sqr() as f64;

                    // Proper PSD calculation: divide by (Fs * N) and correct for window power
                    let mut psd =
                        magnitude_sqr / (sr_value * min_common_length as f64 * window_power);

                    // Apply one-sided doubling for positive frequencies (except DC and Nyquist if present)
                    let is_nyquist = (min_common_length % 2 == 0) && (i == filt_spectrum.len() - 1);
                    if i > 0 && !is_nyquist {
                        psd *= 2.0;
                    }

                    // Convert to dB/Hz
                    let psd_db = if psd > 0.0 {
                        10.0 * psd.log10()
                    } else {
                        -100.0 // Use -100 dB as floor for zero/negative values
                    };

                    filt_series_data.push((freq, psd_db));
                    global_max_y_filt = global_max_y_filt.max(psd_db);
                    max_freq_for_auto_scale = max_freq_for_auto_scale.max(freq);
                }
            }
        }

        // Find peaks for labeling (apply noise floor filtering like gyro plots)
        let unfilt_primary_peak = if !unfilt_series_data.is_empty() {
            unfilt_series_data
                .iter()
                .filter(|(freq, _)| *freq >= SPECTRUM_NOISE_FLOOR_HZ)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };

        let filt_primary_peak = if !filt_series_data.is_empty() {
            filt_series_data
                .iter()
                .filter(|(freq, _)| *freq >= SPECTRUM_NOISE_FLOOR_HZ)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };

        let unfilt_peaks = find_and_sort_peaks_with_threshold(
            &unfilt_series_data,
            unfilt_primary_peak,
            axis_name,
            "Unfiltered D-term PSD",
            PSD_PEAK_LABEL_MIN_VALUE_DB,
        );
        let filt_peaks = find_and_sort_peaks_with_threshold(
            &filt_series_data,
            filt_primary_peak,
            axis_name,
            "Filtered D-term PSD",
            PSD_PEAK_LABEL_MIN_VALUE_DB,
        );

        // Get delay string for this axis for legend display
        let delay_str = if let Some(result) = &delay_by_axis[axis_idx] {
            format!(
                "Delay: {} {:.1}ms (c:{:.2})",
                if result.method.contains("Cross") {
                    "XCorr"
                } else {
                    "TFunc"
                },
                result.delay_ms,
                result.confidence
            )
        } else {
            "Delay: N/A".to_string()
        };

        // Calculate dB-aligned Y-axis range (following Betaflight log viewer approach)
        let db_step = 10.0;
        let overall_min_db = global_max_y_unfilt.min(global_max_y_filt).min(-60.0); // Set reasonable minimum
        let overall_max_db = global_max_y_unfilt.max(global_max_y_filt);

        let min_y_db = (overall_min_db / db_step).floor() * db_step;
        let max_y_db = if overall_max_db == overall_min_db {
            overall_max_db + db_step // prevent zero range
        } else {
            ((overall_max_db / db_step).floor() + 1.0) * db_step
        };

        // Create plot configurations
        let unfiltered_config = if !unfilt_series_data.is_empty() {
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sr_value / 2.0)
            } else {
                sr_value / 2.0
            };

            Some(PlotConfig {
                title: format!("{} Unfiltered D-term (derivative of gyroUnfilt)", axis_name),
                x_range: 0.0..max_freq_display,
                y_range: min_y_db..max_y_db,
                series: vec![PlotSeries {
                    data: unfilt_series_data,
                    label: format!("Unfiltered D-term | {}", delay_str),
                    color: *COLOR_D_TERM_UNFILT,
                    stroke_width: 2,
                }],
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Power Spectral Density (dB/Hz)".to_string(),
                peaks: unfilt_peaks,
                peak_label_threshold: Some(PSD_PEAK_LABEL_MIN_VALUE_DB),
                peak_label_format_string: Some("{:.0}Hz".to_string()),
            })
        } else {
            None
        };

        let filtered_config = if !filt_series_data.is_empty() {
            let max_freq_display = if max_freq_for_auto_scale > 0.0 {
                (max_freq_for_auto_scale * 1.1).min(sr_value / 2.0)
            } else {
                sr_value / 2.0
            };

            Some(PlotConfig {
                title: format!("{} Filtered D-term (flight controller output)", axis_name),
                x_range: 0.0..max_freq_display,
                y_range: min_y_db..max_y_db,
                series: vec![PlotSeries {
                    data: filt_series_data,
                    label: format!("Filtered D-term | {}", delay_str),
                    color: *COLOR_D_TERM_FILT,
                    stroke_width: 2,
                }],
                x_label: "Frequency (Hz)".to_string(),
                y_label: "Power Spectral Density (dB/Hz)".to_string(),
                peaks: filt_peaks,
                peak_label_threshold: Some(PSD_PEAK_LABEL_MIN_VALUE_DB),
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

    draw_dual_spectrum_plot(&output_file, root_name, "D-Term PSD", move |axis_index| {
        if axis_index < axis_spectrums.len() {
            Some(axis_spectrums[axis_index].clone())
        } else {
            None
        }
    })?;

    println!("  D-term PSD plot saved as '{}'", output_file);
    Ok(())
}

// Removed duplicate function: calculate_d_term_filtering_delay_comparison
// Now using the shared implementation from crate::data_analysis::d_term_delay instead
