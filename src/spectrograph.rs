// src/spectrograph.rs

use ndarray::{Array1, Array2, s};
use std::error::Error;
use plotters::style::RGBColor; // Need RGBColor here for the return type
#[allow(unused_imports)] // mean() is used from this trait, but compiler might not detect it
use ndarray_stats::QuantileExt; // Needed for min/max/mean on Array1/ArrayView

use crate::constants::{
    SPECTROGRAPH_WINDOW_S, SPECTROGRAPH_OVERLAP_RATIO,
    SPECTROGRAPH_MIN_LOG_POWER_DB, SPECTROGRAPH_MAX_LOG_POWER_DB,
    // Import color stops needed for mapping
    COLOR_STOP_0, COLOR_STOP_1, COLOR_STOP_2, COLOR_STOP_3
};
use crate::fft_utils; // Use the utility module
use crate::step_response::tukeywin; // Import the tukeywin function

// SpectrographData struct
#[derive(Debug, Clone)]
pub struct SpectrographData {
    // Keep time and frequency for reference, but throttle will be the Y-axis data for plotting
    #[allow(dead_code)] // Keep time field as it's part of the calculated data structure
    pub time: Array1<f64>,        // Center time of each window (for reference)
    pub frequency: Array1<f64>,   // Frequencies for each bin (X-axis data)
    pub throttle: Array1<f64>,    // Mean Throttle (%) for each window (Y-axis data)
    pub power: Array2<f64>,       // Log power spectrum (time_bins x freq_bins)
}

/// Calculates the Spectrograph (Short-Time Fourier Transform power spectrum) of a signal.
/// Takes time, throttle percentage, and signal data.
pub fn calculate_spectrograph(
    time_sec: &Array1<f64>, // Time vector corresponding to the signal
    throttle_percent: &Array1<f64>, // Throttle percentage vector corresponding to the signal
    signal: &Array1<f32>,   // Signal data (gyro or gyroUnfilt)
    sample_rate: f64,
) -> Result<SpectrographData, Box<dyn Error>> {
    let total_len = signal.len();
    if total_len == 0 || time_sec.len() != total_len || throttle_percent.len() != total_len || sample_rate <= 0.0 || signal.iter().any(|v| !v.is_finite()) {
        return Err("Invalid input to calculate_spectrograph: Empty data, length mismatch, or invalid sample rate/values.".into());
    }

    // Calculate window lengths in samples
    let window_size_s = SPECTROGRAPH_WINDOW_S;
    let overlap_ratio = SPECTROGRAPH_OVERLAP_RATIO;

    let window_size_samples_float = window_size_s * sample_rate;
    let window_size_samples = window_size_samples_float.round() as usize;
    if window_size_samples == 0 {
        return Err("Calculated window size is zero. Adjust SPECTROGRAPH_WINDOW_S or sample_rate.".into());
    }
     if window_size_samples > total_len {
         return Err(format!("Window size samples ({}) is larger than input data length ({}) for spectrograph.", window_size_samples, total_len).into());
     }


    let overlap_samples = (window_size_samples as f64 * overlap_ratio).round() as usize;
    let step_samples = window_size_samples.checked_sub(overlap_samples)
        .ok_or("Overlap samples is greater than window size samples for spectrograph.")?;

     // Ensure step is at least 1, unless window_size_samples is 1
     let step_samples = if window_size_samples > 0 { step_samples.max(1) } else { 0 };
    if step_samples == 0 && window_size_samples > 0 {
         return Err(format!("Calculated spectrograph step size is zero (window_size={}, overlap={}). Adjust constants.", window_size_samples, overlap_samples).into());
    }


    // Pad window size to next power of 2 for efficient FFT
    let fft_size = window_size_samples.next_power_of_two();
     // If window_size_samples is 0, next_power_of_two is 1. If fft_size ends up 1 when window_size_samples > 0, handle this.
     if fft_size == 0 && window_size_samples > 0 { return Err("FFT size calculated as zero unexpectedly.".into()); }
     let fft_size = fft_size.max(1); // Ensure FFT size is at least 1


    // Calculate number of frequency bins covering the full Nyquist frequency
    let num_fft_output_bins = fft_size / 2 + 1;
    if num_fft_output_bins == 0 { // This happens if fft_size is 0
        return Err("Calculated FFT output bins is zero.".into());
    }
    let freq_resolution = sample_rate / fft_size as f64;
    // Use all frequency bins up to Nyquist for calculation
    let num_relevant_freq_bins = num_fft_output_bins;


    // Create the frequency vector (centers of bins) - covering up to Nyquist
    let mut freqs = Array1::<f64>::zeros(num_relevant_freq_bins);
    for k in 0..num_relevant_freq_bins {
        freqs[k] = k as f64 * freq_resolution;
    }

    // Generate windows and process
    let mut window_times: Vec<f64> = Vec::new();
    let mut window_mean_throttles_percent: Vec<f64> = Vec::new();
    let mut power_spectra: Vec<Array1<f64>> = Vec::new(); // Store log power per window

    // Use Tukey window with alpha 0.5 (standard for STFT)
    let window_func = tukeywin(window_size_samples, 0.5);


    let mut current_start_sample = 0;
    while current_start_sample + window_size_samples <= total_len {
        let current_end_sample = current_start_sample + window_size_samples;

        // Get window data for signal and throttle
        let window_data = signal.slice(s![current_start_sample..current_end_sample]).to_owned();
        let window_throttle_data = throttle_percent.slice(s![current_start_sample..current_end_sample]).to_owned();

        // Calculate window center time (for reference)
        let window_start_time = time_sec[current_start_sample];
        let window_end_time = time_sec[current_end_sample - 1]; // Last sample in window
        let window_center_time = (window_start_time + window_end_time) / 2.0;

        // Calculate mean throttle percentage for the window (for Y-axis)
        let mean_throttle = if window_throttle_data.len() > 0 {
             window_throttle_data.mean().unwrap_or(0.0) // Use mean from ndarray_stats
        } else {
            0.0
        };


        // Apply window function to signal data only
         if window_data.len() != window_func.len() {
             // This should not happen if window_size_samples is calculated correctly
              eprintln!("Error: Signal window data length mismatch with window function length for window starting at sample {}. Skipping.", current_start_sample);
               current_start_sample += step_samples; // Move to next window
               continue;
         }
        let windowed_signal_data = window_data * &window_func;

        // Pad windowed signal data for FFT
        let mut windowed_data_padded = Array1::<f32>::zeros(fft_size);
        if window_size_samples > 0 { // Avoid slicing 0..0
             windowed_data_padded.slice_mut(s![0..window_size_samples]).assign(&windowed_signal_data);
        }

        // Perform FFT
        let spectrum = fft_utils::fft_forward(&windowed_data_padded);

        // Check spectrum length against the expected bins *before* truncating to relevant freqs
        if spectrum.is_empty() || spectrum.len() != num_fft_output_bins {
             eprintln!("Warning: FFT output length mismatch for window starting at sample {}. Expected {}, got {}. Skipping window.", current_start_sample, num_fft_output_bins, spectrum.len());
             current_start_sample += step_samples; // Move to next window
             continue; // Skip this window
        }

        // Calculate Power Spectrum (|spectrum|^2) and convert to dB
        let mut window_power_db = Array1::<f64>::zeros(num_relevant_freq_bins);
        let epsilon = 1e-12; // Small value for log(0) avoidance

        for k in 0..num_relevant_freq_bins {
            // Use norm_sqr() for the squared magnitude
            let mag_squared = (spectrum[k].norm_sqr() as f64).max(epsilon); // Ensure positive for log
            window_power_db[k] = 10.0 * mag_squared.log10();
             // Ensure finite values
            if !window_power_db[k].is_finite() {
                 eprintln!("Warning: Non-finite power value ({}) calculated at frequency bin {} for window starting at sample {}. Setting to min DB.", window_power_db[k], k, current_start_sample);
                 window_power_db[k] = SPECTROGRAPH_MIN_LOG_POWER_DB; // Or NaN? Setting to min dB seems better for plotting
            }
        }

        window_times.push(window_center_time);
        window_mean_throttles_percent.push(mean_throttle);
        power_spectra.push(window_power_db);

        current_start_sample += step_samples;
    }

     // Handle case where no windows were processed (e.g., total_len < window_size_samples)
     if window_times.is_empty() {
         return Err("No spectrograph windows could be generated (input data too short or window size too large).".into());
     }


    if power_spectra.is_empty() || power_spectra[0].len() != num_relevant_freq_bins {
         // This check might be redundant if window_times is non-empty, but safety first.
        let reason = if power_spectra.is_empty() {"no power spectra calculated"} else {"power spectra length mismatch"};
        return Err(format!("Failed to generate spectrograph data after windowing: {}.", reason).into());
    }


    // Combine power spectra into a 2D Array (Time_bins x Frequency_bins)
    let num_time_bins = window_times.len();
    let power_array = Array2::from_shape_fn((num_time_bins, num_relevant_freq_bins), |(time_idx, freq_idx)| {
        power_spectra[time_idx][freq_idx] // power_spectra is Vec<Array1<f64>>, where each Array1 is power for one window (freqs).
                                         // We want Array2<f64> with shape (time_bins, freqs).
                                         // Indexing: power_array[[time_idx, freq_idx]] should get the power at freq_idx for time_idx
                                         // This corresponds to power_spectra[time_idx][freq_idx]
    });

    Ok(SpectrographData {
        time: Array1::from(window_times),
        frequency: freqs,
        throttle: Array1::from(window_mean_throttles_percent), // Store mean throttle percent
        power: power_array,
    })
}

/// Helper function for linear interpolation between two RGB colors.
fn interpolate_color(c1: &RGBColor, c2: &RGBColor, ratio: f64) -> RGBColor {
    let ratio = ratio.max(0.0).min(1.0);
    RGBColor(
        (c1.0 as f64 * (1.0 - ratio) + c2.0 as f64 * ratio).round() as u8,
        (c1.1 as f64 * (1.0 - ratio) + c2.1 as f64 * ratio).round() as u8,
        (c1.2 as f64 * (1.0 - ratio) + c2.2 as f64 * ratio).round() as u8,
    )
}

/// Maps a log power value (in dB) to an RGB color using a predefined gradient.
/// Uses 4 color stops (Black, Red, Yellow, White) interpolated across the range.
/// Adjusted interpolation points to emphasize Red/Yellow.
pub fn map_log_power_to_color(log_power_db: f64) -> RGBColor {
    let min_db = SPECTROGRAPH_MIN_LOG_POWER_DB;
    let max_db = SPECTROGRAPH_MAX_LOG_POWER_DB;

    // Clamp the value to the range
    let clamped_db = log_power_db.max(min_db).min(max_db);

    // Normalize the value to 0.0..1.0 across the full range [min_db, max_db]
    let normalized_db = if max_db > min_db {
         (clamped_db - min_db) / (max_db - min_db)
    } else {
        // Handle case where min_db == max_db
        if clamped_db >= min_db { 1.0 } else { 0.0 }
    };

    // Define the points in the normalized range [0.0, 1.0] where each color stop is reached.
    // 4 stops: 0.0 (Black), p1 (Red), p2 (Yellow), 1.0 (White)
    // Let's try 0.0, 0.5, 0.8, 1.0
    let p1 = 0.5; // Red at 50% of the normalized range
    let p2 = 0.8; // Yellow at 80% of the normalized range

    if normalized_db < p1 { // 0.0 to p1 -> Black to Red
        interpolate_color(&COLOR_STOP_0, &COLOR_STOP_1, normalized_db / p1)
    } else if normalized_db < p2 { // p1 to p2 -> Red to Yellow
        interpolate_color(&COLOR_STOP_1, &COLOR_STOP_2, (normalized_db - p1) / (p2 - p1))
    } else { // p2 to 1.0 -> Yellow to White
        interpolate_color(&COLOR_STOP_2, &COLOR_STOP_3, (normalized_db - p2) / (1.0 - p2))
    }
}

// src/spectrograph.rs