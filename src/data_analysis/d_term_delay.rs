// src/data_analysis/d_term_delay.rs

use ndarray::Array1;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{MAX_DELAY_FRACTION, MAX_DELAY_SAMPLES, MIN_SAMPLES_FOR_DELAY};
use crate::data_analysis::derivative::calculate_derivative;
use crate::data_analysis::filter_delay;
use crate::data_input::log_data::LogRowData;

/// Calculate filtering delay comparison between unfiltered and filtered D-terms
///
/// This function computes the delay between the unfiltered D-term (calculated as derivative of gyroUnfilt)
/// and the filtered D-term (direct flight controller output) for each axis.
///
/// Note: EmuFlight logs often use debug[0-2] as gyroUnfilt fallback, which is handled by the log parser.
/// Many BTFL logs only include axisD[0] and axisD[1] (Roll/Pitch), missing Yaw D-term data.
///
/// # Arguments
/// * `log_data` - The raw log data containing gyro and D-term values
/// * `sample_rate` - The sample rate in Hz
///
/// # Returns
/// Vector of Option<DelayResult> with fixed length matching AXIS_NAMES.len(),
/// where each element corresponds to the respective axis (or None if calculation failed for that axis).
pub fn calculate_d_term_filtering_delay_comparison(
    log_data: &[LogRowData],
    sample_rate: f64,
) -> Vec<Option<filter_delay::DelayResult>> {
    // Initialize with None for all axes to preserve axis alignment
    let mut results: Vec<Option<filter_delay::DelayResult>> = vec![None; AXIS_NAMES.len()];

    // First, check data availability for diagnosis
    let mut gyro_unfilt_available = [false; 3];
    let mut d_term_available = [false; 3];
    let mut sample_counts = [0; 3];
    let mut d_term_max_values = [0.0f32; 3];

    for row in log_data {
        for axis in 0..AXIS_NAMES.len() {
            if row.gyro_unfilt[axis].is_some() {
                gyro_unfilt_available[axis] = true;
            }
            if let Some(d_term_val) = row.d_term[axis] {
                d_term_available[axis] = true;
                sample_counts[axis] += 1;
                d_term_max_values[axis] = d_term_max_values[axis].max((d_term_val as f32).abs());
            }
        }
    }

    println!("D-term delay analysis diagnostic:");
    // Print diagnosis for debugging
    for (axis_idx, axis_name) in AXIS_NAMES.iter().enumerate() {
        if !gyro_unfilt_available[axis_idx] {
            println!(
                "  {}: gyroUnfilt not available (debug fallback may be used)",
                axis_name
            );
        }
        if !d_term_available[axis_idx] {
            println!("  {}: D-term data not logged", axis_name);
        } else if sample_counts[axis_idx] < 100 {
            println!(
                "  {}: Insufficient D-term samples ({})",
                axis_name, sample_counts[axis_idx]
            );
        } else if d_term_max_values[axis_idx] < 1e-6 {
            println!(
                "  {}: D-term data present but appears disabled (max abs: {:.2e})",
                axis_name, d_term_max_values[axis_idx]
            );
        } else {
            println!(
                "  {}: D-term data available ({} samples, max abs: {:.2e})",
                axis_name, sample_counts[axis_idx], d_term_max_values[axis_idx]
            );
        }
    }

    // Use AXIS_NAMES.iter().enumerate() for consistency with other parts of the codebase
    for (axis_idx, axis_name) in AXIS_NAMES.iter().enumerate() {
        // Skip if either data type is unavailable
        if !gyro_unfilt_available[axis_idx] || !d_term_available[axis_idx] {
            continue;
        }

        // Extract data for this axis
        let mut gyro_unfilt_data: Vec<f32> = Vec::new();
        let mut d_term_filtered_data: Vec<f32> = Vec::new();

        for row in log_data {
            if let (Some(unfilt_val), Some(d_term_val)) =
                (row.gyro_unfilt[axis_idx], row.d_term[axis_idx])
            {
                gyro_unfilt_data.push(unfilt_val as f32);
                d_term_filtered_data.push(d_term_val as f32);
            }
        }

        if gyro_unfilt_data.len() < 100 || d_term_filtered_data.len() < 100 {
            println!(
                "  {}: Insufficient paired samples ({} gyro, {} d-term)",
                axis_name,
                gyro_unfilt_data.len(),
                d_term_filtered_data.len()
            );
            continue;
        }

        // Check if D-term is effectively disabled (all values near zero)
        // This happens when D gain is set to 0 in Betaflight (especially common for Yaw)
        let d_term_max_abs = d_term_filtered_data
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);

        // If maximum absolute D-term value is very small, D gain is likely disabled
        const D_TERM_MIN_THRESHOLD: f32 = 1e-6; // Very small threshold for "effectively zero"
        if d_term_max_abs < D_TERM_MIN_THRESHOLD {
            println!(
                "  {}: D-term appears disabled (max abs value: {:.2e}, likely D gain = 0)",
                axis_name, d_term_max_abs
            );
            continue;
        }

        // Additional check: if D-term variance is extremely low, it's probably disabled
        let d_term_mean =
            d_term_filtered_data.iter().sum::<f32>() / d_term_filtered_data.len() as f32;
        let d_term_variance = d_term_filtered_data
            .iter()
            .map(|&x| (x - d_term_mean).powi(2))
            .sum::<f32>()
            / d_term_filtered_data.len() as f32;
        let d_term_std_dev = d_term_variance.sqrt();

        const D_TERM_MIN_STD_DEV: f32 = 1e-6; // Minimum standard deviation for meaningful D-term
        if d_term_std_dev < D_TERM_MIN_STD_DEV {
            println!(
                "  {}: D-term has no variation (std dev: {:.2e}, likely D gain = 0)",
                axis_name, d_term_std_dev
            );
            continue;
        }

        println!(
            "  {}: D-term active (max: {:.2e}, std dev: {:.2e}) - proceeding with delay calculation",
            axis_name, d_term_max_abs, d_term_std_dev
        );

        // Calculate derivative of unfiltered gyro for comparison with filtered D-term
        let unfilt_d_term = calculate_derivative(&gyro_unfilt_data, sample_rate);

        if unfilt_d_term.len() != d_term_filtered_data.len() {
            let min_len = unfilt_d_term.len().min(d_term_filtered_data.len());
            let unfilt_truncated = &unfilt_d_term[..min_len];
            let filt_truncated = &d_term_filtered_data[..min_len];

            // Use cross-correlation to find delay with D-term specific threshold
            if let Some(result) = calculate_d_term_filtering_delay_enhanced_xcorr(
                &Array1::from_vec(filt_truncated.to_vec()),
                &Array1::from_vec(unfilt_truncated.to_vec()),
                sample_rate,
            ) {
                println!(
                    "  {}: D-term delay calculation successful - {:.2}ms (confidence: {:.0}%)",
                    axis_name,
                    result.delay_ms,
                    result.confidence * 100.0
                );
                results[axis_idx] = Some(result);
            } else {
                println!(
                    "  {}: D-term delay calculation failed - correlation below D-term threshold",
                    axis_name
                );
            }
        } else if let Some(result) = calculate_d_term_filtering_delay_enhanced_xcorr(
            &Array1::from_vec(d_term_filtered_data),
            &Array1::from_vec(unfilt_d_term),
            sample_rate,
        ) {
            println!(
                "  {}: D-term delay calculation successful - {:.2}ms (confidence: {:.0}%)",
                axis_name,
                result.delay_ms,
                result.confidence * 100.0
            );
            results[axis_idx] = Some(result);
        } else {
            println!(
                "  {}: D-term delay calculation failed - correlation below D-term threshold",
                axis_name
            );
        }
    }

    // Summary of results
    let successful_axes: Vec<&str> = results
        .iter()
        .enumerate()
        .filter_map(|(idx, result)| {
            if result.is_some() {
                Some(AXIS_NAMES[idx])
            } else {
                None
            }
        })
        .collect();

    if successful_axes.is_empty() {
        println!("D-term delay calculation: No axes succeeded");
    } else {
        println!(
            "D-term delay calculation successful for: {}",
            successful_axes.join(", ")
        );
    }

    results
}

/// D-term specific enhanced cross-correlation with more lenient correlation threshold
/// D-terms are inherently noisier due to differentiation, so we use a lower threshold
fn calculate_d_term_filtering_delay_enhanced_xcorr(
    filtered: &Array1<f32>,
    unfiltered: &Array1<f32>,
    sample_rate: f64,
) -> Option<filter_delay::DelayResult> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }
    if filtered.len() != unfiltered.len() || filtered.len() < MIN_SAMPLES_FOR_DELAY {
        return None;
    }

    let n = filtered.len();
    let max_delay_samples = (n / MAX_DELAY_FRACTION).min(MAX_DELAY_SAMPLES);
    let mut correlations: Vec<f64> = Vec::with_capacity(max_delay_samples);
    let mut best_correlation = f64::NEG_INFINITY;
    let mut best_delay = 0;

    for delay in 1..max_delay_samples {
        if delay >= n {
            break;
        }
        if let Some(correlation) = compute_d_term_pearson_corr_at_delay(filtered, unfiltered, delay)
        {
            correlations.push(correlation);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        } else {
            correlations.push(0.0);
        }
    }

    // Use a much more lenient threshold for D-terms (0.1 instead of 0.2)
    const D_TERM_CORRELATION_THRESHOLD: f64 = 0.1;
    if best_correlation < D_TERM_CORRELATION_THRESHOLD || best_delay == 0 {
        return None;
    }

    // Parabolic interpolation bounds check fix
    let idx = best_delay - 1; // map delay→index (delay 1 → index 0)
    if idx > 0 && idx < correlations.len() - 1 {
        let y1 = correlations[idx - 1] as f32;
        let y2 = correlations[idx] as f32;
        let y3 = correlations[idx + 1] as f32;
        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        let b = (y3 - y1) / 2.0;
        if a.abs() > 1e-10 {
            let sub_sample_offset = -(b as f64) / (2.0 * a as f64);
            let refined_delay = best_delay as f64 + sub_sample_offset.clamp(-0.5, 0.5);
            return Some(filter_delay::DelayResult {
                method: "D-term Cross-Correlation".to_string(),
                delay_ms: ((refined_delay / sample_rate) * 1000.0) as f32,
                confidence: (((best_correlation + 1.0) / 2.0) as f32).clamp(0.0, 1.0),
                frequency_hz: None,
            });
        }
    }
    Some(filter_delay::DelayResult {
        method: "D-term Cross-Correlation".to_string(),
        delay_ms: ((best_delay as f64 / sample_rate) * 1000.0) as f32,
        confidence: (((best_correlation + 1.0) / 2.0) as f32).clamp(0.0, 1.0),
        frequency_hz: None,
    })
}

/// D-term specific Pearson correlation computation with additional noise handling
fn compute_d_term_pearson_corr_at_delay(
    filtered: &Array1<f32>,
    unfiltered: &Array1<f32>,
    delay: usize,
) -> Option<f64> {
    let n = filtered.len();
    if delay >= n {
        return None;
    }

    let len = n - delay;
    const MIN_SAMPLES_FOR_D_TERM: usize = 50; // Lower requirement for D-terms
    if len < MIN_SAMPLES_FOR_D_TERM {
        return None;
    }

    // Additional bounds check for safety
    let safe_len = len
        .min(filtered.len().saturating_sub(delay))
        .min(unfiltered.len());
    if safe_len < MIN_SAMPLES_FOR_D_TERM {
        return None;
    }

    let mut sum_xy = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;

    for i in 0..safe_len {
        let x = filtered[i + delay] as f64;
        let y = unfiltered[i] as f64;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        sum_x += x;
        sum_y += y;
    }

    let n_f = safe_len as f64;
    let radicand = (n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y);

    if radicand > 0.0 {
        let denominator = radicand.sqrt();
        if denominator > 1e-12 {
            // More lenient denominator threshold
            Some((n_f * sum_xy - sum_x * sum_y) / denominator)
        } else {
            None
        }
    } else {
        None
    }
}
