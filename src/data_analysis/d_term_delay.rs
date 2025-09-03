// src/data_analysis/d_term_delay.rs

use ndarray::Array1;

use crate::axis_names::AXIS_NAMES;
use crate::data_analysis::derivative::calculate_derivative;
use crate::data_analysis::filter_delay;
use crate::data_input::log_data::LogRowData;

/// Calculate filtering delay comparison between unfiltered and filtered D-terms
///
/// This function computes the delay between the unfiltered D-term (calculated as derivative of gyroUnfilt)
/// and the filtered D-term (direct flight controller output) for each axis.
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

    // Use AXIS_NAMES.iter().enumerate() for consistency with other parts of the codebase
    for (axis_idx, _) in AXIS_NAMES.iter().enumerate() {
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
            continue;
        }

        // Calculate derivative of unfiltered gyro for comparison with filtered D-term
        let unfilt_d_term = calculate_derivative(&gyro_unfilt_data, sample_rate);

        if unfilt_d_term.len() != d_term_filtered_data.len() {
            let min_len = unfilt_d_term.len().min(d_term_filtered_data.len());
            let unfilt_truncated = &unfilt_d_term[..min_len];
            let filt_truncated = &d_term_filtered_data[..min_len];

            // Use cross-correlation to find delay
            if let Some(result) = filter_delay::calculate_filtering_delay_enhanced_xcorr(
                &Array1::from_vec(filt_truncated.to_vec()),
                &Array1::from_vec(unfilt_truncated.to_vec()),
                sample_rate,
            ) {
                results[axis_idx] = Some(result);
            }
        } else if let Some(result) = filter_delay::calculate_filtering_delay_enhanced_xcorr(
            &Array1::from_vec(d_term_filtered_data),
            &Array1::from_vec(unfilt_d_term),
            sample_rate,
        ) {
            results[axis_idx] = Some(result);
        }
    }

    results
}
