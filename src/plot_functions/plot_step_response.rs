// src/plot_functions/plot_step_response.rs

use ndarray::{s, Array1, Array2};
use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_STEP_RESPONSE_COMBINED, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_LOW_SP,
    FINAL_NORMALIZED_STEADY_STATE_TOLERANCE, LINE_WIDTH_PLOT, POST_AVERAGING_SMOOTHING_WINDOW,
    RESPONSE_LENGTH_S, STEADY_STATE_END_S, STEADY_STATE_START_S,
};
use crate::data_analysis::calc_step_response; // For average_responses and moving_average_smooth_f64
use crate::data_input::pid_metadata::PidMetadata;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};
use crate::types::{AllStepResponsePlotData, StepResponseResults};

#[allow(clippy::too_many_arguments)]
/// Generates the Stacked Step Response Plot (Blue, Orange, Red)
pub fn plot_step_response(
    step_response_results: &StepResponseResults,
    root_name: &str,
    sample_rate: Option<f64>,
    has_nonzero_f_term_data: &[bool; 3],
    setpoint_threshold: f64,
    show_legend: bool,
    pid_metadata: &PidMetadata,
    peak_values: &[Option<f64>; 3],
    current_pd_ratios: &[Option<f64>; 3],
    assessments: &[Option<&str>; 3],
    recommended_pd_conservative: &[Option<f64>; 3],
    recommended_d_conservative: &[Option<u32>; 3],
    recommended_d_min_conservative: &[Option<u32>; 3],
    recommended_d_max_conservative: &[Option<u32>; 3],
    recommended_pd_aggressive: &[Option<f64>; 3],
    recommended_d_aggressive: &[Option<u32>; 3],
    recommended_d_min_aggressive: &[Option<u32>; 3],
    recommended_d_max_aggressive: &[Option<u32>; 3],
) -> Result<(), Box<dyn Error>> {
    let step_response_plot_duration_s = RESPONSE_LENGTH_S;
    let steady_state_start_s_const = STEADY_STATE_START_S; // from constants
    let steady_state_end_s_const = STEADY_STATE_END_S; // from constants
    let post_averaging_smoothing_window = POST_AVERAGING_SMOOTHING_WINDOW; // from constants
    let color_high_sp: RGBColor = *COLOR_STEP_RESPONSE_HIGH_SP;
    let color_combined: RGBColor = *COLOR_STEP_RESPONSE_COMBINED;
    let color_low_sp: RGBColor = *COLOR_STEP_RESPONSE_LOW_SP;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    let output_file_step = if show_legend {
        format!(
            "{root_name}_Step_Response_stacked_plot_{step_response_plot_duration_s}s_{setpoint_threshold}dps.png"
        )
    } else {
        format!("{root_name}_Step_Response_stacked_plot_{step_response_plot_duration_s}s.png")
    };
    let plot_type_name = "Step Response";
    let sr = sample_rate.unwrap_or(1000.0); // Default sample rate if not provided

    let mut plot_data_per_axis: AllStepResponsePlotData = Default::default();

    // Track global min/max for symmetric Y-axis scaling (issue #115)
    let mut global_resp_min = f64::INFINITY;
    let mut global_resp_max = f64::NEG_INFINITY;

    // Temporary storage for series and metadata before creating final plot_data_per_axis
    let mut temp_axis_data: Vec<Option<(String, Vec<PlotSeries>)>> = vec![None; 3];

    // Compute a safe shared bound for all arrays to prevent out-of-bounds access
    let axis_count = usize::min(
        AXIS_NAMES.len(),
        usize::min(
            step_response_results.len(),
            usize::min(has_nonzero_f_term_data.len(), plot_data_per_axis.len()),
        ),
    );

    // Compute dmax_enabled once to avoid redundant calls in the axis loop
    let dmax_enabled = pid_metadata.is_dmax_enabled();

    for axis_index in 0..axis_count {
        if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) =
            &step_response_results[axis_index]
        {
            let response_length_samples = response_time.len();
            if response_length_samples == 0 || valid_stacked_responses.shape()[0] == 0 {
                continue;
            }

            let num_qc_windows = valid_stacked_responses.shape()[0];

            // Calculate steady-state window indices for this specific response_time array
            let ss_start_idx = (steady_state_start_s_const * sr).floor() as usize;
            let ss_end_idx = (steady_state_end_s_const * sr).ceil() as usize;

            // Ensure indices are within bounds of the response_length_samples
            let current_ss_start_idx = ss_start_idx.min(response_length_samples.saturating_sub(1));
            let current_ss_end_idx = ss_end_idx
                .min(response_length_samples)
                .max(current_ss_start_idx + 1);

            if current_ss_start_idx >= current_ss_end_idx {
                eprintln!("Warning: Axis {axis_index} Step Response: Steady-state window is invalid (start_idx {current_ss_start_idx} >= end_idx {current_ss_end_idx} for response length {response_length_samples}). Skipping final normalization and plot for this axis.");
                continue;
            }

            let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| {
                if v.abs() < setpoint_threshold as f32 {
                    1.0
                } else {
                    0.0
                }
            });
            let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| {
                if v.abs() >= setpoint_threshold as f32 {
                    1.0
                } else {
                    0.0
                }
            });
            let combined_mask: Array1<f32> = Array1::ones(num_qc_windows);

            let process_response = |mask: &Array1<f32>,
                                    stacked_resp: &Array2<f32>,
                                    resp_len_samples_local: usize, // Use a local variable for clarity
                                    ss_start_idx_local: usize,
                                    ss_end_idx_local: usize,
                                    smoothing_window: usize|
             -> Option<Array1<f64>> {
                // Return type simplified to just the response array
                if !mask.iter().any(|&w| w > 0.0) {
                    return None;
                } // No windows selected by mask

                // avg_resp is now an average of *individually Y-corrected* (mostly normalized towards 1) responses
                // from calc_step_response.rs
                calc_step_response::average_responses(stacked_resp, mask, resp_len_samples_local)
                    .ok()
                    .and_then(|avg_resp| {
                        if avg_resp.is_empty() {
                            return None;
                        }

                        let smoothed_resp = calc_step_response::moving_average_smooth_f64(
                            &avg_resp,
                            smoothing_window,
                        );
                        if smoothed_resp.is_empty() {
                            return None;
                        }

                        // Shift the response to start at 0.0
                        let mut shifted_response = smoothed_resp;
                        if !shifted_response.is_empty() {
                            let first_val = shifted_response[0];
                            shifted_response.mapv_inplace(|v| v - first_val);
                        } else {
                            return None; // Should not happen if smoothed_resp wasn't empty
                        }

                        // This steady_state_segment is from the averaged, Y-corrected (upstream),
                        // smoothed, and shifted response. Its mean is used for a final normalization refinement.
                        let steady_state_segment =
                            shifted_response.slice(s![ss_start_idx_local..ss_end_idx_local]);
                        if steady_state_segment.is_empty() {
                            // Guard against empty slice
                            return None;
                        }

                        steady_state_segment
                            .mean()
                            .and_then(|final_ss_mean_for_norm| {
                                if final_ss_mean_for_norm.abs() > 1e-9 {
                                    // Avoid division by zero
                                    // This final normalization step ensures the plotted response aims for 1.0.
                                    // It refines the average of the Y-corrected individual responses.
                                    let normalized_response =
                                        shifted_response.mapv(|v| v / final_ss_mean_for_norm);

                                    // Final check on this fully processed and normalized response
                                    let final_check_ss_segment = normalized_response
                                        .slice(s![ss_start_idx_local..ss_end_idx_local]);
                                    if final_check_ss_segment.is_empty() {
                                        return None;
                                    } // Should not happen

                                    final_check_ss_segment.mean().and_then(
                                        |final_normalized_ss_mean| {
                                            if (final_normalized_ss_mean - 1.0).abs()
                                                <= FINAL_NORMALIZED_STEADY_STATE_TOLERANCE
                                            {
                                                Some(normalized_response)
                                            } else {
                                                None
                                            }
                                        },
                                    )
                                } else {
                                    None
                                }
                            })
                    })
            };

            let mut series = Vec::new();
            if show_legend {
                let final_low_response = process_response(
                    &low_mask,
                    valid_stacked_responses,
                    response_length_samples,
                    current_ss_start_idx,
                    current_ss_end_idx,
                    post_averaging_smoothing_window,
                );
                let final_high_response = process_response(
                    &high_mask,
                    valid_stacked_responses,
                    response_length_samples,
                    current_ss_start_idx,
                    current_ss_end_idx,
                    post_averaging_smoothing_window,
                );
                // The "Combined" response uses all QC'd windows.
                let final_combined_response = process_response(
                    &combined_mask,
                    valid_stacked_responses,
                    response_length_samples,
                    current_ss_start_idx,
                    current_ss_end_idx,
                    post_averaging_smoothing_window,
                );

                if let Some(resp) = final_low_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str =
                        peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{p:.2}"));
                    let latency_str = latency_opt.map_or_else(
                        || "N/A".to_string(),
                        |l_s| format!("{:.0} ms", l_s * 1000.0),
                    );
                    series.push(PlotSeries {
                        data: response_time
                            .iter()
                            .zip(resp.iter())
                            .map(|(&t, &v)| (t, v))
                            .collect(),
                        label: format!(
                            "< {setpoint_threshold} deg/s (Peak: {peak_str}, Td: {latency_str})"
                        ),
                        color: color_low_sp,
                        stroke_width: line_stroke_plot,
                    });
                }
                if let Some(resp) = final_high_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str =
                        peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{p:.2}"));
                    let latency_str = latency_opt.map_or_else(
                        || "N/A".to_string(),
                        |l_s| format!("{:.0} ms", l_s * 1000.0),
                    );
                    series.push(PlotSeries {
                        data: response_time
                            .iter()
                            .zip(resp.iter())
                            .map(|(&t, &v)| (t, v))
                            .collect(),
                        label: format!(
                            "\u{2265} {setpoint_threshold} deg/s (Peak: {peak_str}, Td: {latency_str})"
                        ),
                        color: color_high_sp,
                        stroke_width: line_stroke_plot,
                    });
                }
                if let Some(resp) = final_combined_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str =
                        peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{p:.2}"));
                    let latency_str = latency_opt.map_or_else(
                        || "N/A".to_string(),
                        |l_s| format!("{:.0} ms", l_s * 1000.0),
                    );
                    series.push(PlotSeries {
                        data: response_time
                            .iter()
                            .zip(resp.iter())
                            .map(|(&t, &v)| (t, v))
                            .collect(),
                        label: format!("Combined (Peak: {peak_str}, Td: {latency_str})"), // This is the average of all Y-corrected & QC'd responses
                        color: color_combined,
                        stroke_width: line_stroke_plot,
                    });
                }
            } else {
                // If not showing legend, only plot the "Combined" (average of all QC'd responses)
                let final_combined_response = process_response(
                    &combined_mask,
                    valid_stacked_responses,
                    response_length_samples,
                    current_ss_start_idx,
                    current_ss_end_idx,
                    post_averaging_smoothing_window,
                );
                if let Some(resp) = final_combined_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str =
                        peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{p:.2}"));
                    let latency_str = latency_opt.map_or_else(
                        || "N/A".to_string(),
                        |l_s| format!("{:.0} ms", l_s * 1000.0),
                    );
                    series.push(PlotSeries {
                        data: response_time
                            .iter()
                            .zip(resp.iter())
                            .map(|(&t, &v)| (t, v))
                            .collect(),
                        label: format!("step-response (Peak: {peak_str}, Td: {latency_str})"), // This is the average of all Y-corrected & QC'd responses
                        color: color_combined,
                        stroke_width: line_stroke_plot,
                    });
                }
            }

            if series.is_empty() {
                // eprintln!("Debug: Axis {axis_index} has no series to plot after process_response.");
                continue; // No valid series generated for this axis
            }

            // Calculate y-range from the actual series data
            for s_data in &series {
                for &(_, v) in &s_data.data {
                    if v.is_finite() {
                        global_resp_min = global_resp_min.min(v);
                        global_resp_max = global_resp_max.max(v);
                    }
                }
            }

            // Add current P:D ratio with quality assessment as legend entries for Roll/Pitch
            if axis_index < 2 {
                // Current P:D ratio and assessment
                if let Some(current_pd) = current_pd_ratios[axis_index] {
                    let current_label = if let Some(assessment) = assessments[axis_index] {
                        if let Some(peak) = peak_values[axis_index] {
                            format!(
                                "Current P:D={:.2} (Peak={:.2}, {})",
                                current_pd, peak, assessment
                            )
                        } else {
                            format!("Current P:D={:.2} ({})", current_pd, assessment)
                        }
                    } else {
                        format!("Current P:D={:.2}", current_pd)
                    };
                    series.push(PlotSeries {
                        data: vec![],
                        label: current_label,
                        color: RGBColor(60, 60, 60), // Darker gray for current
                        stroke_width: 0,             // Invisible legend line
                    });
                }

                // Conservative recommendation (uses dmax_enabled computed at function start)
                if let Some(rec_pd) = recommended_pd_conservative[axis_index] {
                    let recommendation_label = if dmax_enabled {
                        // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                        let d_min_str = recommended_d_min_conservative[axis_index]
                            .map_or("N/A".to_string(), |v| v.to_string());
                        let d_max_str = recommended_d_max_conservative[axis_index]
                            .map_or("N/A".to_string(), |v| v.to_string());
                        format!(
                            "Conservative: P:D={:.2} (D-Min≈{}, D-Max≈{})",
                            rec_pd, d_min_str, d_max_str
                        )
                    } else if let Some(rec_d) = recommended_d_conservative[axis_index] {
                        // D-Min/D-Max disabled: show only base D
                        format!("Conservative: P:D={:.2} (D≈{})", rec_pd, rec_d)
                    } else {
                        format!("Conservative: P:D={:.2}", rec_pd)
                    };
                    series.push(PlotSeries {
                        data: vec![],
                        label: recommendation_label,
                        color: RGBColor(100, 100, 100), // Medium gray for conservative
                        stroke_width: 0,                // Invisible legend line
                    });
                }

                // Moderate recommendation
                if let Some(rec_pd) = recommended_pd_aggressive[axis_index] {
                    let recommendation_label = if dmax_enabled {
                        // D-Min/D-Max enabled: show D-Min and D-Max, NOT base D
                        let d_min_str = recommended_d_min_aggressive[axis_index]
                            .map_or("N/A".to_string(), |v| v.to_string());
                        let d_max_str = recommended_d_max_aggressive[axis_index]
                            .map_or("N/A".to_string(), |v| v.to_string());
                        format!(
                            "Moderate:     P:D={:.2} (D-Min≈{}, D-Max≈{})",
                            rec_pd, d_min_str, d_max_str
                        )
                    } else if let Some(rec_d) = recommended_d_aggressive[axis_index] {
                        // D-Min/D-Max disabled: show only base D
                        format!("Moderate:     P:D={:.2} (D≈{})", rec_pd, rec_d)
                    } else {
                        format!("Moderate:     P:D={:.2}", rec_pd)
                    };
                    series.push(PlotSeries {
                        data: vec![],
                        label: recommendation_label,
                        color: RGBColor(70, 70, 70), // Darker gray for moderate
                        stroke_width: 0,             // Invisible legend line
                    });
                }
            }

            // Store title for later use
            let mut title = format!("{} Step Response", AXIS_NAMES[axis_index]);
            if let Some(axis_pid) = pid_metadata.get_axis(axis_index) {
                let firmware_type = pid_metadata.get_firmware_type();
                let pid_info = axis_pid.format_for_title(firmware_type, dmax_enabled);
                if !pid_info.is_empty() {
                    title.push_str(&pid_info);
                }
            }
            if has_nonzero_f_term_data[axis_index] {
                title.push_str(" - Invalid due to Feed-Forward");
            }

            temp_axis_data[axis_index] = Some((title, series));
        }
    }

    // Calculate unified Y-axis range across ALL axes for symmetric scaling (issue #115)
    let (final_resp_min, final_resp_max) =
        if global_resp_min.is_finite() && global_resp_max.is_finite() {
            // Simple symmetric range expansion with 10% padding
            let range = (global_resp_max - global_resp_min).max(0.1);
            let mid = (global_resp_max + global_resp_min) / 2.0;
            let half_range = range * 0.55; // 10% padding = 1.1/2 = 0.55
            (mid - half_range, mid + half_range)
        } else {
            // Default range if no valid data
            eprintln!("Warning: No finite step response data found. Using default range.");
            (-0.2, 1.8) // A reasonable default for normalized step responses
        };

    let x_range = 0f64..step_response_plot_duration_s * 1.05;
    let y_range = final_resp_min..final_resp_max;

    // Now populate plot_data_per_axis with unified Y-axis range
    for axis_index in 0..axis_count {
        if let Some((title, series)) = temp_axis_data[axis_index].take() {
            plot_data_per_axis[axis_index] = Some((
                title,
                x_range.clone(),
                y_range.clone(),
                series,
                "Time (s)".to_string(),
                "Normalized Response".to_string(),
            ));
        }
    }

    draw_stacked_plot(
        &output_file_step,
        root_name,
        plot_type_name,
        move |axis_idx_for_closure| plot_data_per_axis[axis_idx_for_closure].as_ref().cloned(),
    )
}
