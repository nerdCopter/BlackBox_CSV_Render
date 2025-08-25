// src/plot_functions/plot_step_response.rs

use std::error::Error;
use plotters::style::RGBColor;
use ndarray::{Array1, Array2, s};

use crate::plot_framework::{draw_stacked_plot, PlotSeries, calculate_range};
use crate::constants::{
    RESPONSE_LENGTH_S,
    POST_AVERAGING_SMOOTHING_WINDOW, STEADY_STATE_START_S, STEADY_STATE_END_S,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT, FINAL_NORMALIZED_STEADY_STATE_TOLERANCE
};
use crate::data_analysis::calc_step_response; // For average_responses and moving_average_smooth_f64
use crate::data_input::pid_metadata::PidMetadata;

/// Generates the Stacked Step Response Plot (Blue, Orange, Red)
pub fn plot_step_response(
    step_response_results: &[Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3],
    root_name: &str,
    sample_rate: Option<f64>,
    has_nonzero_f_term_data: &[bool; 3],
    setpoint_threshold: f64,
    show_legend: bool,
    pid_metadata: &PidMetadata,
    // Add axis_index_for_debug if you uncomment debug prints in process_response
    // axis_index_for_debug: usize, // Uncomment if using debug prints
) -> Result<(), Box<dyn Error>> {
    let step_response_plot_duration_s = RESPONSE_LENGTH_S;
    let steady_state_start_s_const = STEADY_STATE_START_S; // from constants
    let steady_state_end_s_const = STEADY_STATE_END_S;     // from constants
    let post_averaging_smoothing_window = POST_AVERAGING_SMOOTHING_WINDOW; // from constants
    let color_high_sp: RGBColor = *COLOR_STEP_RESPONSE_HIGH_SP;
    let color_combined: RGBColor = *COLOR_STEP_RESPONSE_COMBINED;
    let color_low_sp: RGBColor = *COLOR_STEP_RESPONSE_LOW_SP;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    let output_file_step = if show_legend {
        format!("{}_Step_Response_stacked_plot_{}s_{}dps.png", root_name, step_response_plot_duration_s, setpoint_threshold)
    } else {
        format!("{}_Step_Response_stacked_plot_{}s.png", root_name, step_response_plot_duration_s)
    };
    let plot_type_name = "Step Response";
    let sr = sample_rate.unwrap_or(1000.0); // Default sample rate if not provided

    let mut plot_data_per_axis: [Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)>; 3] = Default::default();

    for axis_index in 0..3 {
        if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) = &step_response_results[axis_index] {
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
            let current_ss_end_idx = ss_end_idx.min(response_length_samples).max(current_ss_start_idx + 1);


            if current_ss_start_idx >= current_ss_end_idx {
                eprintln!("Warning: Axis {} Step Response: Steady-state window is invalid (start_idx {} >= end_idx {} for response length {}). Skipping final normalization and plot for this axis.",
                    axis_index, current_ss_start_idx, current_ss_end_idx, response_length_samples);
                 continue;
            }

            let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() < setpoint_threshold as f32 { 1.0 } else { 0.0 });
            let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() >= setpoint_threshold as f32 { 1.0 } else { 0.0 });
            let combined_mask: Array1<f32> = Array1::ones(num_qc_windows);

            let process_response = |
                mask: &Array1<f32>,
                stacked_resp: &Array2<f32>,
                resp_len_samples_local: usize, // Use a local variable for clarity
                ss_start_idx_local: usize,
                ss_end_idx_local: usize,
                smoothing_window: usize,
            | -> Option<Array1<f64>> { // Return type simplified to just the response array
                if !mask.iter().any(|&w| w > 0.0) { return None; } // No windows selected by mask

                // avg_resp is now an average of *individually Y-corrected* (mostly normalized towards 1) responses
                // from calc_step_response.rs
                calc_step_response::average_responses(stacked_resp, mask, resp_len_samples_local)
                    .ok()
                    .and_then(|avg_resp| {
                         if avg_resp.is_empty() { return None; }

                         let smoothed_resp = calc_step_response::moving_average_smooth_f64(&avg_resp, smoothing_window);
                         if smoothed_resp.is_empty() { return None; }

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
                         let steady_state_segment = shifted_response.slice(s![ss_start_idx_local..ss_end_idx_local]);
                         if steady_state_segment.is_empty() { // Guard against empty slice
                            return None;
                         }

                         steady_state_segment.mean()
                            .and_then(|final_ss_mean_for_norm| {
                                 if final_ss_mean_for_norm.abs() > 1e-9 { // Avoid division by zero
                                      // This final normalization step ensures the plotted response aims for 1.0.
                                      // It refines the average of the Y-corrected individual responses.
                                      let normalized_response = shifted_response.mapv(|v| v / final_ss_mean_for_norm);

                                      // Final check on this fully processed and normalized response
                                      let final_check_ss_segment = normalized_response.slice(s![ss_start_idx_local..ss_end_idx_local]);
                                      if final_check_ss_segment.is_empty() { return None; } // Should not happen

                                      final_check_ss_segment.mean()
                                        .and_then(|final_normalized_ss_mean| {
                                             if (final_normalized_ss_mean - 1.0).abs() <= FINAL_NORMALIZED_STEADY_STATE_TOLERANCE {
                                                 Some(normalized_response)
                                             } else {
                                                 None
                                             }
                                        })
                                 } else {
                                    None
                                 }
                            })
                    })
            };

            let mut series = Vec::new();
            if show_legend {
                let final_low_response = process_response(&low_mask, valid_stacked_responses, response_length_samples, current_ss_start_idx, current_ss_end_idx, post_averaging_smoothing_window);
                let final_high_response = process_response(&high_mask, valid_stacked_responses, response_length_samples, current_ss_start_idx, current_ss_end_idx, post_averaging_smoothing_window);
                // The "Combined" response uses all QC'd windows.
                let final_combined_response = process_response(&combined_mask, valid_stacked_responses, response_length_samples, current_ss_start_idx, current_ss_end_idx, post_averaging_smoothing_window);

                if let Some(resp) = final_low_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str = peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{:.2}", p));
                    let latency_str = latency_opt.map_or_else(|| "N/A".to_string(), |l_s| format!("{:.0} ms", l_s * 1000.0));
                    series.push(PlotSeries {
                        data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                        label: format!("< {} deg/s (Peak: {}, Td: {})", setpoint_threshold, peak_str, latency_str),
                        color: color_low_sp,
                        stroke_width: line_stroke_plot,
                    });
                }
                if let Some(resp) = final_high_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str = peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{:.2}", p));
                    let latency_str = latency_opt.map_or_else(|| "N/A".to_string(), |l_s| format!("{:.0} ms", l_s * 1000.0));
                    series.push(PlotSeries {
                        data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                        label: format!("\u{2265} {} deg/s (Peak: {}, Td: {})", setpoint_threshold, peak_str, latency_str),
                        color: color_high_sp,
                        stroke_width: line_stroke_plot,
                    });
                }
                if let Some(resp) = final_combined_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str = peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{:.2}", p));
                    let latency_str = latency_opt.map_or_else(|| "N/A".to_string(), |l_s| format!("{:.0} ms", l_s * 1000.0));
                     series.push(PlotSeries {
                        data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                        label: format!("Combined (Peak: {}, Td: {})", peak_str, latency_str), // This is the average of all Y-corrected & QC'd responses
                        color: color_combined,
                        stroke_width: line_stroke_plot,
                    });
                }
            } else { // If not showing legend, only plot the "Combined" (average of all QC'd responses)
                let final_combined_response = process_response(&combined_mask, valid_stacked_responses, response_length_samples, current_ss_start_idx, current_ss_end_idx, post_averaging_smoothing_window);
                if let Some(resp) = final_combined_response {
                    let peak_val_opt = calc_step_response::find_peak_value(&resp);
                    let latency_opt = calc_step_response::calculate_delay_time(&resp, sr);
                    let peak_str = peak_val_opt.map_or_else(|| "N/A".to_string(), |p| format!("{:.2}", p));
                    let latency_str = latency_opt.map_or_else(|| "N/A".to_string(), |l_s| format!("{:.0} ms", l_s * 1000.0));
                    series.push(PlotSeries {
                        data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                        label: format!("step-response (Peak: {}, Td: {})", peak_str, latency_str), // This is the average of all Y-corrected & QC'd responses
                        color: color_combined,
                        stroke_width: line_stroke_plot,
                    });
                }
            }

            if series.is_empty() {
                // eprintln!("Debug: Axis {} has no series to plot after process_response.", axis_index);
                continue; // No valid series generated for this axis
            }

            // Calculate y-range from the actual series data
            let mut resp_min = f64::INFINITY;
            let mut resp_max = f64::NEG_INFINITY;
            for s_data in &series {
                for &(_, v) in &s_data.data {
                    if v.is_finite() {
                        resp_min = resp_min.min(v);
                        resp_max = resp_max.max(v);
                    }
                }
            }

            let (final_resp_min, final_resp_max) = if resp_min.is_finite() && resp_max.is_finite() {
                calculate_range(resp_min, resp_max)
            } else {
                // Default range if no valid data, or handle as error
                eprintln!("Warning: Axis {} has no finite data for Y-axis range calculation. Using default range.", axis_index);
                (-0.2, 1.8) // A reasonable default for normalized step responses
            };

            let x_range = 0f64..step_response_plot_duration_s * 1.05; // Add a little padding to x-axis
            let y_range = final_resp_min..final_resp_max;

            plot_data_per_axis[axis_index] = Some((
                {
                    let mut title = format!("Axis {} Step Response", axis_index);
                    
                    // Add PID information to the title using firmware-specific terminology
                    if let Some(axis_pid) = pid_metadata.get_axis(axis_index) {
                        let firmware_type = pid_metadata.get_firmware_type();
                        let pid_info = axis_pid.format_for_title(firmware_type);
                        if !pid_info.is_empty() {
                            title.push_str(&pid_info);
                        }
                    }
                    
                    // Keep original invalidity logic from master - same for all firmware types
                    if has_nonzero_f_term_data[axis_index] {
                        title.push_str(" - Invalid due Feed-Forward");
                    }
                    title
                },
                x_range,
                y_range,
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
        move |axis_idx_for_closure| { // Renamed to avoid conflict if axis_index is passed for debug
            plot_data_per_axis[axis_idx_for_closure].take()
        },
    )
}

// src/plot_functions/plot_step_response.rs