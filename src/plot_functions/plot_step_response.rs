// src/plot_functions/plot_step_response.rs

use std::error::Error;
use plotters::style::RGBColor;
use ndarray::{Array1, Array2, s};
use ndarray_stats::QuantileExt;

use crate::plot_framework::{draw_stacked_plot, PlotSeries, calculate_range};
use crate::constants::{
    STEP_RESPONSE_PLOT_DURATION_S, SETPOINT_THRESHOLD,
    POST_AVERAGING_SMOOTHING_WINDOW, STEADY_STATE_START_S, STEADY_STATE_END_S,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT,
};
use crate::data_analysis::calc_step_response; // For average_responses and moving_average_smooth_f64

/// Generates the Stacked Step Response Plot (Blue, Orange, Red)
pub fn plot_step_response(
    step_response_results: &[Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3],
    root_name: &str,
    sample_rate: Option<f64>,
    has_nonzero_f_term_data: &[bool; 3],
) -> Result<(), Box<dyn Error>> {
    let step_response_plot_duration_s = STEP_RESPONSE_PLOT_DURATION_S;
    let steady_state_start_s = STEADY_STATE_START_S;
    let steady_state_end_s = STEADY_STATE_END_S;
    let setpoint_threshold = SETPOINT_THRESHOLD;
    let post_averaging_smoothing_window = POST_AVERAGING_SMOOTHING_WINDOW;
    let color_high_sp: RGBColor = *COLOR_STEP_RESPONSE_HIGH_SP;
    let color_combined: RGBColor = *COLOR_STEP_RESPONSE_COMBINED;
    let color_low_sp: RGBColor = *COLOR_STEP_RESPONSE_LOW_SP;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    let output_file_step = format!("{}_step_response_stacked_plot_{}s.png", root_name, step_response_plot_duration_s);
    let plot_type_name = "Step Response";
    let sr = sample_rate.unwrap_or(1000.0);

    let mut plot_data_per_axis: [Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)>; 3] = Default::default();

    for axis_index in 0..3 {
        if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) = &step_response_results[axis_index] {
            let response_length_samples = response_time.len();
            if response_length_samples == 0 || valid_stacked_responses.shape()[0] == 0 {
                 continue;
            }

            let num_qc_windows = valid_stacked_responses.shape()[0];
            let ss_start_sample = (steady_state_start_s * sr).floor() as usize;
            let ss_end_sample = (steady_state_end_s * sr).ceil() as usize;
            let current_ss_start_sample = ss_start_sample.min(response_length_samples);
            let current_ss_end_sample = ss_end_sample.min(response_length_samples);

            if current_ss_start_sample >= current_ss_end_sample {
                eprintln!("Warning: Axis {} Step Response: Steady-state window is invalid (start >= end). Skipping final normalization and plot for this axis.", axis_index);
                 continue;
            }

            let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() < setpoint_threshold as f32 { 1.0 } else { 0.0 });
            let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() >= setpoint_threshold as f32 { 1.0 } else { 0.0 });
            let combined_mask: Array1<f32> = Array1::ones(num_qc_windows);

            let process_response = |
                mask: &Array1<f32>,
                stacked_resp: &Array2<f32>,
                resp_len_samples: usize,
                ss_start_idx: usize,
                ss_end_idx: usize,
                smoothing_window: usize,
            | -> Option<Array1<f64>> {
                if !mask.iter().any(|&w| w > 0.0) { return None; }
                calc_step_response::average_responses(stacked_resp, mask, resp_len_samples)
                    .ok()
                    .and_then(|avg_resp| {
                         if avg_resp.is_empty() { return None; }
                         let smoothed_resp = calc_step_response::moving_average_smooth_f64(&avg_resp, smoothing_window);
                         if smoothed_resp.is_empty() { return None; }
                         let mut shifted_response = smoothed_resp;
                         let first_val = shifted_response[0];
                         shifted_response.mapv_inplace(|v| v - first_val);
                         let steady_state_segment = shifted_response.slice(s![ss_start_idx..ss_end_idx]);
                         steady_state_segment.mean()
                            .and_then(|steady_state_mean| {
                                 if steady_state_mean.abs() > 1e-9 {
                                      let normalized_response = shifted_response.mapv(|v| v / steady_state_mean);
                                      normalized_response.slice(s![ss_start_idx..ss_end_idx]).mean()
                                        .and_then(|normalized_ss_mean| {
                                            const STEADY_STATE_TOLERANCE: f64 = 0.2;
                                             if (normalized_ss_mean - 1.0).abs() <= STEADY_STATE_TOLERANCE {
                                                 Some(normalized_response)
                                             } else { None }
                                        })
                                 } else { None }
                            })
                    })
            };

            let final_low_response = process_response(&low_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);
            let final_high_response = process_response(&high_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);
            let final_combined_response = process_response(&combined_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);

            let final_low_response_cloned = final_low_response.clone();
            let final_high_response_cloned = final_high_response.clone();
            let final_combined_response_cloned = final_combined_response.clone();

            let is_low_response_valid = final_low_response_cloned.is_some();
            let is_high_response_valid = final_high_response_cloned.is_some();
            let is_combined_response_valid = final_combined_response_cloned.is_some();

            if !(is_low_response_valid || is_high_response_valid) {
                continue;
            }

            let mut resp_min = f64::INFINITY;
            let mut resp_max = f64::NEG_INFINITY;
            if let Some(resp) = &final_low_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            if let Some(resp) = &final_high_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            if is_combined_response_valid {
                 if let Some(resp) = &final_combined_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            }

            let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
            let x_range = 0f64..step_response_plot_duration_s * 1.05;
            let y_range = final_resp_min..final_resp_max;

            let mut series = Vec::new();
            if let Some(resp) = final_low_response {
                 series.push(PlotSeries {
                     data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                     label: format!("< {} deg/s", setpoint_threshold),
                     color: color_low_sp,
                     stroke_width: line_stroke_plot,
                 });
            }
            if let Some(resp) = final_high_response {
                 series.push(PlotSeries {
                     data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                     label: format!("\u{2265} {} deg/s", setpoint_threshold),
                     color: color_high_sp,
                     stroke_width: line_stroke_plot,
                 });
             }
            if is_combined_response_valid {
                 if let Some(resp) = final_combined_response {
                     series.push(PlotSeries {
                         data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                         label: "Combined".to_string(),
                         color: color_combined,
                         stroke_width: line_stroke_plot,
                     });
                 }
            }

             plot_data_per_axis[axis_index] = Some((
                {
                    let mut title = format!("Axis {} Step Response", axis_index);
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
        move |axis_index| {
            plot_data_per_axis[axis_index].take()
        },
    )
}

// src/plot_functions/plot_step_response.rs