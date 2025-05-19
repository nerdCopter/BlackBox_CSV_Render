// src/plotting_utils.rs

use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::style::{RGBColor, IntoFont, Color, ShapeStyle, TextStyle};
use plotters::element::{Text, Rectangle, PathElement};
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::coord::{Shift};
use plotters::series::LineSeries;
use plotters::style::colors::{WHITE, BLACK, RED}; // BLACK is used for spectrogram background

use std::error::Error;

use ndarray::{Array1, Array2, s};
use ndarray_stats::QuantileExt;

use crate::constants::{
    PLOT_WIDTH, PLOT_HEIGHT, STEP_RESPONSE_PLOT_DURATION_S, SETPOINT_THRESHOLD,
    POST_AVERAGING_SMOOTHING_WINDOW, STEADY_STATE_START_S, STEADY_STATE_END_S,
    COLOR_PIDSUM_MAIN, COLOR_PIDERROR_MAIN, COLOR_SETPOINT_MAIN,
    COLOR_SETPOINT_VS_GYRO_SP, COLOR_SETPOINT_VS_GYRO_GYRO,
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT, LINE_WIDTH_LEGEND,
    SPECTROGRAM_THROTTLE_BINS, SPECTROGRAM_FFT_WINDOW_SIZE_TARGET, SPECTROGRAM_MAX_FREQ_HZ,
    SPECTROGRAM_POWER_CLIP_MAX, SPECTROGRAM_COLOR_SCALE, SPECTROGRAM_TEXT_COLOR,
    SPECTROGRAM_GRID_COLOR,
};
use crate::log_data::LogRowData;
use crate::step_response;
use crate::fft_utils;


pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

pub fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let (x_range, y_range) = area.get_pixel_range();
    let (width, height) = ((x_range.end - x_range.start) as u32, (y_range.end - y_range.start) as u32);
    let text_style = ("sans-serif", 20).into_font().color(&RED);
    area.draw(&Text::new(
        format!("Axis {} {} Data Unavailable:\n{}", axis_index, plot_type, reason),
        (width as i32 / 2 - 100, height as i32 / 2 - 20),
        text_style,
    ))?;
    Ok(())
}

#[derive(Clone)]
pub struct PlotSeries {
    pub data: Vec<(f64, f64)>,
    pub label: String,
    pub color: RGBColor,
    pub stroke_width: u32,
}

fn draw_single_axis_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    chart_title: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    x_label: &str,
    y_label: &str,
    series: &[PlotSeries],
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(chart_title, ("sans-serif", 20))
        .margin(5).x_label_area_size(30).y_label_area_size(50)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(10).y_labels(5)
        .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

    let mut series_drawn_count = 0;
    for s_item in series { // Renamed s to s_item to avoid conflict with ndarray::s
        if !s_item.data.is_empty() {
            chart.draw_series(LineSeries::new(
                s_item.data.iter().cloned(),
                s_item.color.stroke_width(s_item.stroke_width),
            ))?
            .label(&s_item.label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s_item.color.stroke_width(LINE_WIDTH_LEGEND)));
            series_drawn_count += 1;
        }
    }

    if series_drawn_count > 0 {
        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
            .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
    }

    Ok(())
}

fn draw_stacked_plot<'a, F>(
    output_filename: &'a str,
    root_name: &str,
    plot_type_name: &str,
    mut get_axis_plot_data: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)> + Send + Sync + 'static,
    <BitMapBackend<'a> as DrawingBackend>::ErrorType: 'static,
{
    let root_area = BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?;

    root_area.draw(&Text::new(
        root_name,
        (10, 10),
        ("sans-serif", 24).into_font().color(&BLACK),
    ))?;

    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((3, 1));
    let mut any_axis_plotted = false;

    for axis_index in 0..3 {
        let area = &sub_plot_areas[axis_index];
        match get_axis_plot_data(axis_index) {
            Some((chart_title, x_range, y_range, series_data, x_label, y_label)) => {
                 let has_data = series_data.iter().any(|s_item| !s_item.data.is_empty());
                 let valid_ranges = x_range.end > x_range.start && y_range.end > y_range.start;

                 if has_data && valid_ranges {
                     draw_single_axis_chart(
                         area,
                         &chart_title,
                         x_range,
                         y_range,
                         &x_label,
                         &y_label,
                         &series_data,
                     )?;
                     any_axis_plotted = true;
                 } else {
                      let reason = if !has_data { "No data points" } else { "Invalid ranges" };
                      draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
                 }
            }
            None => {
                 let reason = "Calculation/Data Extraction Failed";
                 draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
            }
        }
    }

    if any_axis_plotted {
        root_area.present()?;
        println!("  Stacked plot saved as '{}'.", output_filename);
    } else {
        root_area.present()?;
        println!("  Skipping '{}' plot saving: No data available for any axis to plot, only placeholder messages shown.", output_filename);
    }

    Ok(())
}

pub fn plot_pidsum_error_setpoint(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_pidsum_error = format!("{}_PIDsum_PIDerror_Setpoint_stacked.png", root_name);
    let plot_type_name = "PIDsum/PIDerror/Setpoint";
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                 let pidsum = row.p_term[axis_index].and_then(|p| {
                    row.i_term[axis_index].and_then(|i| {
                        row.d_term[axis_index].map(|d_val| p + i + d_val) // Renamed d to d_val
                    })
                });
                axis_plot_data[axis_index].push((time, row.setpoint[axis_index], row.gyro[axis_index], pidsum));
            }
        }
    }

    let color_pidsum: RGBColor = *COLOR_PIDSUM_MAIN;
    let color_pid_error: RGBColor = *COLOR_PIDERROR_MAIN;
    let color_setpoint: RGBColor = *COLOR_SETPOINT_MAIN;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_pidsum_error,
        root_name,
        plot_type_name,
        move |axis_index| {
            let data = &axis_plot_data[axis_index];
            if data.is_empty() {
                 return None;
            }

            let mut pidsum_series_data: Vec<(f64, f64)> = Vec::new();
            let mut setpoint_series_data: Vec<(f64, f64)> = Vec::new();
            let mut pid_error_series_data: Vec<(f64, f64)> = Vec::new();

            let mut time_min = f64::INFINITY;
            let mut time_max = f64::NEG_INFINITY;
            let mut val_min = f64::INFINITY;
            let mut val_max = f64::NEG_INFINITY;

            for (time, setpoint, gyro_filt, pidsum) in data {
                time_min = time_min.min(*time);
                time_max = time_max.max(*time);

                if let Some(p) = pidsum {
                    pidsum_series_data.push((*time, *p));
                    val_min = val_min.min(*p);
                    val_max = val_max.max(*p);
                }
                if let Some(s_val) = setpoint { // Renamed s to s_val
                    setpoint_series_data.push((*time, *s_val));
                    val_min = val_min.min(*s_val);
                    val_max = val_max.max(*s_val);
                    if let Some(g) = gyro_filt {
                        let error = s_val - g;
                        pid_error_series_data.push((*time, error));
                        val_min = val_min.min(error);
                        val_max = val_max.max(error);
                    }
                }
            }

            if pidsum_series_data.is_empty() && setpoint_series_data.is_empty() && pid_error_series_data.is_empty() {
                 return None;
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !pidsum_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pidsum_series_data,
                    label: "PIDsum (P+I+D)".to_string(),
                    color: color_pidsum,
                    stroke_width: line_stroke_plot,
                });
            }
            if !pid_error_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pid_error_series_data,
                    label: "PID Error (Setpoint - GyroADC)".to_string(),
                    color: color_pid_error,
                    stroke_width: line_stroke_plot,
                });
            }
            if !setpoint_series_data.is_empty() {
                series.push(PlotSeries {
                    data: setpoint_series_data,
                    label: "Setpoint".to_string(),
                    color: color_setpoint,
                    stroke_width: line_stroke_plot,
                });
            }

            Some((
                format!("Axis {} PIDsum vs PID Error vs Setpoint", axis_index),
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Value".to_string(),
            ))
        },
    )
}

pub fn plot_setpoint_vs_gyro(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_setpoint_gyro = format!("{}_SetpointVsGyro_stacked.png", root_name);
    let plot_type_name = "Setpoint/Gyro";
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
     for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.setpoint[axis_index], row.gyro[axis_index]));
             }
         }
     }

    let color_sp: RGBColor = *COLOR_SETPOINT_VS_GYRO_SP;
    let color_gyro: RGBColor = *COLOR_SETPOINT_VS_GYRO_GYRO;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_setpoint_gyro,
        root_name,
        plot_type_name,
        move |axis_index| {
             let data = &axis_plot_data[axis_index];
             if data.is_empty() {
                 return None;
             }

             let mut setpoint_series_data: Vec<(f64, f64)> = Vec::new();
             let mut gyro_series_data: Vec<(f64, f64)> = Vec::new();

             let mut time_min = f64::INFINITY;
             let mut time_max = f64::NEG_INFINITY;
             let mut val_min = f64::INFINITY;
             let mut val_max = f64::NEG_INFINITY;

             for (time, setpoint, gyro_filt) in data {
                 time_min = time_min.min(*time);
                 time_max = time_max.max(*time);

                 if let Some(s_val) = setpoint { // Renamed s to s_val
                     setpoint_series_data.push((*time, *s_val));
                     val_min = val_min.min(*s_val);
                     val_max = val_max.max(*s_val);
                 }
                 if let Some(g) = gyro_filt {
                     gyro_series_data.push((*time, *g));
                     val_min = val_min.min(*g);
                     val_max = val_max.max(*g);
                 }
             }

            if setpoint_series_data.is_empty() && gyro_series_data.is_empty() {
                 return None;
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !gyro_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: gyro_series_data,
                     label: "Gyro (gyroADC)".to_string(),
                     color: color_gyro,
                     stroke_width: line_stroke_plot,
                 });
            }
            if !setpoint_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: setpoint_series_data,
                     label: "Setpoint".to_string(),
                     color: color_sp,
                     stroke_width: line_stroke_plot,
                 });
             }

            Some((
                format!("Axis {} Setpoint vs Gyro", axis_index),
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Value".to_string(),
            ))
        },
    )
}

pub fn plot_gyro_vs_unfilt(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
    let plot_type_name = "Gyro/UnfiltGyro";
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.gyro[axis_index], row.gyro_unfilt[axis_index]));
             }
         }
     }

    let color_gyro_unfilt: RGBColor = *COLOR_GYRO_VS_UNFILT_UNFILT;
    let color_gyro_filt: RGBColor = *COLOR_GYRO_VS_UNFILT_FILT;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_gyro,
        root_name,
        plot_type_name,
        move |axis_index| {
             let data = &axis_plot_data[axis_index];
             if data.is_empty() {
                 return None;
             }

             let mut filt_series_data: Vec<(f64, f64)> = Vec::new();
             let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();

             let mut time_min = f64::INFINITY;
             let mut time_max = f64::NEG_INFINITY;
             let mut val_min = f64::INFINITY;
             let mut val_max = f64::NEG_INFINITY;

             for (time, gyro_filt, gyro_unfilt) in data {
                 time_min = time_min.min(*time);
                 time_max = time_max.max(*time);

                 if let Some(gf) = gyro_filt {
                     filt_series_data.push((*time, *gf));
                     val_min = val_min.min(*gf);
                     val_max = val_max.max(*gf);
                 }
                 if let Some(gu) = gyro_unfilt {
                     unfilt_series_data.push((*time, *gu));
                     val_min = val_min.min(*gu);
                     val_max = val_max.max(*gu);
                 }
             }

            if filt_series_data.is_empty() && unfilt_series_data.is_empty() {
                 return None;
            }


            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !unfilt_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: unfilt_series_data,
                     label: "Unfiltered Gyro (gyroUnfilt/debug)".to_string(),
                     color: color_gyro_unfilt,
                     stroke_width: line_stroke_plot,
                 });
            }
            if !filt_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: filt_series_data,
                     label: "Filtered Gyro (gyroADC)".to_string(),
                     color: color_gyro_filt,
                     stroke_width: line_stroke_plot,
                 });
            }

            Some((
                format!("Axis {} Filtered vs Unfiltered Gyro", axis_index),
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Gyro Value".to_string(),
            ))
        },
    )
}

pub fn plot_step_response(
    step_response_results: &[Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let step_response_plot_duration_s = STEP_RESPONSE_PLOT_DURATION_S;
    let steady_state_start_s_const = STEADY_STATE_START_S; // Renamed to avoid conflict
    let steady_state_end_s_const = STEADY_STATE_END_S;     // Renamed to avoid conflict
    let setpoint_threshold_const = SETPOINT_THRESHOLD;     // Renamed to avoid conflict
    let post_averaging_smoothing_window_const = POST_AVERAGING_SMOOTHING_WINDOW; // Renamed
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
            let ss_start_sample = (steady_state_start_s_const * sr).floor() as usize;
            let ss_end_sample = (steady_state_end_s_const * sr).ceil() as usize;
            let current_ss_start_sample = ss_start_sample.min(response_length_samples);
            let current_ss_end_sample = ss_end_sample.min(response_length_samples);

            if current_ss_start_sample >= current_ss_end_sample {
                eprintln!("Warning: Axis {} Step Response: Steady-state window is invalid (start >= end). Skipping final normalization and plot for this axis.", axis_index);
                 continue;
            }

            let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() < setpoint_threshold_const as f32 { 1.0 } else { 0.0 });
            let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v.abs() >= setpoint_threshold_const as f32 { 1.0 } else { 0.0 });
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
                step_response::average_responses(stacked_resp, mask, resp_len_samples)
                    .ok()
                    .and_then(|avg_resp| {
                         if avg_resp.is_empty() { return None; }
                         let smoothed_resp = step_response::moving_average_smooth_f64(&avg_resp, smoothing_window);
                         if smoothed_resp.is_empty() { return None; }
                         let mut shifted_response = smoothed_resp;
                         if shifted_response.is_empty() { return None; } // Guard against empty after smoothing
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

            let final_low_response = process_response(&low_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window_const);
            let final_high_response = process_response(&high_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window_const);
            let final_combined_response = process_response(&combined_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window_const);

            let final_low_response_cloned = final_low_response.clone();
            let final_high_response_cloned = final_high_response.clone();
            let final_combined_response_cloned = final_combined_response.clone();

            let is_low_response_valid = final_low_response_cloned.is_some();
            let is_high_response_valid = final_high_response_cloned.is_some();
            let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some();


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
                     label: format!("< {} deg/s", setpoint_threshold_const),
                     color: color_low_sp,
                     stroke_width: line_stroke_plot,
                 });
            }
            if let Some(resp) = final_high_response {
                 series.push(PlotSeries {
                     data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                     label: format!("\u{2265} {} deg/s", setpoint_threshold_const),
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
                format!("Axis {} Step Response", axis_index),
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

fn get_spectrogram_color(value: f32, min_clip: f32, max_clip: f32) -> RGBColor {
    let normalized_value = ((value - min_clip) / (max_clip - min_clip).max(1e-6)).clamp(0.0, 1.0);

    if normalized_value <= SPECTROGRAM_COLOR_SCALE[0].0 { // Check against the threshold of the first color
        return SPECTROGRAM_COLOR_SCALE[0].1;
    }

    for i in 0..(SPECTROGRAM_COLOR_SCALE.len() - 1) {
        let (p1, c1) = SPECTROGRAM_COLOR_SCALE[i];
        let (p2, c2_val) = SPECTROGRAM_COLOR_SCALE[i + 1]; // Use the color from the next point for mixing
        if normalized_value <= p2 {
            let t_f32 = (normalized_value - p1) / (p2 - p1).max(1e-6); // Interpolation factor as f32

            let r1 = c1.rgb().0 as f32;
            let g1 = c1.rgb().1 as f32;
            let b1 = c1.rgb().2 as f32;

            let r2 = c2_val.rgb().0 as f32;
            let g2 = c2_val.rgb().1 as f32;
            let b2 = c2_val.rgb().2 as f32;

            let r_mixed = (r1 * (1.0 - t_f32) + r2 * t_f32).round().clamp(0.0, 255.0) as u8;
            let g_mixed = (g1 * (1.0 - t_f32) + g2 * t_f32).round().clamp(0.0, 255.0) as u8;
            let b_mixed = (b1 * (1.0 - t_f32) + b2 * t_f32).round().clamp(0.0, 255.0) as u8;

            return RGBColor(r_mixed, g_mixed, b_mixed);
        }
    }
    SPECTROGRAM_COLOR_SCALE.last().unwrap().1 // Return the last color if normalized_value is > last threshold
}

fn draw_single_throttle_spectrogram<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title_prefix: &str,
    axis_name: &str,
    psd_matrix: &Array2<f32>, // (num_freq_bins, num_throttle_bins)
    freq_bins: &Array1<f32>,  // Hz values for frequency bins
    throttle_bins: &Array1<f32>, // % values for throttle bins
) -> Result<(), Box<dyn Error>>
where DB::ErrorType: 'static
{
    area.fill(&plotters::style::colors::BLACK)?; // Use explicit BLACK here

    // X-axis: Frequency (Hz), Y-axis: Throttle (%)
    let x_range_spec = 0f32..SPECTROGRAM_MAX_FREQ_HZ;
    let y_range_spec = 0f32..100f32;

    let mut chart = ChartBuilder::on(area)
        .caption(format!("{} {}", title_prefix, axis_name), TextStyle::from(("sans-serif", 18).into_font().color(SPECTROGRAM_TEXT_COLOR)))
        .margin_top(25)
        .margin_right(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range_spec.clone(), y_range_spec.clone())?;

    chart.configure_mesh()
        .axis_style(ShapeStyle::from(SPECTROGRAM_TEXT_COLOR).stroke_width(1))
        .x_desc("Frequency (Hz)")
        .y_desc("% Throttle")
        .label_style(("sans-serif", 12).into_font().color(SPECTROGRAM_TEXT_COLOR))
        .light_line_style(&SPECTROGRAM_GRID_COLOR)
        .bold_line_style(&SPECTROGRAM_GRID_COLOR)
        .x_labels(10) // Number of labels on X axis (Frequency)
        .y_labels(5)  // Number of labels on Y axis (Throttle)
        .draw()?;

    let (num_freq_bins_total, num_throttle_plot_bins) = psd_matrix.dim();
    if num_freq_bins_total == 0 || num_throttle_plot_bins == 0 || freq_bins.len() != num_freq_bins_total || throttle_bins.len() != num_throttle_plot_bins {
        let text_style = ("sans-serif", 16).into_font().color(&RED);
        // Adjust text position based on new axes
        let text_x = x_range_spec.start + (x_range_spec.end - x_range_spec.start) * 0.1; // 10% into Freq range
        let text_y = y_range_spec.start + (y_range_spec.end - y_range_spec.start) * 0.5; // 50% into Throttle range
        chart.plotting_area().draw(&Text::new(
            "Spectrogram data empty or mismatched",
            (text_x, text_y),
            text_style,
        ))?;
        return Ok(());
    }

    let throttle_cell_height = if num_throttle_plot_bins > 0 {
        100.0 / num_throttle_plot_bins as f32
    } else {
        100.0 // Fallback, should be guarded by the empty check above
    };

    let mut max_power_overall = 0.0f32;
    let mut total_power_sum = 0.0f32;
    let mut power_count = 0;

    // Iterate over throttle bins (for Y axis position)
    for j_throttle_idx in 0..num_throttle_plot_bins {
        let throttle_center = throttle_bins[j_throttle_idx];
        let y0_throttle = throttle_center - throttle_cell_height / 2.0;
        let y1_throttle = throttle_center + throttle_cell_height / 2.0;

        // Iterate over frequency bins (for X axis position)
        for i_freq_idx in 0..num_freq_bins_total {
            let freq_center = freq_bins[i_freq_idx];

            // Skip if frequency is beyond the plot's X-axis limit
            if freq_center > SPECTROGRAM_MAX_FREQ_HZ + 1e-3 { // freq_center is X
                continue;
            }
            
            let freq_cell_width = if num_freq_bins_total > 0 {
                if i_freq_idx + 1 < num_freq_bins_total {
                    freq_bins[i_freq_idx + 1] - freq_bins[i_freq_idx]
                } else if i_freq_idx > 0 { // Last bin, use width of previous
                    freq_bins[i_freq_idx] - freq_bins[i_freq_idx - 1]
                } else { // Single frequency bin
                    SPECTROGRAM_MAX_FREQ_HZ / num_freq_bins_total.max(1) as f32
                }
            } else {
                SPECTROGRAM_MAX_FREQ_HZ // Fallback
            };

            let x0_freq = freq_center - freq_cell_width / 2.0;
            let x1_freq = freq_center + freq_cell_width / 2.0;

            let power = psd_matrix[[i_freq_idx, j_throttle_idx]];
            max_power_overall = max_power_overall.max(power);
            if power > 1e-6 { // Consider power significant if > 0
                 total_power_sum += power;
                 power_count += 1;
            }

            let color = get_spectrogram_color(power, 0.0, SPECTROGRAM_POWER_CLIP_MAX);
            // Use explicit BLACK from constants or plotters for comparison
            if color != SPECTROGRAM_COLOR_SCALE[0].1 { // Don't draw if color is same as background (first color in scale)
                let rect_coords = [
                    (x0_freq.max(0.0), y0_throttle.max(0.0)), // (x0, y0)
                    (x1_freq.min(SPECTROGRAM_MAX_FREQ_HZ), y1_throttle.min(100.0)) // (x1, y1)
                ];
                let rect = Rectangle::new(rect_coords, color.filled());
                chart.plotting_area().draw(&rect)?;
            }
        }
    }

    let mean_power = if power_count > 0 { total_power_sum / power_count as f32 } else { 0.0 };

    let text_style_spec = ("sans-serif", 12).into_font().color(SPECTROGRAM_TEXT_COLOR);
    // Position text: X is Frequency axis, Y is Throttle axis
    let text_pos_x = x_range_spec.end * 0.65_f32; // 65% of max frequency
    let text_pos_y = y_range_spec.end * 0.9_f32; // 90% of max throttle (near top of plot)
    chart.plotting_area().draw(&Text::new(
        format!("mean={:.4}\npeak={:.4}", mean_power, max_power_overall),
        (text_pos_x, text_pos_y),
        text_style_spec,
    ))?;

    Ok(())
}


fn draw_spectrogram_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title: &str,
) -> Result<(), Box<dyn Error>>
where DB::ErrorType: 'static {
    area.fill(&WHITE)?;

    // Split the area: top for title text, bottom for the color strip chart
    // Allocate roughly 40% height for title, 60% for strip + labels
    let title_height_ratio = 0.40;
    let title_pixel_height = (area.dim_in_pixel().1 as f32 * title_height_ratio).round() as u32;

    let (title_area, strip_chart_area) = area.split_vertically(title_pixel_height); 

    // Draw title text manually in the top part
    title_area.draw(&Text::new(
        title,
        (5, (title_pixel_height as i32 / 2) - 6), // Center vertically in its area
        ("sans-serif", 12).into_font().color(&plotters::style::colors::BLACK),
    ))?;

    // Use the bottom part for the color strip chart
    let mut chart = ChartBuilder::on(&strip_chart_area)
        .margin(0) // No margin for the chart itself
        .x_label_area_size(15) // Minimal space for X-axis numbers (colorbar scale)
        .y_label_area_size(0)  // No Y-axis labels
        .build_cartesian_2d(0f32..SPECTROGRAM_POWER_CLIP_MAX, 0f32..1f32 /* Y range is nominal for strip */)?;

    chart
        .configure_mesh()
        .disable_y_axis()
        .disable_y_mesh()
        .disable_x_mesh() // No vertical grid lines on colorbar
        .x_labels(5) // Fewer labels on colorbar
        .x_label_formatter(&|x| format!("{:.0}", x)) 
        .label_style(("sans-serif", 10).into_font().color(&plotters::style::colors::BLACK))
        .draw()?;


    let x_coord_range = chart.x_range();
    let x_start = x_coord_range.start;
    let x_end = x_coord_range.end;
    let num_steps = 200; 
    let step_width = (x_end - x_start) / num_steps as f32;

    for i in 0..num_steps {
        let val_start = x_start + i as f32 * step_width;
        let val_end = x_start + (i + 1) as f32 * step_width;
        let mid_val = (val_start + val_end) / 2.0;

        let color = get_spectrogram_color(mid_val, 0.0, SPECTROGRAM_POWER_CLIP_MAX);
        // Draw the rect to fill the entire plotting area of this small chart
        let rect = Rectangle::new([(val_start, 0.0), (val_end, 1.0)], color.filled());
        chart.plotting_area().draw(&rect)?;
    }
    Ok(())
}

pub fn plot_throttle_spectrograms(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let sr = match sample_rate {
        Some(s_rate) => s_rate,
        None => {
            println!("Warning: Sample rate unknown, skipping throttle spectrograms.");
            return Ok(());
        }
    };

    let output_filename = format!("{}_throttle_spectrograms.png", root_name);
    let root_area = BitMapBackend::new(&output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Adjust heights
    let main_title_area_height = 50; 
    let colorbar_area_height = 35; // Overall height for colorbar area (title + strip + labels)
    let top_padding = 5;  // Padding below main title
    let bottom_padding = 5; // Padding above plot grid

    // Layout: Title -> Padding -> Colorbar -> Padding -> Plots
    let (main_title_area, temp_area1) = root_area.split_vertically(main_title_area_height);
    let (padding_top_area, temp_area2) = temp_area1.split_vertically(top_padding);
    let (colorbar_area, temp_area3) = temp_area2.split_vertically(colorbar_area_height);
    let (padding_bottom_area, plot_grid_area) = temp_area3.split_vertically(bottom_padding);
    
    // Fill padding areas with white
    padding_top_area.fill(&WHITE)?;
    padding_bottom_area.fill(&WHITE)?;


    main_title_area.draw(&Text::new(
        format!("{} Throttle Spectrograms", root_name),
        (10, 10), // Position within main_title_area
        ("sans-serif", 24).into_font().color(&plotters::style::colors::BLACK),
    ))?;

    // Margin for colorbar to control width and position within its allocated area
    draw_spectrogram_colorbar(&colorbar_area.margin(0, PLOT_WIDTH / 4, 0, PLOT_WIDTH / 4), "Scale (Power)")?;


    let axis_plot_areas = plot_grid_area.split_evenly((3, 1)); // 3 rows for Roll, Pitch, Yaw
    let axis_names = ["Roll", "Pitch", "Yaw"];
    let mut any_spectrogram_plotted = false;

    for axis_index in 0..3 {
        let row_area = &axis_plot_areas[axis_index];
        // Each row is split into two columns: Unfiltered and Filtered
        let (unfilt_area, filt_area) = row_area.split_horizontally(PLOT_WIDTH / 2);

        // --- Unfiltered Spectrogram ---
        let mut gyro_unfilt_data: Vec<f32> = Vec::new();
        let mut throttle_data_unfilt: Vec<f32> = Vec::new();
        for row in log_data {
            if let (Some(gu), Some(th)) = (row.gyro_unfilt[axis_index], row.throttle) {
                gyro_unfilt_data.push(gu as f32);
                throttle_data_unfilt.push(th as f32);
            }
        }

        if !gyro_unfilt_data.is_empty() && !throttle_data_unfilt.is_empty() {
            match fft_utils::calculate_throttle_psd(
                &Array1::from(gyro_unfilt_data),
                &Array1::from(throttle_data_unfilt),
                sr,
                SPECTROGRAM_THROTTLE_BINS,
                SPECTROGRAM_FFT_WINDOW_SIZE_TARGET,
            ) {
                Ok((psd_matrix, freq_bins, throttle_bins)) => {
                    draw_single_throttle_spectrogram(
                        &unfilt_area,
                        "Unfiltered Gyro",
                        axis_names[axis_index],
                        &psd_matrix,
                        &freq_bins,
                        &throttle_bins,
                    )?;
                    any_spectrogram_plotted = true;
                }
                Err(e) => {
                    eprintln!("Error calculating unfiltered spectrogram for Axis {}: {}", axis_index, e);
                    let area_clone = unfilt_area.clone(); // Clone to draw error message
                    area_clone.fill(&plotters::style::colors::BLACK)?;
                    area_clone.draw(&Text::new(
                        format!("Unfiltered {} Data Error:\n{}", axis_names[axis_index], e),
                        (20_i32, 20_i32),
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
            let area_clone = unfilt_area.clone();
            area_clone.fill(&plotters::style::colors::BLACK)?;
            area_clone.draw(&Text::new(
                format!("Unfiltered {} No Data", axis_names[axis_index]),
                (20_i32, 20_i32),
                ("sans-serif", 14).into_font().color(&RED),
            ))?;
        }

        // --- Filtered Spectrogram ---
        let mut gyro_filt_data: Vec<f32> = Vec::new();
        let mut throttle_data_filt: Vec<f32> = Vec::new();
        for row in log_data {
            if let (Some(gf), Some(th)) = (row.gyro[axis_index], row.throttle) {
                gyro_filt_data.push(gf as f32);
                throttle_data_filt.push(th as f32);
            }
        }

        if !gyro_filt_data.is_empty() && !throttle_data_filt.is_empty() {
            match fft_utils::calculate_throttle_psd(
                &Array1::from(gyro_filt_data),
                &Array1::from(throttle_data_filt),
                sr,
                SPECTROGRAM_THROTTLE_BINS,
                SPECTROGRAM_FFT_WINDOW_SIZE_TARGET,
            ) {
                Ok((psd_matrix, freq_bins, throttle_bins)) => {
                    draw_single_throttle_spectrogram(
                        &filt_area,
                        "Filtered Gyro",
                        axis_names[axis_index],
                        &psd_matrix,
                        &freq_bins,
                        &throttle_bins,
                    )?;
                    any_spectrogram_plotted = true;
                }
                Err(e) => {
                    eprintln!("Error calculating filtered spectrogram for Axis {}: {}", axis_index, e);
                     let area_clone = filt_area.clone();
                     area_clone.fill(&plotters::style::colors::BLACK)?;
                     area_clone.draw(&Text::new(
                        format!("Filtered {} Data Error:\n{}", axis_names[axis_index], e),
                        (20_i32, 20_i32),
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
            let area_clone = filt_area.clone();
            area_clone.fill(&plotters::style::colors::BLACK)?;
            area_clone.draw(&Text::new(
                format!("Filtered {} No Data", axis_names[axis_index]),
                (20_i32, 20_i32),
                ("sans-serif", 14).into_font().color(&RED),
            ))?;
        }
    }

    if any_spectrogram_plotted {
        root_area.present()?;
        println!("  Throttle spectrograms saved as '{}'.", output_filename);
    } else {
        // Still present to show error messages or "No Data" placeholders
        root_area.present()?; 
        println!("  Skipping '{}' throttle spectrogram saving (or only placeholders shown): No actual spectrogram data plotted.", output_filename);
    }

    Ok(())
}

// src/plotting_utils.rs
