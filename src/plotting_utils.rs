// src/plotting_utils.rs

use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::style::{RGBColor, IntoFont, Color, ShapeStyle, TextStyle};
use plotters::element::{Text, Rectangle, PathElement};
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::coord::{Shift};
use plotters::series::LineSeries;
use plotters::style::colors::{WHITE, BLACK, RED};

use std::error::Error;
use std::fs::File;
use std::io::Write;


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
    SPECTROGRAM_THROTTLE_BINS,
    SPECTROGRAM_FFT_TIME_WINDOW_MS,
    SPECTROGRAM_MAX_FREQ_HZ,
    BBE_SCALE_HEATMAP, 
    MIN_POWER_FOR_LOG_SCALE, 
    SPECTROGRAM_TEXT_COLOR, SPECTROGRAM_GRID_COLOR,
};
use crate::log_data::LogRowData;
use crate::step_response;
use crate::fft_utils;


// Helper function to convert HSL to RGB
fn hsl_to_rgb(h_degrees: f32, s: f32, l: f32) -> RGBColor {
    if s == 0.0 { 
        let val = (l * 255.0).round().clamp(0.0, 255.0) as u8;
        return RGBColor(val, val, val);
    }

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = (h_degrees % 360.0) / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h_prime >= 0.0 && h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime >= 1.0 && h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime >= 2.0 && h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime >= 3.0 && h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime >= 4.0 && h_prime < 5.0 {
        (x, 0.0, c)
    } else { 
        (c, 0.0, x)
    };

    RGBColor(
        ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}


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
    let (width, height) = area.dim_in_pixel();
    let text_style = ("sans-serif", 20).into_font().color(&RED);
    area.draw(&Text::new(
        format!("Axis {} {} Data Unavailable:\n{}", axis_index, plot_type, reason),
        (width as i32 / 2 - 150, height as i32 / 2 - 20),
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
    for s_item in series {
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
            .background_style(&WHITE.mix(0.8)).border_style(&plotters::style::colors::BLACK).label_font(("sans-serif", 12)).draw()?;
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
                        row.d_term[axis_index].map(|d_val| p + i + d_val)
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
                if let Some(s_val) = setpoint {
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

                 if let Some(s_val) = setpoint {
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
    let steady_state_start_s_const = STEADY_STATE_START_S;
    let steady_state_end_s_const = STEADY_STATE_END_S;
    let setpoint_threshold_const = SETPOINT_THRESHOLD;
    let post_averaging_smoothing_window_const = POST_AVERAGING_SMOOTHING_WINDOW;
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
                         if shifted_response.is_empty() { return None; }
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

fn get_spectrogram_color(averaged_normalized_amplitude: f32) -> RGBColor {
    // Linearly scale the averaged (N-normalized, 2x scaled) amplitude against BBE_SCALE_HEATMAP
    // An averaged_normalized_amplitude equal to BBE_SCALE_HEATMAP (or higher) results in 1.0 lightness (white).
    let mut lightness = (averaged_normalized_amplitude / BBE_SCALE_HEATMAP).clamp(0.0, 1.0);

    // Apply a gamma correction to enhance visibility of lower-mid range values
    // A gamma of 0.5 (sqrt) will brighten mid-tones. Adjust as needed.
    // BBE's visual output suggests something like this might be happening.
    const GAMMA: f32 = 0.6; // Experiment with this value (e.g., 0.4, 0.5, 0.6)
    lightness = lightness.powf(GAMMA);
    
    hsl_to_rgb(0.0, 1.0, lightness.clamp(0.0, 1.0)) // Hue 0 (Red), Sat 1.0
}


fn draw_single_throttle_spectrogram<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title_prefix: &str,
    axis_name: &str,
    // This matrix contains N-normalized, 2x-scaled averaged amplitudes
    averaged_amplitude_spectrum_matrix: &Array2<f32>, 
    freq_bins: &Array1<f32>,
    throttle_bins: &Array1<f32>,
    // This is the peak of RAW (unnormalized) magnitudes from individual segments
    peak_raw_segment_magnitude_for_text: f32, 
    mut diag_file: Option<&mut File>,
) -> Result<(), Box<dyn Error>>
where DB::ErrorType: 'static
{
    area.fill(&BLACK)?;

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
        .x_labels(10)
        .y_labels(5)
        .draw()?;

    let (num_freq_bins_total, num_throttle_plot_bins) = averaged_amplitude_spectrum_matrix.dim();
    if num_freq_bins_total == 0 || num_throttle_plot_bins == 0 || freq_bins.len() != num_freq_bins_total || throttle_bins.len() != num_throttle_plot_bins {
        let text_style = ("sans-serif", 16).into_font().color(&RED);
        let text_x = x_range_spec.start + (x_range_spec.end - x_range_spec.start) * 0.1;
        let text_y = y_range_spec.start + (y_range_spec.end - y_range_spec.start) * 0.5;
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
        100.0
    };

    let mut total_avg_amplitude_sum = 0.0f32;
    let mut avg_amplitude_count = 0;

    for j_throttle_idx in 0..num_throttle_plot_bins {
        let throttle_center = throttle_bins[j_throttle_idx];
        let y0_throttle = throttle_center - throttle_cell_height / 2.0;
        let y1_throttle = throttle_center + throttle_cell_height / 2.0;

        for i_freq_idx in 0..num_freq_bins_total {
            let freq_center = freq_bins[i_freq_idx];

            if freq_center > SPECTROGRAM_MAX_FREQ_HZ + 1e-3 {
                continue;
            }

            let freq_cell_width = if num_freq_bins_total > 0 {
                if i_freq_idx + 1 < num_freq_bins_total {
                    freq_bins[i_freq_idx + 1] - freq_bins[i_freq_idx]
                } else if i_freq_idx > 0 {
                    freq_bins[i_freq_idx] - freq_bins[i_freq_idx - 1]
                } else {
                    SPECTROGRAM_MAX_FREQ_HZ / num_freq_bins_total.max(1) as f32
                }
            } else {
                SPECTROGRAM_MAX_FREQ_HZ
            };

            let x0_freq = freq_center - freq_cell_width / 2.0;
            let x1_freq = freq_center + freq_cell_width / 2.0;

            let avg_amplitude_val = averaged_amplitude_spectrum_matrix[[i_freq_idx, j_throttle_idx]];
            if avg_amplitude_val > MIN_POWER_FOR_LOG_SCALE { 
                 total_avg_amplitude_sum += avg_amplitude_val;
                 avg_amplitude_count += 1;
            }
            
            let color = get_spectrogram_color(avg_amplitude_val);

            if let Some(file) = diag_file.as_mut() {
                 if avg_amplitude_val > 0.001 { 
                    let lightness_before_gamma = (avg_amplitude_val / BBE_SCALE_HEATMAP).clamp(0.0, 1.0);
                    let lightness_after_gamma = lightness_before_gamma.powf(0.6); // Assuming GAMMA = 0.6
                    if j_throttle_idx % (num_throttle_plot_bins / 5_usize.max(1) + 1) == 0 &&
                       i_freq_idx % (num_freq_bins_total / 10_usize.max(1) + 1) == 0 {
                        writeln!(file, "Diag (draw_single_throttle_spectrogram): FreqBin {}, ThrBin {} -- AvgNormAmp: {:.4} (BBE_SCALE_HEATMAP: {:.2}), Lightness (before/after gamma 0.6): {:.3}/{:.3}, Color: {:?}",
                                 i_freq_idx, j_throttle_idx, avg_amplitude_val, BBE_SCALE_HEATMAP,
                                 lightness_before_gamma, lightness_after_gamma, color)?;
                    }
                }
            }

            let rect_coords = [
                (x0_freq.max(0.0), y0_throttle.max(0.0)),
                (x1_freq.min(SPECTROGRAM_MAX_FREQ_HZ), y1_throttle.min(100.0))
            ];
            let rect = Rectangle::new(rect_coords, color.filled());
            chart.plotting_area().draw(&rect)?;
        }
    }

    let mean_of_avg_amplitudes = if avg_amplitude_count > 0 { total_avg_amplitude_sum / avg_amplitude_count as f32 } else { 0.0 };

    let text_style_spec = ("sans-serif", 12).into_font().color(SPECTROGRAM_TEXT_COLOR);
    let text_pos_x = x_range_spec.end * 0.65_f32;
    let text_pos_y = y_range_spec.end * 0.9_f32;
    chart.plotting_area().draw(&Text::new(
        format!("mean_mag={:.1} peak_mag_seg={:.1}", mean_of_avg_amplitudes, peak_raw_segment_magnitude_for_text),
        (text_pos_x, text_pos_y),
        text_style_spec,
    ))?;
    Ok(())
}

pub fn plot_throttle_spectrograms(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    mut diag_file: Option<&mut File>,
) -> Result<(), Box<dyn Error>> {
    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "--- Spectrogram Colormap Info (plotting_utils.rs @ plot_throttle_spectrograms entry) ---")?;
        writeln!(file, "Using HSL(0, 100%, L) based calculation (Black-Red-White).")?;
        writeln!(file, "Lightness 'L' is ((AveragedNormalizedAmplitude / BBE_SCALE_HEATMAP).powf(GAMMA)).clamp(0.0, 1.0). GAMMA ~0.6")?;
        writeln!(file, "BBE_SCALE_HEATMAP = {}.", BBE_SCALE_HEATMAP)?;
        writeln!(file, "'peak_mag_seg' text shows peak of raw magnitudes from individual FFT segments.")?;
        writeln!(file, "Averaged amplitudes are N-normalized and 2x scaled (for non-DC/Nyquist).")?;
        writeln!(file, "------------------------------------------------------------------------------------")?;
    } else {
        println!("--- Spectrogram Colormap Info (plotting_utils.rs @ plot_throttle_spectrograms entry) ---");
        println!("Using HSL(0, 100%, L) based calculation (Black-Red-White).");
        println!("Lightness 'L' is ((AveragedNormalizedAmplitude / BBE_SCALE_HEATMAP).powf(GAMMA)).clamp(0.0, 1.0). GAMMA ~0.6");
        println!("BBE_SCALE_HEATMAP = {}.", BBE_SCALE_HEATMAP);
        println!("'peak_mag_seg' text shows peak of raw magnitudes from individual FFT segments.");
        println!("Averaged amplitudes are N-normalized and 2x scaled (for non-DC/Nyquist).");
        println!("------------------------------------------------------------------------------------");
    }


    let sr = match sample_rate {
        Some(s_rate) => s_rate,
        None => {
            let msg = "Warning: Sample rate unknown, skipping throttle spectrograms.";
            println!("{}", msg);
            if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?;}
            return Ok(());
        }
    };

    let output_filename = format!("{}_throttle_spectrograms.png", root_name);
    let root_area = BitMapBackend::new(&output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let main_title_area_height = 50;
    let top_padding = 15;
    let (main_title_area, temp_area1) = root_area.split_vertically(main_title_area_height);
    let (_padding_top_area, plot_grid_area) = temp_area1.split_vertically(top_padding);

    main_title_area.draw(&Text::new(
        format!("{} Throttle Spectrograms", root_name),
        (10, 10),
        ("sans-serif", 24).into_font().color(&BLACK),
    ))?;

    let axis_plot_areas = plot_grid_area.split_evenly((3, 1));
    let axis_names = ["Roll", "Pitch", "Yaw"];
    let mut any_spectrogram_plotted = false;

    for axis_index in 0..3 {
        let row_area = &axis_plot_areas[axis_index];
        let (unfilt_area, filt_area) = row_area.split_horizontally(PLOT_WIDTH / 2);

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
                SPECTROGRAM_FFT_TIME_WINDOW_MS,
                diag_file.as_mut().map(|df| &mut **df),
            ) {
                Ok((avg_norm_amp_matrix, freq_bins, throttle_bins, peak_raw_segment_mag_for_text)) => {
                    if let Some(file) = diag_file.as_mut() {
                        writeln!(file, "Plotting Unfiltered Gyro Axis {} with PeakRawSegmentMag (for text): {:.2}", 
                            axis_names[axis_index], peak_raw_segment_mag_for_text)?;
                    }
                    draw_single_throttle_spectrogram(
                        &unfilt_area,
                        "Unfiltered Gyro",
                        axis_names[axis_index],
                        &avg_norm_amp_matrix,
                        &freq_bins,
                        &throttle_bins,
                        peak_raw_segment_mag_for_text,
                        diag_file.as_mut().map(|df| &mut **df),
                    )?;
                    any_spectrogram_plotted = true;
                }
                Err(_e) => {
                    let msg = format!("Error calculating unfiltered spectrogram for Axis {}: {:?}", axis_names[axis_index], _e);
                    eprintln!("{}", msg);
                    if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg).ok(); }
                    let area_clone = unfilt_area.clone();
                    area_clone.fill(&BLACK)?;
                    area_clone.draw(&Text::new(
                        format!("Unfiltered {} Data Error:\n{:?}", axis_names[axis_index], _e),
                        (20_i32, 20_i32),
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
            let msg = format!("Unfiltered {} No Data", axis_names[axis_index]);
            if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?; }
            let area_clone = unfilt_area.clone();
            area_clone.fill(&BLACK)?;
            area_clone.draw(&Text::new( msg, (20_i32, 20_i32), ("sans-serif", 14).into_font().color(&RED), ))?;
        }

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
                SPECTROGRAM_FFT_TIME_WINDOW_MS,
                diag_file.as_mut().map(|df| &mut **df),
            ) {
                 Ok((avg_norm_amp_matrix, freq_bins, throttle_bins, peak_raw_segment_mag_for_text)) => {
                     if let Some(file) = diag_file.as_mut() {
                        writeln!(file, "Plotting Filtered Gyro Axis {} with PeakRawSegmentMag (for text): {:.2}", 
                            axis_names[axis_index], peak_raw_segment_mag_for_text)?;
                    }
                    draw_single_throttle_spectrogram(
                        &filt_area,
                        "Filtered Gyro",
                        axis_names[axis_index],
                        &avg_norm_amp_matrix,
                        &freq_bins,
                        &throttle_bins,
                        peak_raw_segment_mag_for_text,
                        diag_file.as_mut().map(|df| &mut **df),
                    )?;
                    any_spectrogram_plotted = true;
                }
                Err(_e) => {
                    let msg = format!("Error calculating filtered spectrogram for Axis {}: {:?}", axis_names[axis_index], _e);
                    eprintln!("{}", msg);
                    if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg).ok(); }
                     let area_clone = filt_area.clone();
                     area_clone.fill(&BLACK)?;
                     area_clone.draw(&Text::new(
                        format!("Filtered {} Data Error:\n{:?}", axis_names[axis_index], _e),
                        (20_i32, 20_i32),
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
             let msg = format!("Filtered {} No Data", axis_names[axis_index]);
             if let Some(file) = diag_file.as_mut() { writeln!(file, "{}", msg)?; }
            let area_clone = filt_area.clone();
            area_clone.fill(&BLACK)?;
            area_clone.draw(&Text::new( msg, (20_i32, 20_i32), ("sans-serif", 14).into_font().color(&RED), ))?;
        }
    }

    if any_spectrogram_plotted {
        root_area.present()?;
        println!("  Throttle spectrograms saved as '{}'.", output_filename);
    } else {
        root_area.present()?;
        println!("  Skipping '{}' throttle spectrogram saving (or only placeholders shown): No actual spectrogram data plotted.", output_filename);
    }

    Ok(())
}

// src/plotting_utils.rs
