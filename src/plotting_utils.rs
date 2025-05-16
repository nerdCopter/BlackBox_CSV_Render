// src/plotting_utils.rs

// Plotters imports - explicitly list what's used
use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::style::{RGBColor, IntoFont, Color, ShapeStyle, TextStyle};
use plotters::element::{Text, Rectangle, PathElement};
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::coord::{Shift};
use plotters::series::LineSeries;
use plotters::style::colors::{WHITE, BLACK, RED};

use std::error::Error;

use ndarray::{Array1, Array2, s}; // Removed Axis
use ndarray_stats::QuantileExt;

// Explicitly import constants used within this file
// NOTE: We will bind these to local variables outside the closures where needed
use crate::constants::{
    PLOT_WIDTH, PLOT_HEIGHT, STEP_RESPONSE_PLOT_DURATION_S, SETPOINT_THRESHOLD,
    POST_AVERAGING_SMOOTHING_WINDOW, STEADY_STATE_START_S, STEADY_STATE_END_S,
    // Import specific color constants needed
    COLOR_PIDSUM_MAIN, COLOR_PIDERROR_MAIN, COLOR_SETPOINT_MAIN,
    COLOR_SETPOINT_VS_GYRO_SP, COLOR_SETPOINT_VS_GYRO_GYRO,
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT, LINE_WIDTH_LEGEND, // Import line widths
    // Spectrogram constants
    SPECTROGRAM_THROTTLE_BINS, SPECTROGRAM_FFT_WINDOW_SIZE, SPECTROGRAM_MAX_FREQ_HZ,
    SPECTROGRAM_POWER_CLIP_MAX, SPECTROGRAM_COLOR_SCALE, SPECTROGRAM_TEXT_COLOR,
    SPECTROGRAM_GRID_COLOR,
};
use crate::log_data::LogRowData;
use crate::step_response;
use crate::fft_utils;


/// Calculate plot range with padding.
/// Adds 15% padding, or a fixed padding for very small ranges.
pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

/// Draw a "Data Unavailable" message on a plot area.
pub fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let (x_range, y_range) = area.get_pixel_range(); // Use get_pixel_range()
    let (width, height) = ((x_range.end - x_range.start) as u32, (y_range.end - y_range.start) as u32); // Calculate size from range
    let text_style = ("sans-serif", 20).into_font().color(&RED); // Use imported RED
    // Draw text without anchoring for now to avoid potential issues
    area.draw(&Text::new(
        format!("Axis {} {} Data Unavailable:\n{}", axis_index, plot_type, reason), // Format inside Text::new
        (width as i32 / 2 - 100, height as i32 / 2 - 20), // Adjust position slightly
        text_style, // Pass style directly
    ))?;
    Ok(())
}

// Define a struct to hold data for a single line series
// Store RGBColor directly
#[derive(Clone)] // Add Clone derive for easier handling in plot_step_response
pub struct PlotSeries {
    pub data: Vec<(f64, f64)>,
    pub label: String,
    pub color: RGBColor, // Store RGBColor directly
    pub stroke_width: u32, // Add stroke width
}

/// Draws a single chart for one axis within a stacked plot.
// Make concrete to DrawingArea<BitMapBackend, Shift>
fn draw_single_axis_chart( // No longer generic over DB or lifetime 'a
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>, // Concrete type
    chart_title: &str, // Title now includes axis number if desired
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    x_label: &str,
    y_label: &str,
    series: &[PlotSeries], // Use concrete PlotSeries
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
    for s in series {
        if !s.data.is_empty() {
            chart.draw_series(LineSeries::new(
                s.data.iter().cloned(),
                s.color.stroke_width(s.stroke_width), // Use stroke_width from PlotSeries
            ))?
            .label(&s.label)
            // Legend requires a closure capturing the color by move or reference
            // Use LINE_WIDTH_LEGEND for the legend line
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s.color.stroke_width(LINE_WIDTH_LEGEND)));
            series_drawn_count += 1;
        }
    }

    if series_drawn_count > 0 {
        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
            .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
    }

    Ok(())
}

/// Creates a stacked plot image with three subplots for Roll, Pitch, and Yaw.
/// The `get_axis_plot_data` closure provides the data and configuration for each axis.
///
/// The closure `get_axis_plot_data` is called for each axis (0, 1, 2) and should return:
/// - `Some((chart_title, x_range, y_range, series_data, x_label, y_label))`: If data is available for this axis.
/// - `None`: If data is not available for this axis.
///
/// The `plot_type_name` is used for the "Data Unavailable" message.
// Concrete to BitMapBackend, add 'a lifetime to BitMapBackend in where clause
fn draw_stacked_plot<'a, F>(
    output_filename: &'a str, // output_filename must have a lifetime for BitMapBackend
    root_name: &str,
    plot_type_name: &str, // Name used in unavailability message
    mut get_axis_plot_data: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)> + Send + Sync + 'static,
    // BitMapBackend error type needs to be 'static, add 'a lifetime
    <BitMapBackend<'a> as DrawingBackend>::ErrorType: 'static,
{
    let root_area = BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area(); // Use PLOT_WIDTH/HEIGHT from constants
    root_area.fill(&WHITE)?;

    // Add main title on the full drawing area (Only root_name)
    root_area.draw(&Text::new(
        root_name, // Use root_name directly
        (10, 10), // Position near top-left
        ("sans-serif", 24).into_font().color(&BLACK), // Use imported BLACK
    ))?;

    // Create a margined area below the title for the subplots
    let margined_root_area = root_area.margin(50, 5, 5, 5); // Top margin 50px

    // Split the margined area into subplots
    let sub_plot_areas = margined_root_area.split_evenly((3, 1));

    let mut any_axis_plotted = false;

    for axis_index in 0..3 {
        let area = &sub_plot_areas[axis_index];

        // Call the closure mutably
        match get_axis_plot_data(axis_index) {
            Some((chart_title, x_range, y_range, series_data, x_label, y_label)) => {
                 // Check if any series has data points AND ranges are valid before attempting to plot
                 let has_data = series_data.iter().any(|s| !s.data.is_empty());
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
                 let reason = "Calculation/Data Extraction Failed"; // More specific reason
                 draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
            }
        }
    }

    if any_axis_plotted {
        root_area.present()?;
        println!("  Stacked plot saved as '{}'.", output_filename);
    } else {
        // If no axes had data, we still present the plot with "Unavailable" messages
        // This ensures the placeholder file is created
        root_area.present()?;
        println!("  Skipping '{}' plot saving: No data available for any axis to plot, only placeholder messages shown.", output_filename);
    }

    Ok(())
}


/// Generates the Stacked PIDsum vs PID Error vs Setpoint Plot (Green, Blue, Yellow)
pub fn plot_pidsum_error_setpoint(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_pidsum_error = format!("{}_PIDsum_PIDerror_Setpoint_stacked.png", root_name);
    let plot_type_name = "PIDsum/PIDerror/Setpoint";

    // Prepare data for all axes in one pass
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                 let pidsum = row.p_term[axis_index].and_then(|p| {
                    row.i_term[axis_index].and_then(|i| {
                        row.d_term[axis_index].map(|d| p + i + d)
                    })
                });
                axis_plot_data[axis_index].push((time, row.setpoint[axis_index], row.gyro[axis_index], pidsum));
            }
        }
    }

    // Bind constants to local variables outside the closure
    let color_pidsum: RGBColor = *COLOR_PIDSUM_MAIN; // Use the corrected Green color
    let color_pid_error: RGBColor = *COLOR_PIDERROR_MAIN; // Dereference the static reference
    let color_setpoint: RGBColor = *COLOR_SETPOINT_MAIN; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    draw_stacked_plot(
        &output_file_pidsum_error,
        root_name,
        plot_type_name,
        move |axis_index| { // Use move to capture axis_plot_data and local constants
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
                if let Some(s) = setpoint {
                    setpoint_series_data.push((*time, *s));
                    val_min = val_min.min(*s);
                    val_max = val_max.max(*s);
                    if let Some(g) = gyro_filt {
                        let error = s - g;
                        pid_error_series_data.push((*time, error));
                        val_min = val_min.min(error);
                        val_max = val_max.max(error);
                    }
                }
            }

            if pidsum_series_data.is_empty() && setpoint_series_data.is_empty() && pid_error_series_data.is_empty() {
                 return None; // No actual data collected for this axis
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            // Order the series to match the legend in the desired screenshot
            if !pidsum_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pidsum_series_data,
                    label: "PIDsum (P+I+D)".to_string(),
                    color: color_pidsum, // Use captured constant (RGBColor)
                    stroke_width: line_stroke_plot, // Use captured constant
                });
            }
            if !pid_error_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pid_error_series_data,
                    label: "PID Error (Setpoint - GyroADC)".to_string(),
                    color: color_pid_error, // Use captured constant (RGBColor)
                    stroke_width: line_stroke_plot, // Use captured constant
                });
            }
            if !setpoint_series_data.is_empty() {
                series.push(PlotSeries {
                    data: setpoint_series_data,
                    label: "Setpoint".to_string(),
                    color: color_setpoint, // Use captured constant (RGBColor)
                    stroke_width: line_stroke_plot, // Use captured constant
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

/// Generates the Stacked Setpoint vs Gyro Plot (Orange, Blue)
pub fn plot_setpoint_vs_gyro(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_setpoint_gyro = format!("{}_SetpointVsGyro_stacked.png", root_name);
    let plot_type_name = "Setpoint/Gyro";

     // Prepare data for all axes in one pass
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
     for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.setpoint[axis_index], row.gyro[axis_index]));
             }
         }
     }

    // Bind constants to local variables outside the closure
    let color_sp: RGBColor = *COLOR_SETPOINT_VS_GYRO_SP; // Dereference the static reference
    let color_gyro: RGBColor = *COLOR_SETPOINT_VS_GYRO_GYRO; // Use the corrected Teal color
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    draw_stacked_plot(
        &output_file_setpoint_gyro,
        root_name,
        plot_type_name,
        move |axis_index| { // Use move to capture axis_plot_data and local constants
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

                 if let Some(s) = setpoint {
                     setpoint_series_data.push((*time, *s));
                     val_min = val_min.min(*s);
                     val_max = val_max.max(*s);
                 }
                 if let Some(g) = gyro_filt {
                     gyro_series_data.push((*time, *g));
                     val_min = val_min.min(*g);
                     val_max = val_max.max(*g);
                 }
             }

            if setpoint_series_data.is_empty() && gyro_series_data.is_empty() {
                 return None; // No actual data collected for this axis
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            // Order the series to match the legend in the desired screenshot (Gyro then Setpoint)
            if !gyro_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: gyro_series_data,
                     label: "Gyro (gyroADC)".to_string(),
                     color: color_gyro, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
                 });
            }
            if !setpoint_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: setpoint_series_data,
                     label: "Setpoint".to_string(),
                     color: color_sp, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
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

/// Generates the Stacked Gyro vs Unfiltered Gyro Plot (Purple, Orange)
pub fn plot_gyro_vs_unfilt(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
    let plot_type_name = "Gyro/UnfiltGyro";

    // Prepare data for all axes in one pass
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.gyro[axis_index], row.gyro_unfilt[axis_index]));
             }
         }
     }

    // Bind constants to local variables outside the closure
    let color_gyro_unfilt: RGBColor = *COLOR_GYRO_VS_UNFILT_UNFILT; // Dereference the static reference
    let color_gyro_filt: RGBColor = *COLOR_GYRO_VS_UNFILT_FILT; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    draw_stacked_plot(
        &output_file_gyro,
        root_name,
        plot_type_name,
        move |axis_index| { // Use move to capture axis_plot_data and local constants
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
                 return None; // No actual data collected for this axis
            }


            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !unfilt_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: unfilt_series_data,
                     label: "Unfiltered Gyro (gyroUnfilt/debug)".to_string(),
                     color: color_gyro_unfilt, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
                 });
            }
            if !filt_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: filt_series_data,
                     label: "Filtered Gyro (gyroADC)".to_string(),
                     color: color_gyro_filt, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
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

/// Generates the Stacked Step Response Plot (Blue, Orange, Red)
pub fn plot_step_response(
    step_response_results: &[Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3],
    root_name: &str,
    sample_rate: Option<f64>, // Pass sample rate for steady-state calcs
) -> Result<(), Box<dyn Error>> {
    // Bind constants to local variables outside the closure
    let step_response_plot_duration_s = STEP_RESPONSE_PLOT_DURATION_S;
    let steady_state_start_s = STEADY_STATE_START_S;
    let steady_state_end_s = STEADY_STATE_END_S;
    let setpoint_threshold = SETPOINT_THRESHOLD;
    let post_averaging_smoothing_window = POST_AVERAGING_SMOOTHING_WINDOW;
    let color_high_sp: RGBColor = *COLOR_STEP_RESPONSE_HIGH_SP; // Dereference the static reference
    let color_combined: RGBColor = *COLOR_STEP_RESPONSE_COMBINED; // Dereference the static reference
    let color_low_sp: RGBColor = *COLOR_STEP_RESPONSE_LOW_SP; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    let output_file_step = format!("{}_step_response_stacked_plot_{}s.png", root_name, step_response_plot_duration_s); // Use captured variable
    let plot_type_name = "Step Response";

    // Get sample rate for steady-state window calculation (fallback if needed)
    let sr = sample_rate.unwrap_or(1000.0); // Use a reasonable default if sample rate unknown

    // Pre-process results to create owned data for the closure
    let mut plot_data_per_axis: [Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)>; 3] = Default::default();

    for axis_index in 0..3 {
        if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) = &step_response_results[axis_index] {
            let response_length_samples = response_time.len();

            if response_length_samples == 0 || valid_stacked_responses.shape()[0] == 0 {
                 continue; // Skip this axis if no data
            }

            let num_qc_windows = valid_stacked_responses.shape()[0];
            let ss_start_sample = (steady_state_start_s * sr).floor() as usize;
            let ss_end_sample = (steady_state_end_s * sr).ceil() as usize;
            let current_ss_start_sample = ss_start_sample.min(response_length_samples);
            let current_ss_end_sample = ss_end_sample.min(response_length_samples);

            if current_ss_start_sample >= current_ss_end_sample {
                eprintln!("Warning: Axis {} Step Response: Steady-state window is invalid (start >= end). Skipping final normalization and plot for this axis.", axis_index);
                 continue; // Skip this axis
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
                step_response::average_responses(stacked_resp, mask, resp_len_samples)
                    .ok()
                    .and_then(|avg_resp| {
                         if avg_resp.is_empty() { return None; }
                         let smoothed_resp = step_response::moving_average_smooth_f64(&avg_resp, smoothing_window);
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

            // Clone the Options so we can check is_some() and then move the contained Array1
            let final_low_response_cloned = final_low_response.clone();
            let final_high_response_cloned = final_high_response.clone();
            let final_combined_response_cloned = final_combined_response.clone();

            let is_low_response_valid = final_low_response_cloned.is_some();
            let is_high_response_valid = final_high_response_cloned.is_some();
            let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some(); // Only plot combined if high SP was valid


            if !(is_low_response_valid || is_high_response_valid) { // Check if at least Low or High is valid
                continue; // Skip this axis if no valid responses (Low or High)
            }

            let mut resp_min = f64::INFINITY;
            let mut resp_max = f64::NEG_INFINITY;
            if let Some(resp) = &final_low_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            if let Some(resp) = &final_high_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            // Only include combined in range calculation if it will be plotted
            if is_combined_response_valid {
                 if let Some(resp) = &final_combined_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            }


            let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
            let x_range = 0f64..step_response_plot_duration_s * 1.05;
            let y_range = final_resp_min..final_resp_max;

            let mut series = Vec::new();
            // Order the series for drawing (z-index): Low, High, Combined
            if let Some(resp) = final_low_response { // Use original Option to move the data
                 series.push(PlotSeries {
                     data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                     label: format!("< {} deg/s", setpoint_threshold),
                     color: color_low_sp,
                     stroke_width: line_stroke_plot, // Use plot width
                 });
            }
            if let Some(resp) = final_high_response { // Use original Option to move the data
                 series.push(PlotSeries {
                     data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                     label: format!("\u{2265} {} deg/s", setpoint_threshold),
                     color: color_high_sp,
                     stroke_width: line_stroke_plot, // Use plot width
                 });
             }
            // Plot Combined LAST for z-index, but ONLY if high setpoint response was valid and combined calc worked
            if is_combined_response_valid { // Use the boolean flag derived from the cloned Option
                 if let Some(resp) = final_combined_response { // Use original Option to move the data
                     series.push(PlotSeries {
                         data: response_time.iter().zip(resp.iter()).map(|(&t, &v)| (t, v)).collect(),
                         label: "Combined".to_string(),
                         color: color_combined,
                         stroke_width: line_stroke_plot, // Use plot width
                     });
                 }
            }

             plot_data_per_axis[axis_index] = Some((
                format!("Axis {} Step Response", axis_index),
                x_range,
                y_range,
                series, // Move the owned Vec<PlotSeries>
                "Time (s)".to_string(),
                "Normalized Response".to_string(),
             ));
        }
    }

    // Now, the closure only needs to capture plot_data_per_axis by move
    draw_stacked_plot(
        &output_file_step,
        root_name,
        plot_type_name,
        move |axis_index| {
            // Take ownership of the Option for this axis
            plot_data_per_axis[axis_index].take()
        },
    )
}


/// Interpolates color based on a value and a predefined scale.
fn get_spectrogram_color(value: f32, min_clip: f32, max_clip: f32) -> RGBColor {
    let normalized_value = ((value - min_clip) / (max_clip - min_clip).max(1e-6)).clamp(0.0, 1.0);

    if normalized_value <= SPECTROGRAM_COLOR_SCALE[0].0 { // Should be 0.0
        return SPECTROGRAM_COLOR_SCALE[0].1;
    }

    for i in 0..(SPECTROGRAM_COLOR_SCALE.len() - 1) {
        let (p1, c1) = SPECTROGRAM_COLOR_SCALE[i];
        let (p2, _c2) = SPECTROGRAM_COLOR_SCALE[i + 1]; // Prefixed c2
        if normalized_value <= p2 {
            let t = (normalized_value - p1) / (p2 - p1).max(1e-6);
            // mix returns RGBAColor, convert back to RGBColor
            let (r,g,b) = c1.mix(t.into()).rgb();
            return RGBColor(r,g,b);
        }
    }
    SPECTROGRAM_COLOR_SCALE.last().unwrap().1 // Return the last color if above all points
}


/// Draws a single throttle vs frequency spectrogram.
fn draw_single_throttle_spectrogram<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title_prefix: &str,
    axis_name: &str,
    psd_matrix: &Array2<f32>,      // Freq x ThrottleBin
    freq_bins: &Array1<f32>,       // Frequencies for rows
    throttle_bins: &Array1<f32>, // Throttle bin centers for columns
) -> Result<(), Box<dyn Error>>
where DB::ErrorType: 'static
{
    area.fill(&BLACK)?; // Black background for the spectrogram chart area

    let x_range = 0f32..100f32; // Throttle %
    let y_range = 0f32..SPECTROGRAM_MAX_FREQ_HZ;

    let mut chart = ChartBuilder::on(area)
        .caption(format!("{} {}", title_prefix, axis_name), TextStyle::from(("sans-serif", 18).into_font().color(SPECTROGRAM_TEXT_COLOR)))
        .margin_top(25) // Make space for title
        .margin_right(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart.configure_mesh()
        .axis_style(ShapeStyle::from(SPECTROGRAM_TEXT_COLOR).stroke_width(1))
        .x_desc("% Throttle")
        .y_desc("Frequency (Hz)")
        .label_style(("sans-serif", 12).into_font().color(SPECTROGRAM_TEXT_COLOR))
        .light_line_style(&SPECTROGRAM_GRID_COLOR) // Grid lines
        .bold_line_style(&SPECTROGRAM_GRID_COLOR)
        .x_labels(10)
        .y_labels(5)
        .draw()?;

    let (num_freq_bins_total, num_throttle_plot_bins) = psd_matrix.dim();
    if num_freq_bins_total == 0 || num_throttle_plot_bins == 0 || freq_bins.len() != num_freq_bins_total || throttle_bins.len() != num_throttle_plot_bins {
        let text_style = ("sans-serif", 16).into_font().color(&RED);
        // Position text using data coordinates (f32, f32)
        let text_x = x_range.start + (x_range.end - x_range.start) * 0.1;
        let text_y = y_range.start + (y_range.end - y_range.start) * 0.5;
        chart.plotting_area().draw(&Text::new(
            "Spectrogram data empty or mismatched",
            (text_x, text_y),
            text_style,
        ))?;
        return Ok(());
    }

    let throttle_cell_width = if num_throttle_plot_bins > 1 {
        100.0 / num_throttle_plot_bins as f32 // Use average width based on total range and #bins
    } else {
        100.0 // Fallback if only one bin, covers full range
    };


    let mut max_power_overall = 0.0f32;
    let mut total_power_sum = 0.0f32;
    let mut power_count = 0;

    for j in 0..num_throttle_plot_bins { // Iterate throttle bins (columns)
        let throttle_center = throttle_bins[j];
        let x0 = throttle_center - throttle_cell_width / 2.0;
        let x1 = throttle_center + throttle_cell_width / 2.0;

        for i in 0..num_freq_bins_total { // Iterate frequency bins (rows)
            let freq_center = freq_bins[i];
            if freq_center > SPECTROGRAM_MAX_FREQ_HZ + 1e-3 { // Add epsilon for float comparison
                continue; // Don't plot frequencies above max
            }

            // Determine frequency cell height
            let freq_cell_height = if i + 1 < num_freq_bins_total {
                freq_bins[i+1] - freq_bins[i]
            } else if i > 0 {
                freq_bins[i] - freq_bins[i-1] // For the last bin
            } else {
                 SPECTROGRAM_MAX_FREQ_HZ / num_freq_bins_total as f32 // Fallback if only one freq bin (use y_range.end)
            };
            let y0 = freq_center - freq_cell_height / 2.0;
            let y1 = freq_center + freq_cell_height / 2.0;


            let power = psd_matrix[[i, j]];
            max_power_overall = max_power_overall.max(power);
            if power > 1e-6 { // Consider only non-negligible power for mean
                 total_power_sum += power;
                 power_count += 1;
            }

            let color = get_spectrogram_color(power, 0.0, SPECTROGRAM_POWER_CLIP_MAX);
             // Draw if not the same as background (BLACK)
            if color != BLACK {
                let rect = Rectangle::new([(x0.max(0.0), y0.max(0.0)), (x1.min(100.0), y1.min(SPECTROGRAM_MAX_FREQ_HZ))], color.filled());
                chart.plotting_area().draw(&rect)?;
            }
        }
    }

    // Calculate mean of displayed power (non-zero, within freq range)
    let mean_power = if power_count > 0 { total_power_sum / power_count as f32 } else { 0.0 };

    let text_style = ("sans-serif", 12).into_font().color(SPECTROGRAM_TEXT_COLOR);
    // Ensure text coordinates are f32
    let text_pos_x = x_range.end * 0.65_f32;
    let text_pos_y = y_range.end * 0.9_f32;
    chart.plotting_area().draw(&Text::new(
        format!("mean={:.4}\npeak={:.4}", mean_power, max_power_overall),
        (text_pos_x, text_pos_y),
        text_style,
    ))?;


    Ok(())
}

/// Draws a color bar for the spectrogram.
fn draw_spectrogram_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title: &str,
) -> Result<(), Box<dyn Error>>
where DB::ErrorType: 'static
{
    area.fill(&WHITE)?; // White background for the colorbar area

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 12).into_font().color(&BLACK))
        .margin(5)
        .build_cartesian_2d(0f32..SPECTROGRAM_POWER_CLIP_MAX, 0f32..1f32)?; // X is power, Y is dummy

    chart.configure_mesh()
        .disable_y_axis()
        .x_labels(6) // e.g., 0, 0.1, 0.2, 0.3, 0.4, 0.5
        .x_label_formatter(&|x| format!("{:.1}", x))
        .label_style(("sans-serif", 10).into_font().color(&BLACK))
        .draw()?;

    let x_coord_range = chart.x_range(); // This is std::ops::Range<f32>
    let x_start = x_coord_range.start;
    let x_end = x_coord_range.end;
    let num_steps = 100; // Number of rectangles to draw for the gradient
    let step_width = (x_end - x_start) / num_steps as f32;

    for i in 0..num_steps {
        let val_start = x_start + i as f32 * step_width;
        let val_end = x_start + (i + 1) as f32 * step_width;
        let mid_val = (val_start + val_end) / 2.0;

        let color = get_spectrogram_color(mid_val, 0.0, SPECTROGRAM_POWER_CLIP_MAX);
        let rect = Rectangle::new([(val_start, 0.0), (val_end, 1.0)], color.filled());
        chart.plotting_area().draw(&rect)?;
    }
    Ok(())
}


/// Generates the Stacked Throttle vs Frequency Spectrogram plot.
pub fn plot_throttle_spectrograms(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let sr = match sample_rate {
        Some(s) => s,
        None => {
            println!("Warning: Sample rate unknown, skipping throttle spectrograms.");
            return Ok(());
        }
    };

    let output_filename = format!("{}_throttle_spectrograms.png", root_name);
    let root_area = BitMapBackend::new(&output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?; // Overall white background

    let main_title_area_height = 50;
    let colorbar_area_height = 70;
    let _plot_area_height = PLOT_HEIGHT - main_title_area_height - colorbar_area_height - 20; // 20 for bottom margin, prefixed

    let (main_title_area, remaining_area) = root_area.split_vertically(main_title_area_height);
    let (colorbar_area, plot_grid_area) = remaining_area.split_vertically(colorbar_area_height);

    // Draw main title
    main_title_area.draw(&Text::new(
        format!("{} Throttle Spectrograms", root_name),
        (10, 10), // Pixel coordinates for root area
        ("sans-serif", 24).into_font().color(&BLACK),
    ))?;

    // Draw color bar
    draw_spectrogram_colorbar(&colorbar_area.margin(0, 100, 0, 100), "Scale (Power)")?; // Margin for centering

    // Split plot area into 3 rows (Roll, Pitch, Yaw)
    let axis_plot_areas = plot_grid_area.split_evenly((3, 1)); // 3 rows, 1 column
    let axis_names = ["Roll", "Pitch", "Yaw"];

    let mut any_spectrogram_plotted = false;

    for axis_index in 0..3 {
        let row_area = &axis_plot_areas[axis_index];
        // Split each row into 2 columns (Unfiltered, Filtered)
        let (unfilt_area, filt_area) = row_area.split_horizontally(PLOT_WIDTH / 2);

        // --- Process Unfiltered Gyro ---
        let mut gyro_unfilt_data: Vec<f32> = Vec::new();
        let mut throttle_data_unfilt: Vec<f32> = Vec::new();
        for row in log_data {
            // Throttle comes from row.throttle (which is setpoint[3])
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
                SPECTROGRAM_FFT_WINDOW_SIZE,
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
                    let area_clone = unfilt_area.clone(); // Clone for message drawing if needed
                    area_clone.fill(&BLACK)?; // Ensure black background for message
                    area_clone.draw(&Text::new(
                        format!("Unfiltered {} Data Error:\n{}", axis_names[axis_index], e),
                        (20_i32, 20_i32), // Pixel coordinates for sub-area message
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
            let area_clone = unfilt_area.clone();
            area_clone.fill(&BLACK)?;
            area_clone.draw(&Text::new(
                format!("Unfiltered {} No Data", axis_names[axis_index]),
                (20_i32, 20_i32), // Pixel coordinates for sub-area message
                ("sans-serif", 14).into_font().color(&RED),
            ))?;
        }

        // --- Process Filtered Gyro ---
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
                SPECTROGRAM_FFT_WINDOW_SIZE,
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
                     area_clone.fill(&BLACK)?;
                     area_clone.draw(&Text::new(
                        format!("Filtered {} Data Error:\n{}", axis_names[axis_index], e),
                        (20_i32, 20_i32), // Pixel coordinates for sub-area message
                        ("sans-serif", 14).into_font().color(&RED),
                    ))?;
                }
            }
        } else {
            let area_clone = filt_area.clone();
            area_clone.fill(&BLACK)?;
            area_clone.draw(&Text::new(
                format!("Filtered {} No Data", axis_names[axis_index]),
                (20_i32, 20_i32), // Pixel coordinates for sub-area message
                ("sans-serif", 14).into_font().color(&RED),
            ))?;
        }
    }


    if any_spectrogram_plotted {
        root_area.present()?;
        println!("  Throttle spectrograms saved as '{}'.", output_filename);
    } else {
        root_area.present()?; // Still save the file with titles/messages
        println!("  Skipping '{}' throttle spectrogram saving: No data available for any axis to plot.", output_filename);
    }

    Ok(())
}

// src/plotting_utils.rs