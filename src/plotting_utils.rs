// src/plotting_utils.rs

// Plotters imports - explicitly list what's used
use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::style::{RGBColor, IntoFont, Color};
use plotters::element::Text;
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::element::PathElement;
use plotters::series::LineSeries;
use plotters::style::colors::{WHITE, BLACK, RED};

use std::error::Error;

use ndarray::{Array1, Array2, s};
use ndarray_stats::QuantileExt;

// Explicitly import constants used within this file
// NOTE: We will bind these to local variables outside the closures where needed
use crate::constants::{
    PLOT_WIDTH, PLOT_HEIGHT, STEP_RESPONSE_PLOT_DURATION_S, SETPOINT_THRESHOLD,
    POST_AVERAGING_SMOOTHING_WINDOW, STEADY_STATE_START_S, STEADY_STATE_END_S,
    SPECTRUM_Y_AXIS_FLOOR,
    SPECTRUM_NOISE_FLOOR_HZ, SPECTRUM_Y_AXIS_HEADROOM_FACTOR,
    // Import specific color constants needed
    COLOR_PIDSUM_MAIN, COLOR_PIDERROR_MAIN, COLOR_SETPOINT_MAIN,
    COLOR_SETPOINT_VS_GYRO_SP, COLOR_SETPOINT_VS_GYRO_GYRO,
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT, LINE_WIDTH_LEGEND, // Import line widths
};
use crate::log_data::LogRowData;
use crate::step_response;


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
    // BitMapBackend error type needs to be 'static
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


/// Creates a stacked plot image with three rows and two columns for subplots.
/// The `get_axis_plot_data` closure provides the data and configuration for each axis's
/// two subplots (left column, right column).
///
/// The closure `get_axis_plot_data` is called for each axis (0, 1, 2) and should return:
/// - `Some([Option<PlotConfig>, Option<PlotConfig>])`: An array of two Options, where each Option
///   contains `(chart_title, x_range, y_range, series_data, x_label, y_label)` for its respective subplot.
///   A `None` for a specific subplot means no data is available for that subplot.
/// - `None`: If no data or configuration is available for *either* subplot for this axis.
///
/// The `plot_type_name` is used for the "Data Unavailable" message.
fn draw_dual_spectrum_plot<'a, F>(
    output_filename: &'a str,
    root_name: &str,
    plot_type_name: &str,
    mut get_axis_plot_data: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<[Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)>; 2]> + Send + Sync + 'static,
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
    let sub_plot_areas = margined_root_area.split_evenly((3, 2)); // 3 rows, 2 columns

    let mut any_plot_drawn = false;

    for axis_index in 0..3 {
        // Get the plot configurations for the current axis (left and right subplots)
        let plots_for_axis_option = get_axis_plot_data(axis_index);

        for col_idx in 0..2 { // 0 for left (unfiltered), 1 for right (filtered)
            let area = &sub_plot_areas[axis_index * 2 + col_idx]; // Calculate the correct subplot index

            if let Some(plots_for_axis) = plots_for_axis_option.as_ref() {
                if let Some(plot_config) = plots_for_axis[col_idx].as_ref() {
                    let (chart_title, x_range, y_range, series_data, x_label, y_label) = plot_config;

                    let has_data = series_data.iter().any(|s| !s.data.is_empty());
                    let valid_ranges = x_range.end > x_range.start && y_range.end > y_range.start;

                    if has_data && valid_ranges {
                        draw_single_axis_chart(
                            area,
                            chart_title,
                            x_range.clone(),
                            y_range.clone(),
                            x_label,
                            y_label,
                            series_data,
                        )?;
                        any_plot_drawn = true;
                    } else {
                        let reason = if !has_data { "No data points" } else { "Invalid ranges" };
                        draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
                    }
                } else {
                    // No plot config for this specific subplot
                    draw_unavailable_message(area, axis_index, plot_type_name, "Data Not Available")?;
                }
            } else {
                // No data/config for this entire axis (both subplots)
                draw_unavailable_message(area, axis_index, plot_type_name, "Data Not Available")?;
            }
        }
    }

    if any_plot_drawn {
        root_area.present()?;
        println!("  Stacked plot saved as '{}'.", output_filename);
    } else {
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

            // CodeRabbit
            // combined_response_valid logic is overly restrictive
            // Combined response validity currently depends on high-SP validity:
            // 
            // let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some();
            // If only the low-SP mask passes QC, the combined mask may still be meaningful but will be suppressed.
            // Recommend dropping the is_high_response_valid dependency:
            // 
            // -let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some();
            // +let is_combined_response_valid = final_combined_response_cloned.is_some();

            //let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some(); // Only plot combined if high SP was valid
            let is_combined_response_valid = final_combined_response_cloned.is_some();

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

/// Generates a stacked plot with two columns per axis, showing Unfiltered and Filtered Gyro spectrums.
pub fn plot_gyro_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{}_Gyro_Spectrums_comparative.png", root_name);
    let plot_type_name = "Gyro Spectrums";

    // Extract sample rate early; if None, return Ok(()) to skip plotting.
    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Gyro Spectrum Plot: Sample rate could not be determined.");
        return Ok(()); // Exit early as sample rate is essential for frequency calculation
    };

    // Pre-calculate all FFT data for all axes.
    // Stores: [axis_index] -> ([unfilt_series_data, filt_series_data], max_amp_unfilt, max_amp_filt).
    let mut all_fft_data: [Option<([Vec<(f64, f64)>; 2], f64, f64)>; 3] = Default::default();

    for axis_index in 0..3 {
            let mut unfilt_samples: Vec<f32> = Vec::new();
            let mut filt_samples: Vec<f32> = Vec::new();

            for row in log_data {
                if let (Some(unfilt_val), Some(filt_val)) = (row.gyro_unfilt[axis_index], row.gyro[axis_index]) {
                    unfilt_samples.push(unfilt_val as f32);
                    filt_samples.push(filt_val as f32);
                }
            }

            // Only proceed if both unfiltered and filtered data are available for this axis
            if unfilt_samples.is_empty() || filt_samples.is_empty() {
                continue;
            }

            // For simplicity and aligned FFTs, process only up to the minimum available length.
            let min_len = unfilt_samples.len().min(filt_samples.len());
            if min_len == 0 { continue; }

            let unfilt_samples_slice = &unfilt_samples[0..min_len];
            let filt_samples_slice = &filt_samples[0..min_len];

            // Apply Hanning window (Tukey window with alpha = 1.0)
            let window_func = crate::step_response::tukeywin(min_len, 1.0);

            let unfilt_windowed: Array1<f32> = Array1::from_vec(unfilt_samples_slice.to_vec()) * &window_func;
            let filt_windowed: Array1<f32> = Array1::from_vec(filt_samples_slice.to_vec()) * &window_func;

            // Pad to next power of 2 for efficient FFT computation
            let fft_padded_len = min_len.next_power_of_two();
            let mut padded_unfilt = Array1::<f32>::zeros(fft_padded_len);
            padded_unfilt.slice_mut(s![0..min_len]).assign(&unfilt_windowed);
            let mut padded_filt = Array1::<f32>::zeros(fft_padded_len);
            padded_filt.slice_mut(s![0..min_len]).assign(&filt_windowed);

            // Perform Fast Fourier Transform (FFT)
            let unfilt_spec = crate::fft_utils::fft_forward(&padded_unfilt);
            let filt_spec = crate::fft_utils::fft_forward(&padded_filt);

            if unfilt_spec.is_empty() || filt_spec.is_empty() {
                continue;
            }

            let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();
            let mut filt_series_data: Vec<(f64, f64)> = Vec::new();

            // Calculate frequencies for the x-axis and amplitudes (magnitudes) for the y-axis
            let freq_step = sr_value / fft_padded_len as f64;
            // For real-valued inputs, the spectrum is symmetric, so we only need the first half + 1 (for DC and Nyquist)
            let num_unique_freqs = if fft_padded_len % 2 == 0 { fft_padded_len / 2 + 1 } else { (fft_padded_len + 1) / 2 };

            let mut max_amp_unfilt = 0.0f64;
            let mut max_amp_filt = 0.0f64;

            for i in 0..num_unique_freqs {
                let freq_val = i as f64 * freq_step;
                let amp_unfilt = unfilt_spec[i].norm() as f64; // Calculate magnitude (amplitude), no capping here yet
                let amp_filt = filt_spec[i].norm() as f64; // Calculate magnitude (amplitude), no capping here yet

                unfilt_series_data.push((freq_val, amp_unfilt));
                filt_series_data.push((freq_val, amp_filt));

                max_amp_unfilt = max_amp_unfilt.max(amp_unfilt);
                max_amp_filt = max_amp_filt.max(amp_filt);
            }

            // Calculate dynamic Y-axis cap based on max amplitude after noise floor
            let noise_floor_sample_idx = (SPECTRUM_NOISE_FLOOR_HZ / freq_step) as usize;

            let max_amp_after_noise_floor_unfilt = unfilt_series_data[noise_floor_sample_idx..]
                .iter()
                .map(|&(_, amp)| amp)
                .fold(0.0f64, |max_val, amp| max_val.max(amp));

            let max_amp_after_noise_floor_filt = filt_series_data[noise_floor_sample_idx..]
                .iter()
                .map(|&(_, amp)| amp)
                .fold(0.0f64, |max_val, amp| max_val.max(amp));

            let y_max_unfilt = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_unfilt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);
            let y_max_filt = SPECTRUM_Y_AXIS_FLOOR.max(max_amp_after_noise_floor_filt * SPECTRUM_Y_AXIS_HEADROOM_FACTOR);
            // Store the processed data for this axis
            all_fft_data[axis_index] = Some(([unfilt_series_data, filt_series_data], y_max_unfilt, y_max_filt));
    }

    // Pass the pre-calculated data to the plotting function
    draw_dual_spectrum_plot(
        &output_file,
        root_name,
        plot_type_name,
        // This closure provides the specific plot configuration for each subplot
        // It consumes the `all_fft_data` by moving it.
        move |axis_index| {
            if let Some((series_data, max_amp_unfilt, max_amp_filt)) = all_fft_data[axis_index].take() {
                let max_freq_val = sr_value / 2.0; // Nyquist frequency as max X-axis value
                let x_range = 0.0..max_freq_val * 1.05; // Extend X-axis slightly for readability

                // Calculate Y-axis ranges for each subplot independently
                let (min_amp_unfilt, plot_max_amp_unfilt) = calculate_range(0.0, max_amp_unfilt);

                // We now directly use the calculated y_max values for the range
                let y_range_unfilt = 0.0..max_amp_unfilt;
                let y_range_filt = 0.0..max_amp_filt;

                // Create PlotSeries for unfiltered gyro
                let unfilt_plot_series = vec![
                    PlotSeries {
                        data: series_data[0].clone(), // Clone the Vec to avoid move errors from array
                        label: "Unfiltered Gyro".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_UNFILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];
                // Create PlotSeries for filtered gyro
                let filt_plot_series = vec![
                    PlotSeries {
                        data: series_data[1].clone(), // Clone the Vec to avoid move errors from array
                        label: "Filtered Gyro".to_string(),
                        color: *COLOR_GYRO_VS_UNFILT_FILT,
                        stroke_width: LINE_WIDTH_PLOT,
                    }
                ];

                // Return an array of two Options, one for the left subplot and one for the right
                Some([
                    Some((
                        format!("{} Unfiltered Gyro Spectrum", ["Roll", "Pitch", "Yaw"][axis_index]),
                        x_range.clone(), // Clone x_range as it's used by both plots
                        y_range_unfilt,
                        unfilt_plot_series,
                        "Frequency (Hz)".to_string(),
                        "Amplitude".to_string(),
                    )),
                    Some((
                        format!("{} Filtered Gyro Spectrum", ["Roll", "Pitch", "Yaw"][axis_index]),
                        x_range,
                        y_range_filt,
                        filt_plot_series,
                        "Frequency (Hz)".to_string(),
                        "Amplitude".to_string(),
                    )),
                ])
            } else {
                Some([None, None]) // If no data for this axis, return two Nones to trigger "Data Unavailable" messages
            }
        },
    )
}

// src/plotting_utils.rs
