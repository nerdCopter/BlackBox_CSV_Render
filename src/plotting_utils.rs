// src/plotting_utils.rs

// Plotters imports - explicitly list what's used
use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
// Corrected imports: Removed IntoRGB and CORNFLOWERBLUE as they are not used directly here
use plotters::style::{RGBColor, IntoFont, Color}; // Import ShapeStyle for tick configuration
use plotters::style::colors::{WHITE, BLACK, RED}; // Keep necessary standard colors
// Corrected import for GREY
use plotters::style::colors::full_palette::GREY;
use plotters::element::Text;
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::element::PathElement;
use plotters::series::LineSeries;
use plotters::element::Rectangle; // Import Rectangle for heatmap cells
use plotters::coord::Shift; // Import Shift for drawing area types
// Removed unused EmptyElement import
// use plotters::prelude::EmptyElement; // Import EmptyElement if needed


use std::error::Error;

use ndarray::{Array1, Array2, s};
// Keep QuantileExt here as it's used within the plot_step_response closure
use ndarray_stats::QuantileExt;

// Explicitly import constants used within this file
// NOTE: We will bind these to local variables outside the closures where needed
use crate::constants::{
    PLOT_WIDTH, PLOT_HEIGHT, STEP_RESPONSE_PLOT_DURATION_S,
    SETPOINT_THRESHOLD, POST_AVERAGING_SMOOTHING_WINDOW,
    STEADY_STATE_START_S, STEADY_STATE_END_S,
    // Import specific color constants needed
    COLOR_PIDSUM_MAIN, COLOR_PIDERROR_MAIN, COLOR_SETPOINT_MAIN,
    COLOR_SETPOINT_VS_PIDSUM_SP, COLOR_SETPOINT_VS_PIDSUM_PID,
    COLOR_SETPOINT_VS_GYRO_SP, COLOR_SETPOINT_VS_GYRO_GYRO,
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT,
    COLOR_STEP_RESPONSE_LOW_SP, COLOR_STEP_RESPONSE_HIGH_SP, COLOR_STEP_RESPONSE_COMBINED,
    LINE_WIDTH_PLOT, LINE_WIDTH_LEGEND, // Import line widths
    // Import spectrograph constants
    SPECTROGRAPH_MAX_FREQ_HZ, // Keep for default/fallback frequency range
    // Removed unused spectrograph color/power constants from here
};
use crate::log_data::LogRowData; // Still needed for other plots
use crate::step_response; // Still needed for step response plot
use crate::spectrograph::{SpectrographData, map_log_power_to_color}; // Import spectrograph types and helper


/// Calculate plot range with padding.
/// Adds 15% padding, or a fixed padding for very small ranges.
pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

/// Draw a "Data Unavailable" message on a plot area.
// Add axis_index parameter
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
pub struct PlotSeries { // Removed lifetime 'a
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
        // Set the number of labels explicitly to control grid density
        .x_labels(10).y_labels(5)
        // Change grid line color to grey
        .light_line_style(&GREY.mix(0.5)).label_style(("sans-serif", 12).into_font().color(&BLACK)).draw()?; // Labels in Black on white background

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
    // main_plot_title: &str, // Removed main_plot_title
    plot_type_name: &str, // Name used in unavailability message
    mut get_axis_plot_data: F, // Changed F to be mutable
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<(String, std::ops::Range<f64>, std::ops::Range<f64>, Vec<PlotSeries>, String, String)> + Send + Sync + 'static, // Changed Fn to FnMut
    // BitMapBackend error type needs to be 'static, add 'a lifetime
    <BitMapBackend<'a> as DrawingBackend>::ErrorType: 'static,
{
    let root_area = BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area(); // Use PLOT_WIDTH/HEIGHT from constants
    root_area.fill(&WHITE)?; // Default background is white

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
                     draw_single_axis_chart( // Removed axis_index argument
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
    // let main_plot_title = "PIDsum vs PID Error vs Setpoint"; // Removed
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
        // main_plot_title, // Removed
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


/// Generates the Stacked Setpoint vs PIDsum Plot (Yellow, Red)
pub fn plot_setpoint_vs_pidsum(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
    // let main_plot_title = "Setpoint vs PIDsum"; // Removed
    let plot_type_name = "Setpoint/PIDsum";

     // Prepare data for all axes in one pass
    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                 let pidsum = row.p_term[axis_index].and_then(|p| {
                    row.i_term[axis_index].and_then(|i| {
                        row.d_term[axis_index].map(|d| p + i + d)
                    })
                });
                axis_plot_data[axis_index].push((time, row.setpoint[axis_index], pidsum));
            }
        }
    }

    // Bind constants to local variables outside the closure
    let color_setpoint: RGBColor = *COLOR_SETPOINT_VS_PIDSUM_SP; // Dereference the static reference
    let color_pidsum_vs: RGBColor = *COLOR_SETPOINT_VS_PIDSUM_PID; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    draw_stacked_plot(
        &output_file_setpoint,
        root_name,
        // main_plot_title, // Removed
        plot_type_name,
        move |axis_index| { // Use move to capture axis_plot_data and local constants
             let data = &axis_plot_data[axis_index];
             if data.is_empty() {
                 return None;
             }

             let mut setpoint_series_data: Vec<(f64, f64)> = Vec::new();
             let mut pidsum_series_data: Vec<(f64, f64)> = Vec::new();

             let mut time_min = f64::INFINITY;
             let mut time_max = f64::NEG_INFINITY;
             let mut val_min = f64::INFINITY;
             let mut val_max = f64::NEG_INFINITY;

             for (time, setpoint, pidsum) in data {
                 time_min = time_min.min(*time);
                 time_max = time_max.max(*time);

                 if let Some(s) = setpoint {
                     setpoint_series_data.push((*time, *s));
                     val_min = val_min.min(*s);
                     val_max = val_max.max(*s);
                 }
                 if let Some(p) = pidsum {
                     pidsum_series_data.push((*time, *p));
                     val_min = val_min.min(*p);
                     val_max = val_max.max(*p);
                 }
             }

            if setpoint_series_data.is_empty() && pidsum_series_data.is_empty() {
                 return None; // No actual data collected for this axis
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !setpoint_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: setpoint_series_data,
                     label: "Setpoint".to_string(),
                     color: color_setpoint, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
                 });
             }
             if !pidsum_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: pidsum_series_data,
                     label: "PIDsum".to_string(),
                     color: color_pidsum_vs, // Use captured constant (RGBColor)
                     stroke_width: line_stroke_plot, // Use captured constant
                 });
            }

            Some((
                format!("Axis {} Setpoint vs PIDsum", axis_index),
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
    // let main_plot_title = "Setpoint vs Gyro"; // Removed
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
        // main_plot_title, // Removed
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
    // let main_plot_title = "Gyro vs Unfiltered Gyro"; // Removed
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
    // Corrected typo: UNfilt -> UNFILT
    let color_gyro_filt: RGBColor = *COLOR_GYRO_VS_UNFILT_FILT; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    draw_stacked_plot(
        &output_file_gyro,
        root_name,
        // main_plot_title, // Removed
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
    // Mark as unused with _
    let _steady_state_start_s = STEADY_STATE_START_S;
    let _steady_state_end_s = STEADY_STATE_END_S;
    let setpoint_threshold = SETPOINT_THRESHOLD;
    let post_averaging_smoothing_window = POST_AVERAGING_SMOOTHING_WINDOW;
    let color_high_sp: RGBColor = *COLOR_STEP_RESPONSE_HIGH_SP; // Dereference the static reference
    let color_combined: RGBColor = *COLOR_STEP_RESPONSE_COMBINED; // Dereference the static reference
    let color_low_sp: RGBColor = *COLOR_STEP_RESPONSE_LOW_SP; // Dereference the static reference
    let line_stroke_plot = LINE_WIDTH_PLOT; // Use plot width

    let output_file_step = format!("{}_step_response_stacked_plot_{}s.png", root_name, step_response_plot_duration_s); // Use captured variable
    // let main_plot_title = &format!("Step Response (~{}s)", step_response_plot_duration_s); // Removed
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
            // Move calculation outside the closure definition
            let ss_start_sample = (STEADY_STATE_START_S * sr).floor() as usize;
            let ss_end_sample = (STEADY_STATE_END_S * sr).ceil() as usize;

            // Ensure steady state window is within the response length
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
                         // Fix: Use if let Ok(...) for min/max on Array1 from ndarray-stats
                         steady_state_segment.mean()
                            .and_then(|steady_state_mean| {
                                 if steady_state_mean.abs() > 1e-9 {
                                      let normalized_response = shifted_response.mapv(|v| v / steady_state_mean);
                                      // Fix: Use if let Ok(...) for min/max on Array1 from ndarray-stats
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

            // Now call process_response with the variables calculated above
            let final_low_response = process_response(&low_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);
            let final_high_response = process_response(&high_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);
            let final_combined_response = process_response(&combined_mask, valid_stacked_responses, response_length_samples, current_ss_start_sample, current_ss_end_sample, post_averaging_smoothing_window);

            // Clone the Options so we can check is_some() and then move the contained Array1
            let final_low_response_cloned = final_low_response.clone();
            let final_high_response_cloned = final_high_response.clone();
            let final_combined_response_cloned = final_combined_response.clone();

            let is_low_response_valid = final_low_response_cloned.is_some();
            let is_high_response_valid = final_high_response_cloned.is_some();
            let is_combined_response_valid = is_high_response_valid && final_combined_response_cloned.is_some(); // Only plot combined if high is valid


            if !(is_low_response_valid || is_high_response_valid) { // Only need low or high to be valid to plot axis
                continue; // Skip this axis if no valid responses
            }

            let mut resp_min = f64::INFINITY;
            let mut resp_max = f64::NEG_INFINITY;
            // Fix: Use if let Ok(...) for min/max on Array1 from ndarray-stats
            if let Some(resp) = &final_low_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            if let Some(resp) = &final_high_response_cloned { if let Ok(min_val) = resp.min() { resp_min = resp_min.min(*min_val); } if let Ok(max_val) = resp.max() { resp_max = resp_max.max(*max_val); } }
            // Only include combined in range calculation if it will be plotted (i.e., if high_response is valid)
            if is_combined_response_valid { // Use the boolean flag
                 // Fix: Use if let Ok(...) for min/max on Array1 from ndarray-stats
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
            // Plot Combined LAST for z-index, but ONLY if high setpoint response was valid
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
        // main_plot_title, // Removed
        plot_type_name,
        move |axis_index| {
            // Take ownership of the Option for this axis
            plot_data_per_axis[axis_index].take()
        },
    )
}

/// Draws a single chart for one axis within a stacked spectrograph plot.
// Takes optional data for filtered and unfiltered spectrographs for this axis.
fn draw_single_axis_spectrograph(
    area: &DrawingArea<BitMapBackend<'_>, Shift>, // Use concrete type
    axis_index: usize, // Pass axis index for unavailable message
    filtered_data: Option<SpectrographData>,
    unfiltered_data: Option<SpectrographData>,
) -> Result<(), Box<dyn Error>> {

    let has_filtered = filtered_data.is_some();
    let has_unfiltered = unfiltered_data.is_some();

    if !has_filtered && !has_unfiltered {
        // Use axis_index here
        area.fill(&WHITE)?; // Fill white even if no data for consistency
        draw_unavailable_message(area, axis_index, "Spectrograph", "No data available for filtered or unfiltered gyro")?;
        return Ok(())
    }

    // Split the area horizontally to plot filtered and unfiltered side-by-side
    // Fix: split_evenly returns a Vec, access elements by index
    let areas = area.split_evenly((1, 2));
    let left_area = &areas[0];
    let right_area = &areas[1];


    // Determine frequency range (X-axis) from the actual data
    let freq_range_start = if has_filtered {
        filtered_data.as_ref().unwrap().frequency.get(0).copied().unwrap_or(0.0)
    } else if has_unfiltered {
        unfiltered_data.as_ref().unwrap().frequency.get(0).copied().unwrap_or(0.0)
    } else {
        0.0 // Default if no data
    };

    let freq_range_end = if has_filtered {
        let num_freq_bins = filtered_data.as_ref().unwrap().frequency.len();
         if num_freq_bins > 0 { filtered_data.as_ref().unwrap().frequency[num_freq_bins - 1] } else { SPECTROGRAPH_MAX_FREQ_HZ }
    } else if has_unfiltered {
        let num_freq_bins = unfiltered_data.as_ref().unwrap().frequency.len();
         if num_freq_bins > 0 { unfiltered_data.as_ref().unwrap().frequency[num_freq_bins - 1] } else { SPECTROGRAPH_MAX_FREQ_HZ }
    } else {
        SPECTROGRAPH_MAX_FREQ_HZ // Default if no data
    };

    let freq_range = freq_range_start..freq_range_end;


    // Determine throttle range (Y-axis)
    let throttle_range = 0.0..100.0; // Throttle percentage from 0% to 100%


    // --- Plot Filtered Spectrograph (Left Side) ---
    if let Some(data) = filtered_data {
         let mut chart = ChartBuilder::on(&left_area)
             // Use axis_index in title
             // Change title color to black for white background
             .caption(format!("Axis {} Filtered Gyro", axis_index), ("sans-serif", 15).into_font().color(&BLACK))
             .margin(5)
             .x_label_area_size(30)
             .y_label_area_size(50) // Ensure space for labels
             // X-axis: Frequency, Y-axis: Throttle Percentage
             .build_cartesian_2d(freq_range.clone(), throttle_range.clone())?;

        // Fill background with white
        left_area.fill(&WHITE)?;

         chart.configure_mesh()
             // Add axis descriptions and labels
             .x_desc("Frequency (Hz)")
             .y_desc("Throttle (%)")
             // Configure Y-axis labels for 0, 25, 50, 75, 100 % explicitly
             //.y_label_area_size(50) // Ensure space for labels
             .y_labels(5) // Request 5 labels (0, 25, 50, 75, 100)
             .x_label_style(("sans-serif", 10).into_font().color(&BLACK))
             .y_label_style(("sans-serif", 10).into_font().color(&BLACK))
             .draw()?;


         // Manually draw heatmap rectangles
         let power_shape = data.power.shape(); // Power is (time_bins, freq_bins)
         let num_time_bins = power_shape[0];
         let num_freq_bins = power_shape[1];

         // Get estimated steps for rectangle size
         // Use frequency vector for X step
         let estimated_freq_step = if num_freq_bins > 1 { data.frequency.get(1).copied().unwrap_or(0.0) - data.frequency.get(0).copied().unwrap_or(0.0) } else { (freq_range.end - freq_range.start) / num_freq_bins.max(1) as f64 }; // Fallback step
         // Use throttle vector for Y step
         let estimated_throttle_step = if num_time_bins > 1 { data.throttle.get(1).copied().unwrap_or(0.0) - data.throttle.get(0).copied().unwrap_or(0.0) } else { (throttle_range.end - throttle_range.start) / num_time_bins.max(1) as f64 }; // Fallback step


         let df_half = estimated_freq_step.abs() / 2.0;
         let dt_half = estimated_throttle_step.abs() / 2.0;


         for time_idx in 0..num_time_bins {
             let throttle_val = data.throttle[time_idx]; // This is the Y-center for this row of freq bins

             for freq_idx in 0..num_freq_bins {
                 let f = data.frequency[freq_idx]; // This is the X-center for this column

                 let log_power = data.power[[time_idx, freq_idx]]; // Access power data

                 let color = map_log_power_to_color(log_power);

                 // Define rectangle corners (X: Frequency, Y: Throttle %)
                 let x1 = f - df_half;
                 let x2 = f + df_half;
                 let y1 = throttle_val - dt_half;
                 let y2 = throttle_val + dt_half;

                 // Ensure coordinates are within plot bounds
                 let x1 = x1.max(freq_range.start).min(freq_range.end);
                 let x2 = x2.max(freq_range.start).min(freq_range.end);
                 let y1 = y1.max(throttle_range.start).min(throttle_range.end);
                 let y2 = y2.max(throttle_range.start).min(throttle_range.end);


                 // Draw the rectangle
                 if x1 < x2 && y1 < y2 { // Only draw if the rectangle has positive dimensions
                     chart.draw_series(std::iter::once(
                         Rectangle::new([(x1, y1), (x2, y2)], color.filled())
                     ))?;
                 }
             }
         }


     } else {
        // Use axis_index here
        left_area.fill(&WHITE)?; // Fill background white even if unavailable
        draw_unavailable_message(&left_area, axis_index, "Filtered Spectrograph", "Data not available")?;
     }

    // --- Plot Unfiltered Spectrograph (Right Side) ---
    if let Some(data) = unfiltered_data {
        let mut chart = ChartBuilder::on(&right_area)
            // Use axis_index in title
            // Change title color to black for white background
            .caption(format!("Axis {} Unfiltered Gyro", axis_index), ("sans-serif", 15).into_font().color(&BLACK))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50) // Adjusted Y label size
            .y_label_area_size(0) // No Y-label area on the right plot
            // X-axis: Frequency, Y-axis: Throttle Percentage
            .build_cartesian_2d(freq_range.clone(), throttle_range.clone())?;

        // Fill background with white
        right_area.fill(&WHITE)?;

        chart.configure_mesh()
            // Add axis descriptions and labels
            .x_desc("Frequency (Hz)")
            // No Y-desc on the right chart
            .x_label_style(("sans-serif", 10).into_font().color(&BLACK))
             // Y-axis labels are needed, but not the description or mesh lines for Y
            .y_labels(5) // Request 5 labels (0, 25, 50, 75, 100)
            .y_label_style(("sans-serif", 10).into_font().color(&BLACK))
            .draw()?;


         // Manually draw heatmap rectangles
         let power_shape = data.power.shape(); // Power is (time_bins, freq_bins)
         let num_time_bins = power_shape[0];
         let num_freq_bins = power_shape[1];

         // Get estimated steps (should be similar, but recalculate just in case)
         // Use frequency vector for X step
         let estimated_freq_step = if num_freq_bins > 1 { data.frequency.get(1).copied().unwrap_or(0.0) - data.frequency.get(0).copied().unwrap_or(0.0) } else { (freq_range.end - freq_range.start) / num_freq_bins.max(1) as f64 }; // Fallback step
         // Use throttle vector for Y step
         let estimated_throttle_step = if num_time_bins > 1 { data.throttle.get(1).copied().unwrap_or(0.0) - data.throttle.get(0).copied().unwrap_or(0.0) } else { (throttle_range.end - throttle_range.start) / num_time_bins.max(1) as f64 }; // Fallback step


         let df_half = estimated_freq_step.abs() / 2.0;
         let dt_half = estimated_throttle_step.abs() / 2.0;


         for time_idx in 0..num_time_bins {
             let throttle_val = data.throttle[time_idx]; // This is the Y-center for this row of freq bins

             for freq_idx in 0..num_freq_bins {
                 let f = data.frequency[freq_idx]; // This is the X-center for this column

                 let log_power = data.power[[time_idx, freq_idx]]; // Access power data
                 let color = map_log_power_to_color(log_power);

                 // Define rectangle corners (X: Frequency, Y: Throttle %)
                 let x1 = f - df_half;
                 let x2 = f + df_half;
                 let y1 = throttle_val - dt_half;
                 let y2 = throttle_val + dt_half;

                  // Ensure coordinates are within plot bounds
                 let x1 = x1.max(freq_range.start).min(freq_range.end);
                 let x2 = x2.max(freq_range.start).min(freq_range.end);
                 let y1 = y1.max(throttle_range.start).min(throttle_range.end);
                 let y2 = y2.max(throttle_range.start).min(throttle_range.end);

                 // Draw the rectangle
                 if x1 < x2 && y1 < y2 { // Only draw if the rectangle has positive dimensions
                     chart.draw_series(std::iter::once(
                         Rectangle::new([(x1, y1), (x2, y2)], color.filled())
                     ))?;
                 }
             }
         }


     } else {
        // Use axis_index here
        right_area.fill(&WHITE)?; // Fill background white even if unavailable
        draw_unavailable_message(&right_area, axis_index, "Unfiltered Spectrograph", "Data not available")?;
     }


    Ok(())
}

/// Creates a stacked spectrograph plot image with three subplots for Roll, Pitch, and Yaw,
/// showing filtered and unfiltered spectrographs side-by-side for each axis.
// New function specifically for spectrographs
pub fn plot_spectrographs(
    spectrograph_filtered_results: &[Option<SpectrographData>; 3], // Filtered spectrograph results
    spectrograph_unfiltered_results: &[Option<SpectrographData>; 3], // Unfiltered spectrograph results
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_filename = format!("{}_gyro_spectrographs.png", root_name);
    let plot_type_name = "Gyro Spectrograph";

    let root_area = BitMapBackend::new(&output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?; // Fill the root area white

    // Main title (now in black for white background)
    root_area.draw(&Text::new(
        root_name,
        (10, 10),
        ("sans-serif", 24).into_font().color(&BLACK), // Use BLACK for main title
    ))?;

    // Create a margined area below the title for the subplots
    let margined_root_area = root_area.margin(50, 5, 5, 5); // Top margin 50px

    // Split the margined area into subplots (3 rows, 1 column)
    let sub_plot_areas = margined_root_area.split_evenly((3, 1));

    let mut any_axis_plotted = false;

    for axis_index in 0..3 {
        let area = &sub_plot_areas[axis_index];
        // Removed axis_title_prefix here as it's built inside draw_single_axis_spectrograph

        let filtered_data = spectrograph_filtered_results[axis_index].clone(); // Clone the data for the closure
        let unfiltered_data = spectrograph_unfiltered_results[axis_index].clone(); // Clone the data

        if filtered_data.is_some() || unfiltered_data.is_some() {
            draw_single_axis_spectrograph(
                area,
                axis_index, // Pass axis_index
                filtered_data,
                unfiltered_data,
            )?;
            any_axis_plotted = true;
        } else {
            // If no data for this axis at all
             area.fill(&WHITE)?; // Fill background white even if unavailable
             draw_unavailable_message(area, axis_index, plot_type_name, "No data available for this axis")?;
        }
    }

    if any_axis_plotted {
         root_area.present()?;
         println!("  Stacked spectrograph plot saved as '{}'.", output_filename);
     } else {
         // Present the plot with "Unavailable" messages if no axes had data
         root_area.present()?;
         println!("  Skipping '{}' spectrograph plot saving: No data available for any axis to plot, only placeholder messages shown.", output_filename);
     }


    Ok(())
}

// src/plotting_utils.rs