// src/plot_framework.rs

// Plotters imports - explicitly list what's used
use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::style::{RGBColor, IntoFont, Color};
use plotters::element::Text;
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::element::PathElement;
use plotters::series::LineSeries;
use plotters::style::colors::{WHITE, BLACK, RED}; // Keep specific colors used by framework

use std::error::Error;
use std::ops::Range; // Import Range for PlotConfig

// Explicitly import constants used within this file by the framework
use crate::constants::{
    PLOT_WIDTH, PLOT_HEIGHT,
    PEAK_LABEL_MIN_AMPLITUDE,
    LINE_WIDTH_LEGEND,
};

/// Calculate plot range with padding.
/// Adds 15% padding, or a fixed padding for very small ranges.
pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    // Ensure we always return (lower, upper)
    let (min, max) = if min_val <= max_val {
        (min_val, max_val)
    } else {
        (max_val, min_val)
    };
    let range = (max - min).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min - padding, max + padding)}

/// Draw a "Data Unavailable" message on a plot area.
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

// Define a struct to hold data for a single line series
#[derive(Clone)]
pub struct PlotSeries {
    pub data: Vec<(f64, f64)>,
    pub label: String,
    pub color: RGBColor,
    pub stroke_width: u32,
}

/// Struct to hold configuration for a single plot.
pub struct PlotConfig {
    pub title: String,
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub series: Vec<PlotSeries>,
    pub x_label: String,
    pub y_label: String,
    pub peaks: Vec<(f64, f64)>, // Changed from Option<(f64, f64)> to Vec<(f64, f64)>
}

/// Struct to hold spectrum data for a single axis, distinguishing between unfiltered and filtered.
pub struct AxisSpectrum {
    pub unfiltered: Option<PlotConfig>,
    pub filtered:   Option<PlotConfig>,
}

/// Draws a single chart for one axis within a stacked plot.
fn draw_single_axis_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    chart_title: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    x_label: &str,
    y_label: &str,
    series: &[PlotSeries],
    label_prefix: &str, // For primary peak label
    peaks_info: &[(f64, f64)], // Changed from Option<(f64, f64)> to &[(f64, f64)]
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(chart_title, ("sans-serif", 20))
        .margin(5).x_label_area_size(30).y_label_area_size(50)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(20).y_labels(10)
        .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

    let mut series_drawn_count = 0;
    for s in series {
        if !s.data.is_empty() {
            chart.draw_series(LineSeries::new(
                s.data.iter().cloned(),
                s.color.stroke_width(s.stroke_width),
            ))?
            .label(&s.label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s.color.stroke_width(LINE_WIDTH_LEGEND)));
            series_drawn_count += 1;
        }
    }

    if series_drawn_count > 0 {
        chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
            .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
    }

    let plotting_area_offset = chart.plotting_area().get_base_pixel();
    let (area_x_range, area_y_range) = area.get_pixel_range();
    let area_width = (area_x_range.end - area_x_range.start) as i32;
    let area_height = (area_y_range.end - area_y_range.start) as i32;
    const TEXT_WIDTH_ESTIMATE: i32 = 400; 
    const TEXT_HEIGHT_ESTIMATE: i32 = 20;

    for (idx, &(peak_freq, peak_amp)) in peaks_info.iter().enumerate() {
        if peak_amp > PEAK_LABEL_MIN_AMPLITUDE {
            let peak_pixel_coords_relative_to_plotting_area = chart.backend_coord(&(peak_freq, peak_amp));
            
            let mut text_x = peak_pixel_coords_relative_to_plotting_area.0 - plotting_area_offset.0 + 55;
            let mut text_y = peak_pixel_coords_relative_to_plotting_area.1 - plotting_area_offset.1 + 15;

            let label_text;

            if idx == 0 { // Primary peak
                label_text = format!("{} Peak amplitude {:.0} at {:.0} Hz", label_prefix, peak_amp, peak_freq);
            } else { // Secondary peaks
                label_text = format!("Secondary: {:.0} at {:.0} Hz", peak_amp, peak_freq);
            }
            
            // Adjust text position to stay within bounds
            text_x = text_x.max(0).min(area_width - TEXT_WIDTH_ESTIMATE);
            text_y = text_y.max(0).min(area_height - TEXT_HEIGHT_ESTIMATE);

            area.draw(&Text::new(
                label_text,
                (text_x, text_y),
                ("sans-serif", 15).into_font().color(&BLACK)
            ))?;
        }
    }
    Ok(())
}

/// Creates a stacked plot image with three subplots for Roll, Pitch, and Yaw.
pub fn draw_stacked_plot<'a, F>(
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
                         "", // No specific label prefix for generic stacked plots
                         &[], // Pass empty slice for peaks as this function doesn't handle them yet
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

/// Creates a stacked plot image with three rows and two columns for subplots.
pub fn draw_dual_spectrum_plot<'a, F>(
    output_filename: &'a str,
    root_name: &str,
    plot_type_name: &str,
    mut get_axis_plot_data: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<AxisSpectrum> + Send + Sync + 'static,
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
    let sub_plot_areas = margined_root_area.split_evenly((3, 2));
    let mut any_plot_drawn = false;

    for axis_index in 0..3 {
        let plots_for_axis_option = get_axis_plot_data(axis_index);
        let current_axis_name = ["Roll", "Pitch", "Yaw"][axis_index];

        for col_idx in 0..2 {
            let area = &sub_plot_areas[axis_index * 2 + col_idx];
            let plot_config_option = plots_for_axis_option.as_ref().and_then(|axis_spectrum| {
                if col_idx == 0 {
                    axis_spectrum.unfiltered.as_ref()
                } else {
                    axis_spectrum.filtered.as_ref()
                }
            });

            if let Some(plot_config) = plot_config_option {
                let has_data = plot_config.series.iter().any(|s| !s.data.is_empty());
                let valid_ranges = plot_config.x_range.end > plot_config.x_range.start && plot_config.y_range.end > plot_config.y_range.start;

                if has_data && valid_ranges {
                    let label_prefix_string = if col_idx == 0 {
                        format!("{} Unfiltered", current_axis_name)
                    } else {
                        format!("{} Filtered", current_axis_name)
                    };
                    draw_single_axis_chart(
                        area,
                        &plot_config.title,
                        plot_config.x_range.clone(),
                        plot_config.y_range.clone(),
                        &plot_config.x_label,
                        &plot_config.y_label,
                        &plot_config.series,
                        label_prefix_string.as_str(),
                        &plot_config.peaks, // Pass the Vec of peaks
                    )?;
                    any_plot_drawn = true;
                } else {
                    let reason = if !has_data { "No data points" } else { "Invalid ranges" };
                    draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
                }
            } else {
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

// src/plot_framework.rs
