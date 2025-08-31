// src/plot_framework.rs

use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::element::PathElement;
use plotters::element::Rectangle;
use plotters::element::Text;
use plotters::series::LineSeries;
use plotters::style::colors::{BLACK, RED, WHITE};
use plotters::style::{Color, IntoFont, RGBColor};

use std::error::Error;
use std::ops::Range;

use crate::constants::{
    HEATMAP_MAX_PSD_DB, HEATMAP_MIN_PSD_DB, LINE_WIDTH_LEGEND, PEAK_LABEL_MIN_AMPLITUDE,
    PLOT_HEIGHT, PLOT_WIDTH,
};

/// Calculate plot range with padding.
/// Adds 15% padding, or a fixed padding for very small ranges.
pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let (min, max) = if min_val <= max_val {
        (min_val, max_val)
    } else {
        (max_val, min_val)
    };
    let range = (max - min).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min - padding, max + padding)
}

/// Draw a "Data Unavailable" message on a plot area.
pub fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
    reason: &str,
) -> Result<(), Box<dyn Error>> {
    let axis_names = ["Roll", "Pitch", "Yaw"];
    let axis_name = axis_names[axis_index];
    let (x_range, y_range) = area.get_pixel_range();
    let (width, height) = (
        (x_range.end - x_range.start) as u32,
        (y_range.end - y_range.start) as u32,
    );
    let text_style = ("sans-serif", 20).into_font().color(&RED);
    area.draw(&Text::new(
        format!("{axis_name} {plot_type} Data Unavailable:\n{reason}"),
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

pub struct PlotConfig {
    pub title: String,
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub series: Vec<PlotSeries>,
    pub x_label: String,
    pub y_label: String,
    pub peaks: Vec<(f64, f64)>,
    pub peak_label_threshold: Option<f64>,
    pub peak_label_format_string: Option<String>,
}

pub struct AxisSpectrum {
    pub unfiltered: Option<PlotConfig>,
    pub filtered: Option<PlotConfig>,
}

pub struct HeatmapData {
    pub x_bins: Vec<f64>,
    pub y_bins: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

pub struct HeatmapPlotConfig {
    pub title: String,
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub heatmap_data: HeatmapData,
    pub x_label: String,
    pub y_label: String,
}

pub struct AxisHeatmapSpectrum {
    pub unfiltered: Option<HeatmapPlotConfig>,
    pub filtered: Option<HeatmapPlotConfig>,
}

fn map_db_to_color(db_value: f64, min_db: f64, max_db: f64) -> RGBColor {
    let clamped_db = db_value.max(min_db).min(max_db);
    let normalized_value = (clamped_db - min_db) / (max_db - min_db);

    let color = colorous::VIRIDIS.eval_continuous(normalized_value);
    RGBColor(color.r, color.g, color.b)
}

/// Draws a single chart using a PlotConfig struct, allowing dynamic peak labeling.
/// This version is used by `draw_dual_spectrum_plot` (for PSD and Spectrum plots).
fn draw_single_axis_chart_with_config(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    plot_config: &PlotConfig,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(&plot_config.title, ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(plot_config.x_range.clone(), plot_config.y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc(&plot_config.x_label)
        .y_desc(&plot_config.y_label)
        .x_labels(20)
        .y_labels(10)
        .light_line_style(WHITE.mix(0.7))
        .label_style(("sans-serif", 12))
        .draw()?;

    let mut series_drawn_count = 0;
    for s in &plot_config.series {
        if !s.data.is_empty() {
            chart
                .draw_series(LineSeries::new(
                    s.data.iter().cloned(),
                    s.color.stroke_width(s.stroke_width),
                ))?
                .label(&s.label)
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)],
                        s.color.stroke_width(LINE_WIDTH_LEGEND),
                    )
                });
            series_drawn_count += 1;
        }
    }

    if series_drawn_count > 0 {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 12))
            .draw()?;
    }

    let area_offset = area.get_base_pixel();
    let (area_x_range, area_y_range) = area.get_pixel_range();
    let area_width = area_x_range.end - area_x_range.start;
    let area_height = area_y_range.end - area_y_range.start;
    const TEXT_WIDTH_ESTIMATE: i32 = 300;
    const TEXT_HEIGHT_ESTIMATE: i32 = 20;

    let peak_label_threshold = plot_config
        .peak_label_threshold
        .unwrap_or(PEAK_LABEL_MIN_AMPLITUDE);
    let peak_label_format_string_ref = plot_config
        .peak_label_format_string
        .as_deref()
        .unwrap_or("{:.0}");

    for (idx, &(peak_freq, peak_amp)) in plot_config.peaks.iter().enumerate() {
        if peak_amp > peak_label_threshold {
            let peak_pixel_coords_relative_to_plotting_area =
                chart.backend_coord(&(peak_freq, peak_amp));
            let mut text_x = peak_pixel_coords_relative_to_plotting_area.0 - area_offset.0;
            let mut text_y = peak_pixel_coords_relative_to_plotting_area.1 - area_offset.1;

            let formatted_peak_amp = if peak_label_format_string_ref == "{:.2} dB" {
                format!("{peak_amp:.2} dB")
            } else {
                format!("{peak_amp:.0}")
            };

            let label_text = if idx == 0 {
                format!("Primary Peak: {formatted_peak_amp} at {peak_freq:.0} Hz")
            } else {
                format!("Peak: {formatted_peak_amp} at {peak_freq:.0} Hz")
            };

            text_x = text_x.max(0).min(area_width - TEXT_WIDTH_ESTIMATE);
            text_y = text_y.max(0).min(area_height - TEXT_HEIGHT_ESTIMATE);

            area.draw(&Text::new(
                label_text,
                (text_x, text_y),
                ("sans-serif", 15).into_font().color(&BLACK),
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
    F: FnMut(
            usize,
        ) -> Option<(
            String,
            std::ops::Range<f64>,
            std::ops::Range<f64>,
            Vec<PlotSeries>,
            String,
            String,
        )> + Send
        + Sync
        + 'static,
    <BitMapBackend<'a> as DrawingBackend>::ErrorType: 'static,
{
    let root_area =
        BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?;
    root_area.draw(&Text::new(
        root_name,
        (10, 10),
        ("sans-serif", 24).into_font().color(&BLACK),
    ))?;
    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((3, 1));
    let mut any_axis_plotted = false;

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..3 {
        let area = &sub_plot_areas[axis_index];
        match get_axis_plot_data(axis_index) {
            Some((chart_title, x_range, y_range, series_data, x_label, y_label)) => {
                let has_data = series_data.iter().any(|s| !s.data.is_empty());
                let valid_ranges = x_range.end > x_range.start && y_range.end > y_range.start;
                if has_data && valid_ranges {
                    let temp_plot_config = PlotConfig {
                        title: chart_title,
                        x_range,
                        y_range,
                        series: series_data,
                        x_label,
                        y_label,
                        peaks: vec![],
                        peak_label_threshold: None,
                        peak_label_format_string: None,
                    };
                    draw_single_axis_chart_with_config(area, &temp_plot_config)?;
                    any_axis_plotted = true;
                } else {
                    let reason = if !has_data {
                        "No data points"
                    } else {
                        "Invalid ranges"
                    };
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
        println!("  Stacked plot saved as '{output_filename}'.");
    } else {
        root_area.present()?;
        println!("  Skipping '{output_filename}' plot saving: No data available for any axis to plot, only placeholder messages shown.");
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
    let root_area =
        BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
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
                let has_data = !plot_config.series.is_empty()
                    && plot_config.series.iter().any(|s| !s.data.is_empty());
                let valid_ranges = plot_config.x_range.end > plot_config.x_range.start
                    && plot_config.y_range.end > plot_config.y_range.start;

                if has_data && valid_ranges {
                    draw_single_axis_chart_with_config(area, plot_config)?;
                    any_plot_drawn = true;
                } else {
                    let reason = if !has_data {
                        "No data points"
                    } else {
                        "Invalid ranges"
                    };
                    draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
                }
            } else {
                draw_unavailable_message(area, axis_index, plot_type_name, "Data Not Available")?;
            }
        }
    }

    if any_plot_drawn {
        root_area.present()?;
        println!("  Stacked plot saved as '{output_filename}'.");
    } else {
        root_area.present()?;
        println!("  Skipping '{output_filename}' plot saving: No data available for any axis to plot, only placeholder messages shown.");
    }
    Ok(())
}

/// Draws a single heatmap chart (spectrogram) for one axis within a stacked plot.
fn draw_single_heatmap_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    chart_title: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    x_label: &str,
    y_label: &str,
    heatmap_data: &HeatmapData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(chart_title, ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(10)
        .y_labels(10)
        .light_line_style(WHITE.mix(0.7))
        .label_style(("sans-serif", 12))
        .draw()?;

    // Calculate bin widths for rectangle sizing
    let x_bin_width = if heatmap_data.x_bins.len() > 1 {
        heatmap_data.x_bins[1] - heatmap_data.x_bins[0]
    } else {
        1.0 // fallback for single bin
    };
    let y_bin_width = if heatmap_data.y_bins.len() > 1 {
        heatmap_data.y_bins[1] - heatmap_data.y_bins[0]
    } else {
        1.0 // fallback for single bin
    };

    for (x_idx, &x_val) in heatmap_data.x_bins.iter().enumerate() {
        for (y_idx, &y_val) in heatmap_data.y_bins.iter().enumerate() {
            if let Some(row) = heatmap_data.values.get(x_idx) {
                if let Some(&psd_db) = row.get(y_idx) {
                    let color = map_db_to_color(psd_db, HEATMAP_MIN_PSD_DB, HEATMAP_MAX_PSD_DB);

                    let rect = Rectangle::new(
                        [
                            (x_val - x_bin_width * 0.5, y_val - y_bin_width * 0.5),
                            (x_val + x_bin_width * 0.5, y_val + y_bin_width * 0.5),
                        ],
                        color.filled(),
                    );
                    chart.draw_series(std::iter::once(rect))?;
                }
            }
        }
    }
    Ok(())
}

pub fn draw_dual_heatmap_plot<'a, F>(
    output_filename: &'a str,
    root_name: &str,
    plot_type_name: &str,
    mut get_axis_plot_data: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(usize) -> Option<AxisHeatmapSpectrum> + Send + Sync + 'static,
    <BitMapBackend<'a> as DrawingBackend>::ErrorType: 'static,
{
    let root_area =
        BitMapBackend::new(output_filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
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
                let has_data = !plot_config.heatmap_data.values.is_empty()
                    && plot_config
                        .heatmap_data
                        .values
                        .iter()
                        .any(|row| !row.is_empty());
                let valid_ranges = plot_config.x_range.end > plot_config.x_range.start
                    && plot_config.y_range.end > plot_config.y_range.start;

                if has_data && valid_ranges {
                    draw_single_heatmap_chart(
                        area,
                        &plot_config.title,
                        plot_config.x_range.clone(),
                        plot_config.y_range.clone(),
                        &plot_config.x_label,
                        &plot_config.y_label,
                        &plot_config.heatmap_data,
                    )?;
                    any_plot_drawn = true;
                } else {
                    let reason = if !has_data {
                        "No data points"
                    } else {
                        "Invalid ranges"
                    };
                    draw_unavailable_message(area, axis_index, plot_type_name, reason)?;
                }
            } else {
                draw_unavailable_message(area, axis_index, plot_type_name, "Data Not Available")?;
            }
        }
    }

    if any_plot_drawn {
        root_area.present()?;
        println!("  Stacked heatmap plot saved as '{output_filename}'.");
    } else {
        root_area.present()?;
        println!("  Skipping '{output_filename}' heatmap plot saving: No data available for any axis to plot, only placeholder messages shown.");
    }
    Ok(())
}

// src/plot_framework.rs
