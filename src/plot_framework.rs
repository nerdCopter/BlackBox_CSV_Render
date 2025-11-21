// src/plot_framework.rs

use plotters::backend::{BitMapBackend, DrawingBackend};
use plotters::chart::{ChartBuilder, SeriesLabelPosition};
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::element::Circle;
use plotters::element::PathElement;
use plotters::element::Rectangle;
use plotters::element::Text;
use plotters::series::LineSeries;
use plotters::style::colors::{BLACK, RED, WHITE};
use plotters::style::{Color, IntoFont, RGBColor};

use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::sync::OnceLock;

use crate::constants::{
    FILTERED_D_TERM_MIN_THRESHOLD, FONT_SIZE_AXIS_LABEL, FONT_SIZE_CHART_TITLE, FONT_SIZE_LEGEND,
    FONT_SIZE_MAIN_TITLE, FONT_SIZE_MESSAGE, FONT_SIZE_PEAK_LABEL, HEATMAP_MIN_PSD_DB,
    LINE_WIDTH_LEGEND, MAX_PEAKS_TO_LABEL, PEAK_LABEL_BOTTOM_MARGIN_PX, PEAK_LABEL_MIN_AMPLITUDE,
    PLOT_HEIGHT, PLOT_WIDTH, PSD_PEAK_LABEL_MIN_VALUE_DB, RIGHT_ALIGN_THRESHOLD,
};

/// Special prefix for cutoff line series to avoid showing them in legends
pub const CUTOFF_LINE_PREFIX: &str = "__CUTOFF_LINE__";
/// Special prefix for dotted cutoff line series
pub const CUTOFF_LINE_DOTTED_PREFIX: &str = "__CUTOFF_LINE_DOTTED__";

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
    // Constants for text rendering
    const CHAR_WIDTH_RATIO: f32 = 0.6; // Approximate character width relative to font size
    const LINE_HEIGHT_SPACING: i32 = 4; // Additional spacing between lines

    let axis_name = if axis_index < crate::axis_names::AXIS_NAMES.len() {
        crate::axis_names::AXIS_NAMES[axis_index]
    } else {
        "Unknown"
    };
    let (x_range, y_range) = area.get_pixel_range();
    let (width, height) = (
        (x_range.end - x_range.start) as u32,
        (y_range.end - y_range.start) as u32,
    );
    let message = format!("{axis_name} {plot_type} Data Unavailable:\n{reason}");

    // Estimate text dimensions for better centering
    let estimated_char_width = (FONT_SIZE_MESSAGE as f32 * CHAR_WIDTH_RATIO) as i32;
    let estimated_line_height = FONT_SIZE_MESSAGE + LINE_HEIGHT_SPACING;

    // Find the longest line to estimate width
    let lines: Vec<&str> = message.split('\n').collect();
    let max_line_length = lines.iter().map(|line| line.len()).max().unwrap_or(0);
    let estimated_text_width = max_line_length.saturating_mul(estimated_char_width as usize) as i32;
    let estimated_text_height = lines.len().saturating_mul(estimated_line_height as usize) as i32;

    // Calculate center position with better offset estimation
    let center_x = width as i32 / 2 - estimated_text_width / 2;
    let center_y = height as i32 / 2 - estimated_text_height / 2;

    let text_style = ("sans-serif", FONT_SIZE_MESSAGE).into_font().color(&RED);
    area.draw(&Text::new(message, (center_x, center_y), text_style))?;
    Ok(())
}

#[derive(Clone)]
pub struct PlotSeries {
    pub data: Vec<(f64, f64)>,
    pub label: String,
    pub color: RGBColor,
    pub stroke_width: u32,
}

/// Represents a frequency range to be shaded on the plot (e.g., dynamic notch range)
#[derive(Clone)]
pub struct FrequencyRange {
    pub min_hz: f64,
    pub max_hz: f64,
    pub color: RGBColor,
    pub opacity: f64, // 0.0 to 1.0
    pub label: String,
}

#[derive(Clone)]
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
    pub frequency_ranges: Option<Vec<FrequencyRange>>,
}

#[derive(Clone)]
pub struct AxisSpectrum {
    pub unfiltered: Option<PlotConfig>,
    pub filtered: Option<PlotConfig>,
}

#[derive(Clone)]
pub struct HeatmapData {
    pub x_bins: Vec<f64>,
    pub y_bins: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct HeatmapPlotConfig {
    pub title: String,
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub heatmap_data: HeatmapData,
    pub x_label: String,
    pub y_label: String,
    pub max_db: f64, // Maximum dB value for color scaling
}

#[derive(Clone)]
pub struct AxisHeatmapSpectrum {
    pub unfiltered: Option<HeatmapPlotConfig>,
    pub filtered: Option<HeatmapPlotConfig>,
}

fn map_db_to_color(db_value: f64, min_db: f64, max_db: f64) -> RGBColor {
    // Validate input parameters
    if !db_value.is_finite() || !min_db.is_finite() || !max_db.is_finite() {
        return RGBColor(0, 0, 0); // Black for invalid values
    }

    // Ensure span is non-zero to avoid division by zero
    let span = (max_db - min_db).abs().max(1e-9);

    // Clamp db_value to valid range
    let clamped_db = db_value.clamp(min_db, max_db);

    // Compute normalized value with clamping to ensure [0.0, 1.0] range
    let t = ((clamped_db - min_db) / span).clamp(0.0, 1.0);

    let color = colorous::VIRIDIS.eval_continuous(t);
    RGBColor(color.r, color.g, color.b)
}

/// Draws a single chart using a PlotConfig struct, allowing dynamic peak labeling.
/// This version is used by `draw_dual_spectrum_plot` (for PSD and Spectrum plots).
fn draw_single_axis_chart_with_config(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    plot_config: &PlotConfig,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(&plot_config.title, ("sans-serif", FONT_SIZE_CHART_TITLE))
        .margin(5)
        .x_label_area_size(50) // Increased for more space below horizontal axis label
        .y_label_area_size(50)
        .build_cartesian_2d(plot_config.x_range.clone(), plot_config.y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc(&plot_config.x_label)
        .y_desc(&plot_config.y_label)
        .x_labels(20)
        .y_labels(10)
        .y_label_formatter(&|y| {
            // Format Y-axis labels with "k" and "M" notation for large values (spectrum plots)
            // Keep dB values as-is (they're typically small/negative)
            // Use decimal formatting for normalized response values (step response plots)
            if !plot_config.y_label.contains("dB") {
                if y.abs() >= 1_000_000.0 {
                    format!("{:.1}M", y / 1_000_000.0)
                } else if y.abs() >= 1000.0 {
                    format!("{:.0}k", y / 1000.0)
                } else if y.abs() < 10.0
                    && (y.fract() != 0.0 || plot_config.y_label.contains("Response"))
                {
                    // Use decimal formatting for small values with fractional parts
                    // or for step response plots (which use normalized values 0.0-2.0)
                    format!("{:.1}", y)
                } else {
                    format!("{:.0}", y)
                }
            } else {
                format!("{:.0}", y)
            }
        })
        .light_line_style(WHITE.mix(0.7))
        .label_style(("sans-serif", FONT_SIZE_AXIS_LABEL))
        .draw()?;

    // Draw frequency range shading BEFORE series (so data appears on top)
    if let Some(ranges) = &plot_config.frequency_ranges {
        for range in ranges {
            // Create a semi-transparent shaded rectangle covering the frequency range
            let rect_color = range.color.mix(range.opacity);

            // Draw the shaded region from min to max frequency, covering full Y range
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (range.min_hz, plot_config.y_range.start),
                    (range.max_hz, plot_config.y_range.end),
                ],
                rect_color.filled(),
            )))?;
        }
    }

    let mut legend_series_count = 0;

    // Add series to legend FIRST (so they appear before frequency ranges)
    for s in &plot_config.series {
        // Handle legend-only series (empty data but has label)
        if s.data.is_empty() {
            if !s.label.is_empty() {
                // Create a dummy invisible point just for the legend entry
                let dummy_x = plot_config.x_range.start;
                let dummy_y = plot_config.y_range.start;
                chart
                    .draw_series(std::iter::once(Circle::new(
                        (dummy_x, dummy_y),
                        0, // Size 0 = invisible
                        s.color.filled(),
                    )))?
                    .label(&s.label)
                    .legend(move |(x, y)| {
                        // For invisible legend entries (stroke_width == 0), use a zero-length PathElement
                        // instead of empty vec which may be brittle
                        PathElement::new(
                            if s.stroke_width == 0 {
                                vec![] // Zero-length path for invisible entries
                            } else {
                                vec![(x, y), (x + 20, y)] // Normal legend line
                            },
                            s.color.stroke_width(if s.stroke_width == 0 {
                                0
                            } else {
                                LINE_WIDTH_LEGEND
                            }),
                        )
                    });
                legend_series_count += 1;
            }
            continue;
        }

        // Series has data - draw it
        if !s.data.is_empty() {
            // Special handling for cutoff lines: label starts with __CUTOFF_LINE__
            if (s.label.starts_with(CUTOFF_LINE_PREFIX)
                || s.label.starts_with(CUTOFF_LINE_DOTTED_PREFIX))
                && s.data.len() == 2
            {
                // Draw a vertical line at the cutoff frequency without adding to legend
                let mut cutoff_freq = s.data[0].0;
                if !cutoff_freq.is_finite() {
                    continue; // skip malformed input
                }
                // Keep the line within the plotted X range
                cutoff_freq = cutoff_freq.clamp(plot_config.x_range.start, plot_config.x_range.end);
                let y0 = plot_config.y_range.start;
                let y1 = plot_config.y_range.end;

                // Draw dotted line for CUTOFF_LINE_DOTTED_PREFIX, solid line for regular CUTOFF_LINE_PREFIX
                if s.label.starts_with(CUTOFF_LINE_DOTTED_PREFIX) {
                    // Draw dotted line by drawing small segments
                    let y_range = y1 - y0;
                    let num_segments = 20; // Number of dash segments
                    let segment_length = y_range / (num_segments as f64 * 2.0); // Half for dash, half for gap

                    for i in 0..num_segments {
                        let y_start = y0 + (i as f64 * 2.0) * segment_length;
                        let y_end = y_start + segment_length;
                        if y_end <= y1 {
                            chart.draw_series(LineSeries::new(
                                vec![(cutoff_freq, y_start), (cutoff_freq, y_end)],
                                s.color.stroke_width(s.stroke_width),
                            ))?;
                        }
                    }
                } else {
                    // Draw solid line
                    chart.draw_series(LineSeries::new(
                        vec![(cutoff_freq, y0), (cutoff_freq, y1)],
                        s.color.stroke_width(s.stroke_width),
                    ))?;
                }
                // No legend entry for cutoff lines
            } else {
                // Regular series - only add legend if label is not empty
                let series = chart.draw_series(LineSeries::new(
                    s.data.iter().cloned(),
                    s.color.stroke_width(s.stroke_width),
                ))?;

                if !s.label.is_empty() {
                    series.label(&s.label).legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)],
                            s.color.stroke_width(LINE_WIDTH_LEGEND),
                        )
                    });
                    legend_series_count += 1;
                }
            }
        }
    }

    // Add frequency ranges to legend AFTER series (so they appear at the end of legend)
    if let Some(ranges) = &plot_config.frequency_ranges {
        for range in ranges {
            if !range.label.is_empty() {
                // Create a dummy series for legend entry with the range color
                let rect_color = range.color.mix(0.4);
                // Draw an invisible point to create a legend entry
                chart
                    .draw_series(std::iter::once(PathElement::new(
                        vec![(plot_config.x_range.start, plot_config.y_range.start)],
                        rect_color.stroke_width(0),
                    )))?
                    .label(&range.label)
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)],
                            rect_color.stroke_width(LINE_WIDTH_LEGEND),
                        )
                    });
                legend_series_count += 1;
            }
        }
    }

    if legend_series_count > 0 {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", FONT_SIZE_LEGEND))
            .draw()?;
    }

    let area_offset = area.get_base_pixel();
    let area_x_range = area.get_pixel_range().0;
    let area_width = area_x_range.end - area_x_range.start;
    let area_y_range = area.get_pixel_range().1;
    let area_height = area_y_range.end - area_y_range.start;

    let peak_label_threshold = plot_config.peak_label_threshold.unwrap_or_else(|| {
        // Select appropriate threshold based on plot type
        if plot_config.y_label.contains("dB") {
            // PSD plots use dB scale
            PSD_PEAK_LABEL_MIN_VALUE_DB
        } else if plot_config.title.contains("D-term") {
            // D-term plots need higher threshold due to different amplitude scale
            FILTERED_D_TERM_MIN_THRESHOLD
        } else {
            // Default to gyro spectrum threshold for other plot types
            PEAK_LABEL_MIN_AMPLITUDE
        }
    });
    let peak_label_format_string_ref = plot_config
        .peak_label_format_string
        .as_deref()
        .unwrap_or("{:.0}");

    // Collect and prepare peak labels for bottom positioning
    let mut peak_labels = Vec::new();
    for &(peak_freq, peak_amp) in plot_config.peaks.iter() {
        if peak_amp > peak_label_threshold {
            let formatted_peak_amp = if plot_config.y_label.to_lowercase().contains("db")
                || peak_label_format_string_ref.contains("dB")
            {
                format!("{peak_amp:.2} dB")
            } else if peak_amp >= 1_000_000.0 {
                // Use "M" notation for million+ values for better readability
                format!("{:.1}M", peak_amp / 1_000_000.0)
            } else if peak_amp >= 1000.0 {
                // Use "k" notation for thousand+ values for better readability
                format!("{:.1}k", peak_amp / 1000.0)
            } else {
                format!("{peak_amp:.0}")
            };

            // Store frequency, amplitude, and formatted text for later sorting
            peak_labels.push((peak_freq, peak_amp, formatted_peak_amp));
        }
    }

    // Position labels at the bottom of the plot area, with fixed row assignment by peak priority
    if !peak_labels.is_empty() {
        // Sort peaks by amplitude (magnitude) in descending order for proper priority assignment
        // Primary peak (largest amplitude) should be in top row, followed by secondary peaks
        peak_labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        const LABEL_HEIGHT: i32 = 20;
        let mut row_positions: Vec<Vec<(i32, i32)>> = Vec::new();

        // Initialize row tracking for each peak priority level
        for _ in 0..MAX_PEAKS_TO_LABEL {
            row_positions.push(Vec::new());
        }

        for (peak_idx, (peak_freq, _peak_amp, formatted_peak_amp)) in
            peak_labels.into_iter().enumerate()
        {
            if peak_idx >= MAX_PEAKS_TO_LABEL {
                break; // Only label up to MAX_PEAKS_TO_LABEL peaks
            }

            // Create label text based on sorted position (primary = first after sorting)
            let label_text = if peak_idx == 0 {
                format!("▲ Primary Peak: {formatted_peak_amp} at {peak_freq:.0} Hz")
            } else {
                format!("▲ Peak: {formatted_peak_amp} at {peak_freq:.0} Hz")
            };

            // Get the X coordinate of the peak (horizontally aligned)
            let peak_x_pixel = chart
                .backend_coord(&(peak_freq, plot_config.y_range.start))
                .0
                - area_offset.0;
            // Rough per-character estimate (used for overlap probing). We'll compute a
            // tighter width for right-aligned placement (including the triangle) below.
            let label_width_estimate = (label_text.len() as f32 * 8.0) as i32; // Rough estimate

            // --- Exact text measurement helper (uses system TTF if available) ---
            static FONT_DATA_BYTES: OnceLock<Option<&'static [u8]>> = OnceLock::new();

            fn find_system_font_bytes() -> Option<&'static [u8]> {
                // Try common Linux font locations; return first found
                let candidates = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                ];
                for p in candidates.iter() {
                    if Path::new(p).exists() {
                        if let Ok(bytes) = fs::read(p) {
                            // Leak into static for rusttype lifetime
                            let leaked = Box::leak(bytes.into_boxed_slice());
                            return Some(&*leaked);
                        }
                    }
                }
                None
            }

            fn get_system_font_bytes() -> Option<&'static [u8]> {
                *FONT_DATA_BYTES.get_or_init(find_system_font_bytes)
            }

            // Measure text pixel width using rusttype if a system font is available.
            // Font measurement uses system TTF (DejaVu/Liberation/Free Sans) for width calculation,
            // while rendering uses Inconsolata (provided by plotters). Both are monospace with
            // similar metrics, ensuring accurate right-alignment.
            fn measured_text_width_px(text: &str, font_px: f32) -> Option<i32> {
                if let Some(font_bytes) = get_system_font_bytes() {
                    if let Some(font) = rusttype::Font::try_from_bytes(font_bytes) {
                        use rusttype::Scale;
                        let scale = Scale::uniform(font_px);
                        let mut width = 0.0f32;
                        for ch in text.chars() {
                            let glyph = font.glyph(ch).scaled(scale);
                            width += glyph.h_metrics().advance_width;
                        }
                        // Subtract one monospace character width (advance includes trailing space)
                        let avg_char_width = font_px * 0.56;
                        let width_px = (width - avg_char_width).ceil() as i32;
                        return Some(width_px.max(0));
                    }
                }
                None
            }

            // Assign row so that primary is at the top (highest y), secondary below, etc.
            let target_row = MAX_PEAKS_TO_LABEL - 1 - peak_idx;
            let mut horizontal_offset = 0;

            // Find a position in this row, potentially offsetting horizontally to avoid overlap
            loop {
                let label_start = peak_x_pixel + horizontal_offset;
                let label_end = label_start + label_width_estimate;
                let text_y = area_height
                    - ((target_row as i32) * LABEL_HEIGHT)
                    - PEAK_LABEL_BOTTOM_MARGIN_PX;

                // Check if this position overlaps with existing labels in this row
                let mut overlaps = false;
                for &(existing_start, existing_end) in &row_positions[target_row] {
                    if !(label_end < existing_start || label_start > existing_end) {
                        overlaps = true;
                        break;
                    }
                }

                if !overlaps {
                    // Determine if label would go off the right edge of the plot area
                    let plot_right_edge = area_width;
                    let label_fits = label_start + label_width_estimate <= plot_right_edge;

                    // Debug info will be printed after computing force_right_aligned

                    // Use a single triangle_width variable with a tunable ratio
                    const TRIANGLE_WIDTH_RATIO: f32 = 0.5; // Inconsolata triangle ▲ is ~50% of font width
                    let triangle_width =
                        (FONT_SIZE_PEAK_LABEL as f32 * TRIANGLE_WIDTH_RATIO) as i32;
                    let label_text_no_triangle = label_text.trim_start_matches('▲').trim_start();
                    // Force right-aligned logic for peaks in the rightmost 10% of the plot
                    let force_right_aligned =
                        peak_x_pixel > (area_width as f32 * RIGHT_ALIGN_THRESHOLD) as i32;
                    let (draw_text, draw_pos, recorded_start, recorded_end) = if label_fits
                        && !force_right_aligned
                    {
                        // Left-aligned: same behavior as original implementation
                        let left_x = label_start - (triangle_width / 2);
                        (
                            label_text.clone(),
                            (left_x, text_y),
                            // record bounding box as before (label_start,label_end) to preserve overlap behavior
                            label_start,
                            label_end,
                        )
                    } else {
                        // Right-aligned: compute width including the triangle glyph and right-align
                        let right_aligned_text = format!("{} ▲", label_text_no_triangle);
                        // Measure the full right-aligned string width using advance widths
                        let tri_label_width = measured_text_width_px(
                            &right_aligned_text,
                            FONT_SIZE_PEAK_LABEL as f32,
                        )
                        .unwrap_or_else(|| {
                            // fallback to adaptive estimator
                            let mut w = 0.0f32;
                            for ch in right_aligned_text.chars() {
                                w += match ch {
                                    'i' | 'l' | 'I' | 'j' | '\'' | '|' | ':' | '.' | ',' => 0.35,
                                    ' ' => 0.40,
                                    'f' | 't' | 'r' | 'c' | 'k' | 's' => 0.55,
                                    '0'..='9' => 0.6,
                                    'm' | 'w' | 'M' | 'W' => 1.0,
                                    _ => 0.75,
                                };
                            }
                            (w * FONT_SIZE_PEAK_LABEL as f32) as i32
                        });
                        // If the label (including triangle) fits to the left of the peak, right-align it.
                        // Otherwise fall back to left-aligned placement.
                        if tri_label_width + 4 <= peak_x_pixel {
                            // Position text so triangle tip points to peak frequency.
                            // tri_label_width includes the triangle glyph's advance width.
                            // The triangle is left-aligned within its advance width box,
                            // so we need to add the full triangle_width to account for its visual position.
                            let right_x = peak_x_pixel - tri_label_width + (triangle_width);
                            let right_x_clamped = if right_x < 0 { 0 } else { right_x };
                            (
                                right_aligned_text,
                                (right_x_clamped, text_y),
                                right_x_clamped,
                                right_x_clamped + tri_label_width,
                            )
                        } else {
                            // Fallback to left-aligned placement
                            let left_x = label_start - (triangle_width / 2);
                            (label_text.clone(), (left_x, text_y), label_start, label_end)
                        }
                    };

                    area.draw(&Text::new(
                        draw_text.as_str(),
                        draw_pos,
                        ("Inconsolata", FONT_SIZE_PEAK_LABEL)
                            .into_font()
                            .color(&BLACK),
                    ))?;

                    // Record this position
                    row_positions[target_row].push((recorded_start, recorded_end));
                    break;
                } else {
                    // Try offsetting horizontally by the label width
                    horizontal_offset += label_width_estimate;
                }
            }
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
        ("sans-serif", FONT_SIZE_MAIN_TITLE)
            .into_font()
            .color(&BLACK),
    ))?;
    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((3, 1));
    let mut any_axis_plotted = false;

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..crate::axis_names::AXIS_NAMES.len() {
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
                        frequency_ranges: None,
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
        ("sans-serif", FONT_SIZE_MAIN_TITLE)
            .into_font()
            .color(&BLACK),
    ))?;
    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((3, 2));
    let mut any_plot_drawn = false;

    for axis_index in 0..crate::axis_names::AXIS_NAMES.len() {
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
#[allow(clippy::too_many_arguments)]
fn draw_single_heatmap_chart(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    chart_title: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    x_label: &str,
    y_label: &str,
    heatmap_data: &HeatmapData,
    max_db: f64, // Use dynamic max_db instead of hardcoded value
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(chart_title, ("sans-serif", FONT_SIZE_CHART_TITLE))
        .margin(5)
        .x_label_area_size(50) // Increased for more space below horizontal axis label
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(10)
        .y_labels(10)
        .y_label_formatter(&|y| {
            // Format Y-axis labels with "k" notation for large values (spectrum plots)
            // Keep dB values as-is (they're typically small/negative)
            if y.abs() >= 1000.0 && !y_label.contains("dB") {
                format!("{:.0}k", y / 1000.0)
            } else {
                format!("{:.0}", y)
            }
        })
        .light_line_style(WHITE.mix(0.7))
        .label_style(("sans-serif", FONT_SIZE_AXIS_LABEL))
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
                    // Ensure safe_max_db has sufficient span to avoid division by zero
                    let safe_max_db = max_db.max(HEATMAP_MIN_PSD_DB + 1.0);

                    // Clamp psd_db to valid range before color mapping
                    let clamped_psd_db = psd_db.clamp(HEATMAP_MIN_PSD_DB, safe_max_db);

                    let color = map_db_to_color(clamped_psd_db, HEATMAP_MIN_PSD_DB, safe_max_db);

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
        ("sans-serif", FONT_SIZE_MAIN_TITLE)
            .into_font()
            .color(&BLACK),
    ))?;
    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((3, 2));
    let mut any_plot_drawn = false;

    for axis_index in 0..crate::axis_names::AXIS_NAMES.len() {
        let plots_for_axis_option = get_axis_plot_data(axis_index);

        // Compute a single axis_max_db for both unfiltered and filtered plots
        let axis_max_db = if let Some(ref axis_spectrum) = plots_for_axis_option {
            let unfilt_max = axis_spectrum
                .unfiltered
                .as_ref()
                .map(|c| c.max_db)
                .unwrap_or(f64::NEG_INFINITY);
            let filt_max = axis_spectrum
                .filtered
                .as_ref()
                .map(|c| c.max_db)
                .unwrap_or(f64::NEG_INFINITY);
            let computed_max = unfilt_max.max(filt_max);
            // Replace NEG_INFINITY result with a safe floor value
            if computed_max == f64::NEG_INFINITY {
                HEATMAP_MIN_PSD_DB + 1.0
            } else {
                computed_max
            }
        } else {
            HEATMAP_MIN_PSD_DB + 1.0
        };

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
                        axis_max_db,
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
