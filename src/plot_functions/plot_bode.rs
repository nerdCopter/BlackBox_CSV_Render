// src/plot_functions/plot_bode.rs

use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::coord::Shift;
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::element::{Circle, Text};
use plotters::prelude::IntoLogRange;
use plotters::series::LineSeries;
use plotters::style::colors::{BLACK, GREEN, RED, WHITE};
use plotters::style::{IntoFont, RGBColor, ShapeStyle};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    FONT_SIZE_CHART_TITLE, FONT_SIZE_LEGEND, FREQUENCY_EPSILON, LINE_WIDTH_PLOT, PLOT_HEIGHT,
    PLOT_WIDTH,
};
use crate::data_analysis::transfer_function_estimation::{
    calculate_stability_margins, estimate_transfer_function_h1, Confidence, StabilityMargins,
    TransferFunctionResult,
};
use crate::data_input::log_data::LogRowData;
use crate::font_config::{FONT_TUPLE_AXIS_LABEL, FONT_TUPLE_CHART_TITLE, FONT_TUPLE_MAIN_TITLE};

/// Minimum coherence threshold for filtering Bode plot data
const MIN_COHERENCE_FOR_PLOT: f64 = 0.1;

/// Plot Bode analysis for all three axes (Roll, Pitch, Yaw)
///
/// Generates a single 3×3 grid plot with magnitude, phase, and coherence for all axes
pub fn plot_bode_analysis(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Bode Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Estimate transfer functions for all three axes
    let mut tf_results = Vec::new();
    let mut margins_results = Vec::new();

    for (axis_index, &axis_name) in AXIS_NAMES.iter().enumerate() {
        // Estimate transfer function
        let tf_result = match estimate_transfer_function_h1(log_data, sr_value, axis_index) {
            Ok(tf) => tf,
            Err(e) => {
                println!("  Skipping Bode plot for {}: {}", axis_name, e);
                continue; // Skip this axis, process remaining axes
            }
        };

        // Calculate stability margins
        let margins = match calculate_stability_margins(&tf_result) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "  Warning: Could not calculate stability margins for {}: {}",
                    axis_name, e
                );
                StabilityMargins::default()
            }
        };

        tf_results.push(tf_result);
        margins_results.push(margins);
    }

    // Early exit if no valid axes
    if tf_results.is_empty() {
        println!("\nINFO: Skipping Bode Plot: No valid transfer function data for any axis.");
        return Ok(());
    }

    // Generate single combined plot filename
    let output_file = format!("{}_Bode_Analysis.png", root_name);

    // Create the 3×3 grid plot
    match create_bode_grid_plot(&output_file, root_name, &tf_results, &margins_results) {
        Ok(_) => println!("  Generated Bode analysis plot: {}", output_file),
        Err(e) => {
            println!("  Error creating Bode plot: {}", e);
            return Err(e);
        }
    }

    // Print warnings for all axes
    for (axis_idx, margins) in margins_results.iter().enumerate() {
        if !margins.warnings.is_empty() {
            let axis_name = &tf_results[axis_idx].axis_name;
            println!("  Bode Analysis Warnings for {}:", axis_name);
            for warning in &margins.warnings {
                println!("    - {}", warning);
            }
        }
    }

    Ok(())
}

/// Create a grid Bode plot (1 to 3 axes × 3 plot types)
fn create_bode_grid_plot(
    output_file: &str,
    root_name: &str,
    tf_results: &[TransferFunctionResult],
    margins_results: &[StabilityMargins],
) -> Result<(), Box<dyn Error>> {
    if tf_results.is_empty() || tf_results.len() > 3 {
        return Err(format!(
            "Expected 1-3 axes for Bode grid plot, got {}",
            tf_results.len()
        )
        .into());
    }
    if tf_results.len() != margins_results.len() {
        return Err("Transfer function and margins results must have same length".into());
    }

    // Validate all transfer functions
    for tf in tf_results {
        if !tf.is_valid() || tf.is_empty() {
            return Err("Invalid or empty transfer function data".into());
        }
    }

    // Create main drawing area with standard plot dimensions
    let root = BitMapBackend::new(output_file, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    // Add main title with log filename
    root.draw(&Text::new(
        root_name,
        (20, 20),
        FONT_TUPLE_MAIN_TITLE.into_font().color(&BLACK),
    ))?;

    // Split into N×3 grid (N = number of axes)
    let areas = root
        .margin(60, 10, 10, 10)
        .split_evenly((tf_results.len(), 3));

    // Draw legend for confidence colors
    draw_confidence_legend(&root, PLOT_WIDTH, PLOT_HEIGHT)?;

    // Determine global frequency range for consistency
    let mut global_freq_min = f64::INFINITY;
    let mut global_freq_max: f64 = 0.0;

    for tf in tf_results {
        let (filtered_freq, _, _, _) = filter_by_coherence(tf, MIN_COHERENCE_FOR_PLOT);
        if !filtered_freq.is_empty() {
            global_freq_min = global_freq_min.min(*filtered_freq.first().unwrap());
            global_freq_max = global_freq_max.max(*filtered_freq.last().unwrap());
        }
    }

    // Guard against all axes having empty filtered data
    if global_freq_min.is_infinite() || global_freq_max == 0.0 {
        println!("\nINFO: Skipping Bode Plot: All axes have insufficient coherence for plotting.");
        return Ok(());
    }

    let freq_min = global_freq_min.max(1.0);
    let freq_max = global_freq_max.min(tf_results[0].sample_rate_hz / 2.0);

    // Plot grid: rows are axes, columns are plot types (Mag, Phase, Coh)
    for axis_index in 0..tf_results.len() {
        let tf = &tf_results[axis_index];
        let margins = &margins_results[axis_index];
        let axis_name = &tf.axis_name;

        // Filter data
        let (filtered_freq, filtered_mag, filtered_phase, filtered_coh) =
            filter_by_coherence(tf, MIN_COHERENCE_FOR_PLOT);

        if filtered_freq.is_empty() {
            continue;
        }

        // Column 0: Magnitude
        let mag_area = &areas[axis_index * 3];
        draw_magnitude_subplot(
            mag_area,
            axis_name,
            &filtered_freq,
            &filtered_mag,
            margins,
            freq_min,
            freq_max,
        )?;

        // Column 1: Phase
        let phase_area = &areas[axis_index * 3 + 1];
        draw_phase_subplot(
            phase_area,
            axis_name,
            &filtered_freq,
            &filtered_phase,
            margins,
            freq_min,
            freq_max,
        )?;

        // Column 2: Coherence
        let coh_area = &areas[axis_index * 3 + 2];
        draw_coherence_subplot(
            coh_area,
            axis_name,
            &filtered_freq,
            &filtered_coh,
            freq_min,
            freq_max,
        )?;
    }

    root.present()?;
    Ok(())
}

/// Filter transfer function data by coherence threshold
fn filter_by_coherence(
    tf: &TransferFunctionResult,
    min_coherence: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut freq = Vec::new();
    let mut mag = Vec::new();
    let mut phase = Vec::new();
    let mut coh = Vec::new();

    for i in 0..tf.len() {
        if tf.coherence[i] >= min_coherence {
            freq.push(tf.frequency_hz[i]);
            mag.push(tf.magnitude_db[i]);
            phase.push(tf.phase_deg[i]);
            coh.push(tf.coherence[i]);
        }
    }

    (freq, mag, phase, coh)
}

/// Draw magnitude subplot with axis label
fn draw_magnitude_subplot(
    area: &DrawingArea<BitMapBackend, Shift>,
    axis_name: &str,
    freq: &[f64],
    mag: &[f64],
    margins: &StabilityMargins,
    freq_min: f64,
    freq_max: f64,
) -> Result<(), Box<dyn Error>> {
    if freq.is_empty() || mag.is_empty() {
        return Err("Empty data for magnitude plot".into());
    }

    // Determine magnitude range
    let mag_min = mag.iter().copied().fold(f64::INFINITY, f64::min);
    let mag_max = mag.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mag_range_min = (mag_min - 10.0).floor();
    let mag_range_max = (mag_max + 10.0).ceil();

    let mut chart = ChartBuilder::on(area)
        .caption(
            format!("Bode Analysis - {} Magnitude", axis_name),
            FONT_TUPLE_CHART_TITLE.into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (freq_min..freq_max).log_scale(),
            mag_range_min..mag_range_max,
        )?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude (dB)")
        .x_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .y_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .draw()?;

    // Draw 0 dB reference line
    let zero_db_data: Vec<(f64, f64)> = vec![(freq_min, 0.0), (freq_max, 0.0)];
    chart.draw_series(LineSeries::new(
        zero_db_data,
        ShapeStyle::from(&BLACK).stroke_width(1),
    ))?;

    // Draw magnitude curve
    let mag_data: Vec<(f64, f64)> = freq.iter().copied().zip(mag.iter().copied()).collect();
    chart.draw_series(LineSeries::new(
        mag_data,
        ShapeStyle::from(&RGBColor(0, 100, 200)).stroke_width(LINE_WIDTH_PLOT),
    ))?;

    // Mark gain crossover
    if let Some(f_c) = margins.gain_crossover_hz {
        if f_c >= freq_min && f_c <= freq_max {
            let color = confidence_color(margins.confidence);

            // Vertical line
            chart.draw_series(LineSeries::new(
                vec![(f_c, mag_range_min), (f_c, mag_range_max)],
                ShapeStyle::from(&color).stroke_width(2),
            ))?;

            // Marker on curve
            chart.draw_series(std::iter::once(Circle::new(
                (f_c, 0.0),
                5,
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    Ok(())
}

/// Draw phase subplot with axis label
fn draw_phase_subplot(
    area: &DrawingArea<BitMapBackend, Shift>,
    axis_name: &str,
    freq: &[f64],
    phase: &[f64],
    margins: &StabilityMargins,
    freq_min: f64,
    freq_max: f64,
) -> Result<(), Box<dyn Error>> {
    if freq.is_empty() || phase.is_empty() {
        return Err("Empty data for phase plot".into());
    }

    // Determine phase range
    let phase_min = phase.iter().copied().fold(f64::INFINITY, f64::min);
    let phase_max = phase.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let phase_range_min = (phase_min - 30.0).floor();
    let phase_range_max = (phase_max + 30.0).ceil();

    let mut chart = ChartBuilder::on(area)
        .caption(
            format!("Bode Analysis - {} Phase", axis_name),
            FONT_TUPLE_CHART_TITLE.into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (freq_min..freq_max).log_scale(),
            phase_range_min..phase_range_max,
        )?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Phase (degrees)")
        .x_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .y_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .draw()?;

    // Draw -180° reference line
    let minus_180_data: Vec<(f64, f64)> = vec![(freq_min, -180.0), (freq_max, -180.0)];
    chart.draw_series(LineSeries::new(
        minus_180_data,
        ShapeStyle::from(&BLACK).stroke_width(1),
    ))?;

    // Draw phase curve
    let phase_data: Vec<(f64, f64)> = freq.iter().copied().zip(phase.iter().copied()).collect();
    chart.draw_series(LineSeries::new(
        phase_data,
        ShapeStyle::from(&RGBColor(200, 0, 100)).stroke_width(LINE_WIDTH_PLOT),
    ))?;

    // Mark phase crossover
    if let Some(f_p) = margins.phase_crossover_hz {
        if f_p >= freq_min && f_p <= freq_max {
            let color = confidence_color(margins.confidence);

            // Vertical line
            chart.draw_series(LineSeries::new(
                vec![(f_p, phase_range_min), (f_p, phase_range_max)],
                ShapeStyle::from(&color).stroke_width(2),
            ))?;

            // Marker on curve (need to interpolate phase at f_p)
            if let Some(phase_at_fp) = interpolate(freq, phase, f_p) {
                chart.draw_series(std::iter::once(Circle::new(
                    (f_p, phase_at_fp),
                    5,
                    ShapeStyle::from(&color).filled(),
                )))?;
            }
        }
    }

    Ok(())
}

/// Draw coherence subplot with axis label
fn draw_coherence_subplot(
    area: &DrawingArea<BitMapBackend, Shift>,
    axis_name: &str,
    freq: &[f64],
    coh: &[f64],
    freq_min: f64,
    freq_max: f64,
) -> Result<(), Box<dyn Error>> {
    if freq.is_empty() || coh.is_empty() {
        return Err("Empty data for coherence plot".into());
    }

    let mut chart = ChartBuilder::on(area)
        .caption(
            format!("Bode Analysis - {} Coherence", axis_name),
            FONT_TUPLE_CHART_TITLE.into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((freq_min..freq_max).log_scale(), 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Coherence")
        .x_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .y_label_style(FONT_TUPLE_AXIS_LABEL.into_font())
        .draw()?;

    // Draw 0.5 reference line (medium quality threshold)
    let half_coh_data: Vec<(f64, f64)> = vec![(freq_min, 0.5), (freq_max, 0.5)];
    chart.draw_series(LineSeries::new(
        half_coh_data,
        ShapeStyle::from(&BLACK).stroke_width(1),
    ))?;

    // Draw coherence curve
    let coh_data: Vec<(f64, f64)> = freq.iter().copied().zip(coh.iter().copied()).collect();
    chart.draw_series(LineSeries::new(
        coh_data,
        ShapeStyle::from(&RGBColor(100, 150, 0)).stroke_width(LINE_WIDTH_PLOT),
    ))?;

    Ok(())
}

/// Get color based on confidence level
fn confidence_color(conf: Confidence) -> RGBColor {
    match conf {
        Confidence::High => GREEN,
        Confidence::Medium => RGBColor(255, 140, 0), // Dark orange - more visible than yellow
        Confidence::Low => RED,
    }
}

/// Draw legend explaining confidence color coding
fn draw_confidence_legend(
    root: &DrawingArea<BitMapBackend, Shift>,
    width: u32,
    _height: u32,
) -> Result<(), Box<dyn Error>> {
    let legend_x = width as i32 - 220;
    let legend_y = 40;
    let box_size = 12;
    let spacing = 25;

    // Legend background (white rectangle with border)
    root.draw(&plotters::element::Rectangle::new(
        [
            (legend_x - 10, legend_y - 10),
            (width as i32 - 10, legend_y + 95),
        ],
        plotters::style::ShapeStyle::from(&BLACK)
            .stroke_width(1)
            .filled(),
    ))?;

    root.draw(&plotters::element::Rectangle::new(
        [
            (legend_x - 8, legend_y - 8),
            (width as i32 - 12, legend_y + 93),
        ],
        plotters::style::ShapeStyle::from(&WHITE)
            .stroke_width(0)
            .filled(),
    ))?;

    // Title
    root.draw(&Text::new(
        "Confidence Level",
        (legend_x, legend_y + 5),
        ("sans-serif", FONT_SIZE_CHART_TITLE)
            .into_font()
            .color(&BLACK),
    ))?;

    // High confidence (green)
    root.draw(&plotters::element::Rectangle::new(
        [
            (legend_x, legend_y + 20),
            (legend_x + box_size, legend_y + 20 + box_size),
        ],
        plotters::style::ShapeStyle::from(&GREEN).filled(),
    ))?;
    root.draw(&Text::new(
        "High (>0.7)",
        (legend_x + 20, legend_y + 30),
        ("sans-serif", FONT_SIZE_LEGEND).into_font().color(&BLACK),
    ))?;

    // Medium confidence (dark orange)
    root.draw(&plotters::element::Rectangle::new(
        [
            (legend_x, legend_y + 20 + spacing),
            (legend_x + box_size, legend_y + 20 + spacing + box_size),
        ],
        plotters::style::ShapeStyle::from(&RGBColor(255, 140, 0)).filled(),
    ))?;
    root.draw(&Text::new(
        "Medium (0.4-0.7)",
        (legend_x + 20, legend_y + 20 + spacing + 10),
        ("sans-serif", FONT_SIZE_LEGEND).into_font().color(&BLACK),
    ))?;

    // Low confidence (red)
    root.draw(&plotters::element::Rectangle::new(
        [
            (legend_x, legend_y + 20 + 2 * spacing),
            (legend_x + box_size, legend_y + 20 + 2 * spacing + box_size),
        ],
        plotters::style::ShapeStyle::from(&RED).filled(),
    ))?;
    root.draw(&Text::new(
        "Low (<0.4)",
        (legend_x + 20, legend_y + 20 + 2 * spacing + 10),
        ("sans-serif", FONT_SIZE_LEGEND).into_font().color(&BLACK),
    ))?;

    Ok(())
}

/// Linear interpolation helper
///
/// # Panics
/// In debug builds, panics if `x` is not sorted in ascending order.
/// In release builds, behavior is undefined if `x` is not sorted.
fn interpolate(x: &[f64], y: &[f64], x_target: f64) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    // Validate x is sorted (debug builds only)
    debug_assert!(
        x.windows(2).all(|w| w[0] <= w[1]),
        "interpolate requires sorted x array"
    );

    // Find bracketing indices
    let mut idx = 0;
    while idx < x.len() - 1 && x[idx + 1] < x_target {
        idx += 1;
    }

    // Clamp to bounds
    if x_target < x[0] {
        return Some(y[0]);
    }
    if idx >= x.len() - 1 {
        return Some(y[y.len() - 1]);
    }

    // Linear interpolation
    let x1 = x[idx];
    let x2 = x[idx + 1];
    let y1 = y[idx];
    let y2 = y[idx + 1];

    // Guard against duplicate x values
    if (x2 - x1).abs() < FREQUENCY_EPSILON {
        return Some(y1);
    }

    let t = (x_target - x1) / (x2 - x1);
    Some(y1 + t * (y2 - y1))
}
