// src/plot_functions/plot_bode.rs

use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::coord::Shift;
use plotters::drawing::{DrawingArea, IntoDrawingArea};
use plotters::element::{Circle, Text};
use plotters::prelude::IntoLogRange;
use plotters::series::LineSeries;
use plotters::style::colors::{BLACK, GREEN, RED, WHITE, YELLOW};
use plotters::style::{IntoFont, RGBColor, ShapeStyle};
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{LINE_WIDTH_PLOT, PLOT_HEIGHT, PLOT_WIDTH};
use crate::data_analysis::transfer_function_estimation::{
    calculate_stability_margins, estimate_transfer_function_h1, Confidence, StabilityMargins,
    TransferFunctionResult,
};
use crate::data_input::log_data::LogRowData;
use crate::font_config::{FONT_TUPLE_AXIS_LABEL, FONT_TUPLE_CHART_TITLE, FONT_TUPLE_MAIN_TITLE};

/// Plot Bode analysis for all three axes (Roll, Pitch, Yaw)
///
/// Generates transfer function plots with magnitude, phase, and coherence subplots
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

    // Generate Bode plots for each axis
    for (axis_index, &axis_name) in AXIS_NAMES.iter().enumerate() {
        // Estimate transfer function
        let tf_result = match estimate_transfer_function_h1(log_data, sr_value, axis_index) {
            Ok(tf) => tf,
            Err(e) => {
                println!("  Skipping Bode plot for {}: {}", axis_name, e);
                continue;
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

        // Generate plot filename
        let output_file = format!("{}_bode_{}.png", root_name, axis_name.to_lowercase());

        // Create the plot
        match create_bode_plot(&output_file, &tf_result, &margins) {
            Ok(_) => println!("  Generated Bode plot: {}", output_file),
            Err(e) => println!("  Error creating Bode plot for {}: {}", axis_name, e),
        }

        // Print warnings if any
        if !margins.warnings.is_empty() {
            println!("  Bode Analysis Warnings for {}:", axis_name);
            for warning in &margins.warnings {
                println!("    - {}", warning);
            }
        }
    }

    Ok(())
}

/// Create a three-row Bode plot with magnitude, phase, and coherence
fn create_bode_plot(
    output_file: &str,
    tf: &TransferFunctionResult,
    margins: &StabilityMargins,
) -> Result<(), Box<dyn Error>> {
    if !tf.is_valid() || tf.is_empty() {
        return Err("Invalid or empty transfer function data".into());
    }

    // Create main drawing area
    let root = BitMapBackend::new(output_file, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create title with stability margins
    let title = format_title(&tf.axis_name, margins);
    root.draw(&Text::new(
        title,
        (10, 10),
        FONT_TUPLE_MAIN_TITLE.into_font().color(&BLACK),
    ))?;

    // Split into three rows
    let areas = root.margin(50, 5, 5, 5).split_evenly((3, 1));

    // Filter data to coherence > 0.1 regions (very permissive - coherence plot shows quality)
    let (filtered_freq, filtered_mag, filtered_phase, filtered_coh) = filter_by_coherence(tf, 0.1);

    if filtered_freq.is_empty() {
        return Err("No data with coherence > 0.1".into());
    }

    // Determine frequency range (logarithmic, 1 Hz to Nyquist frequency)
    let nyquist = tf.sample_rate_hz / 2.0;
    let freq_min = 1.0_f64.max(*filtered_freq.first().unwrap_or(&1.0));
    let freq_max = nyquist.min(*filtered_freq.last().unwrap_or(&100.0));

    // Plot 1: Magnitude (dB)
    draw_magnitude_plot(
        &areas[0],
        &filtered_freq,
        &filtered_mag,
        margins,
        freq_min,
        freq_max,
    )?;

    // Plot 2: Phase (degrees)
    draw_phase_plot(
        &areas[1],
        &filtered_freq,
        &filtered_phase,
        margins,
        freq_min,
        freq_max,
    )?;

    // Plot 3: Coherence (0-1)
    draw_coherence_plot(&areas[2], &filtered_freq, &filtered_coh, freq_min, freq_max)?;

    root.present()?;
    Ok(())
}

/// Format plot title with stability margin information
fn format_title(axis_name: &str, margins: &StabilityMargins) -> String {
    let pm_str = if let Some(pm) = margins.phase_margin_deg {
        format!("PM: {:.1}°", pm)
    } else {
        "PM: N/A".to_string()
    };

    let gm_str = if let Some(gm) = margins.gain_margin_db {
        format!("GM: {:.1} dB", gm)
    } else {
        "GM: N/A".to_string()
    };

    format!("Bode Plot - {} ({}, {})", axis_name, pm_str, gm_str)
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

/// Draw magnitude subplot
fn draw_magnitude_plot(
    area: &DrawingArea<BitMapBackend, Shift>,
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
        .caption("Magnitude", FONT_TUPLE_CHART_TITLE.into_font())
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

/// Draw phase subplot
fn draw_phase_plot(
    area: &DrawingArea<BitMapBackend, Shift>,
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
        .caption("Phase", FONT_TUPLE_CHART_TITLE.into_font())
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

/// Draw coherence subplot
fn draw_coherence_plot(
    area: &DrawingArea<BitMapBackend, Shift>,
    freq: &[f64],
    coh: &[f64],
    freq_min: f64,
    freq_max: f64,
) -> Result<(), Box<dyn Error>> {
    if freq.is_empty() || coh.is_empty() {
        return Err("Empty data for coherence plot".into());
    }

    let mut chart = ChartBuilder::on(area)
        .caption("Coherence", FONT_TUPLE_CHART_TITLE.into_font())
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
        Confidence::Medium => YELLOW,
        Confidence::Low => RED,
    }
}

/// Linear interpolation helper
fn interpolate(x: &[f64], y: &[f64], x_target: f64) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    // Find bracketing indices
    let mut idx = 0;
    while idx < x.len() - 1 && x[idx + 1] < x_target {
        idx += 1;
    }

    if idx >= x.len() - 1 {
        return Some(y[y.len() - 1]);
    }

    // Linear interpolation
    let x1 = x[idx];
    let x2 = x[idx + 1];
    let y1 = y[idx];
    let y2 = y[idx + 1];

    let t = (x_target - x1) / (x2 - x1);
    Some(y1 + t * (y2 - y1))
}
