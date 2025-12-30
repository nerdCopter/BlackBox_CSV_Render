// src/plot_functions/plot_setpoint_derivative.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{COLOR_SETPOINT_DERIVATIVE, LINE_WIDTH_PLOT};
use crate::data_analysis::derivative;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{calculate_range, draw_stacked_plot, PlotSeries};
use crate::types::AllAxisPlotData2Simple;

/// Generates the Stacked Setpoint Derivative (Rate of Change) Plot
///
/// Displays the first derivative (rate of change) of setpoint values
/// for Roll, Pitch, and Yaw axes as time-domain plots.
///
/// # Arguments
/// * `log_data` - Log data containing setpoint values and timestamps
/// * `root_name` - Base filename for output
/// * `sample_rate` - Sampling frequency in Hz (required)
///
/// # Returns
/// Result indicating success or error
pub fn plot_setpoint_derivative(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let sr = sample_rate.ok_or("Setpoint Derivative plot requires sample_rate to be available")?;

    let output_file = format!("{root_name}_SetpointDerivative_stacked.png");
    let plot_type_name = "Setpoint Derivative";

    // Extract time and setpoint data for all 3 axes
    let mut axis_plot_data: AllAxisPlotData2Simple = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..AXIS_NAMES.len() {
                if let Some(setpoint) = row.setpoint[axis_index] {
                    axis_plot_data[axis_index].push((time, setpoint));
                }
            }
        }
    }

    let color_derivative: RGBColor = *COLOR_SETPOINT_DERIVATIVE;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    // Pre-calculate derivatives for all axes to determine common Y-axis range
    let mut all_derivatives: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut global_val_min = f64::INFINITY;
    let mut global_val_max = f64::NEG_INFINITY;
    let mut time_min = f64::INFINITY;
    let mut time_max = f64::NEG_INFINITY;

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..AXIS_NAMES.len() {
        let data = &axis_plot_data[axis_index];
        if data.is_empty() {
            all_derivatives.push(Vec::new());
            continue;
        }

        // Extract time and setpoint values
        let (times, setpoints): (Vec<f64>, Vec<f64>) =
            data.iter().map(|(time, sp)| (*time, *sp)).unzip();

        // Convert setpoints to f32 for derivative calculation
        let setpoints_f32: Vec<f32> = setpoints.iter().map(|&sp| sp as f32).collect();

        // Calculate derivatives
        let derivatives = derivative::calculate_derivative(&setpoints_f32, sr);

        if derivatives.is_empty() {
            all_derivatives.push(Vec::new());
            continue;
        }

        // Pair derivative values with time points
        let derivative_series_data: Vec<(f64, f64)> = times
            .iter()
            .zip(derivatives.iter())
            .map(|(time, deriv)| (*time, *deriv as f64))
            .collect();

        // Update global ranges
        for (time, deriv) in &derivative_series_data {
            time_min = time_min.min(*time);
            time_max = time_max.max(*time);
            global_val_min = global_val_min.min(*deriv);
            global_val_max = global_val_max.max(*deriv);
        }

        all_derivatives.push(derivative_series_data);
    }

    // Calculate common Y-axis range for all axes
    let (common_y_min, common_y_max) = calculate_range(global_val_min, global_val_max);

    // Determine symmetric half-range with a safety constant from constants.rs.
    // Ensure we add a small headroom to make plotting visually less tight.
    let mut half_range = crate::constants::SETPOINT_DERIVATIVE_Y_AXIS_MAX;
    let global_half = common_y_min.abs().max(common_y_max.abs());
    if global_half > half_range {
        // If real data exceeds the chosen static value, expand to cover it to avoid clipping
        half_range = global_half;
    }
    // Add 5% headroom for visibility
    half_range *= 1.05;

    // Log / annotate whether we are using the static minimum or expanding
    if half_range > crate::constants::SETPOINT_DERIVATIVE_Y_AXIS_MAX * 1.01 {
        println!("Note: Setpoint derivative Y-axis expanded to ±{:.0} because data exceeded static limit ±{:.0}", half_range, crate::constants::SETPOINT_DERIVATIVE_Y_AXIS_MAX);
    } else {
        println!("Using static setpoint derivative Y-axis ±{:.0}", half_range);
    }

    draw_stacked_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        let derivative_series_data = &all_derivatives[axis_index];

        if derivative_series_data.is_empty() {
            return None;
        }

        let x_range = time_min..time_max;
        let y_range = -half_range..half_range;

        let series = vec![PlotSeries {
            data: derivative_series_data.clone(),
            label: "d(Setpoint)/dt".to_string(),
            color: color_derivative,
            stroke_width: line_stroke_plot,
        }];

        Some((
            format!("{} Setpoint Derivative", AXIS_NAMES[axis_index]),
            x_range,
            y_range,
            series,
            "Time (s)".to_string(),
            "Rate of Change (deg/s)".to_string(),
        ))
    })
}
