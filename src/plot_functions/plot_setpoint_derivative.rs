// src/plot_functions/plot_setpoint_derivative.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{COLOR_SETPOINT_DERIVATIVE, LINE_WIDTH_PLOT};
use crate::data_analysis::derivative;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};
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

        // Convert setpoints to f32 for derivative calculation (shared utility expects f32).
        // Minor precision loss (f64→f32→f64) is acceptable for visualization purposes.
        let setpoints_f32: Vec<f32> = setpoints.iter().map(|&sp| sp as f32).collect();

        // Calculate derivatives
        let derivatives = derivative::calculate_derivative(&setpoints_f32, sr);

        if derivatives.is_empty() {
            all_derivatives.push(Vec::new());
            continue;
        }

        // Pair derivative values with time points, applying sanity filters
        let mut derivative_series_data: Vec<(f64, f64)> = Vec::new();
        let mut outlier_count = 0;
        for (idx, (time, deriv)) in times.iter().zip(derivatives.iter()).enumerate() {
            let deriv_f64 = *deriv as f64;

            // Calculate dt (time delta from previous sample) for outlier detection
            let dt = if idx > 0 {
                time - times[idx - 1]
            } else {
                1.0 / sr // Use expected sample period for first sample
            };

            // Filter: skip derivatives from unreasonably small time deltas (logging glitches)
            // or from implausibly large rates (data corruption)
            if dt < crate::constants::SETPOINT_DERIVATIVE_MIN_DT {
                outlier_count += 1;
                continue;
            }
            if deriv_f64.abs() > crate::constants::SETPOINT_DERIVATIVE_OUTLIER_THRESHOLD {
                outlier_count += 1;
                continue;
            }

            derivative_series_data.push((*time, deriv_f64));
        }

        if outlier_count > 0 {
            println!(
                "Warning: Setpoint derivative plot (axis {}) filtered {} outliers (dt<{:.0e} or |deriv|>{:.0})",
                AXIS_NAMES[axis_index],
                outlier_count,
                crate::constants::SETPOINT_DERIVATIVE_MIN_DT,
                crate::constants::SETPOINT_DERIVATIVE_OUTLIER_THRESHOLD
            );
        }

        // Update global ranges (using filtered data only)
        for (time, deriv) in &derivative_series_data {
            time_min = time_min.min(*time);
            time_max = time_max.max(*time);
            global_val_min = global_val_min.min(*deriv);
            global_val_max = global_val_max.max(*deriv);
        }

        all_derivatives.push(derivative_series_data);
    }

    // Determine symmetric half-range using 95th percentile-based scaling
    // This avoids letting extreme maneuvers (acro/freestyle spikes) dominate visualization.
    // Analysis of 146 flight logs shows P95 is typically ~1000 deg/s², allowing 95% of flights
    // to display well without compression from occasional extreme events.
    let static_min = crate::constants::SETPOINT_DERIVATIVE_Y_AXIS_MIN;

    // Collect absolute derivative magnitudes across all axes
    let mut all_abs_vals: Vec<f64> = all_derivatives
        .iter()
        .flat_map(|axis| axis.iter().map(|(_, v)| v.abs()))
        .filter(|v| v.is_finite())
        .collect();

    // Compute p95 if we have enough samples using NaN-safe sorting
    let mut p95_candidate = 0.0_f64;
    if !all_abs_vals.is_empty() {
        all_abs_vals.sort_by(|a, b| a.total_cmp(b));
        let idx = ((all_abs_vals.len() - 1) as f64
            * crate::constants::SETPOINT_DERIVATIVE_EXPANSION_PERCENTILE)
            .floor() as usize;
        p95_candidate = all_abs_vals[idx] * crate::constants::SETPOINT_DERIVATIVE_PERCENTILE_SCALE;
    }

    let global_half = global_val_min.abs().max(global_val_max.abs());

    // Use the maximum of: static_min, p95_candidate, and observed global max (to avoid clipping)
    let mut half_range = static_min.max(p95_candidate).max(global_half);

    // Add headroom for visibility
    half_range *= 1.0 + crate::constants::SETPOINT_DERIVATIVE_Y_AXIS_HEADROOM_FACTOR;

    // Log / annotate whether we are using the static minimum or expanding (and reason)
    if half_range > static_min * 1.01 {
        println!("Note: Setpoint derivative Y-axis expanded to ±{:.0} (p95*1.2={:.0}, observed={:.0}, static_min={:.0})", half_range, p95_candidate, global_half, static_min);
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

        // Include the actual data maximum in the chart title for quick identification
        let format_max = if global_half.abs() >= 1000.0 {
            format!("{:.0}k", global_half / 1000.0)
        } else {
            format!("{:.0}", global_half)
        };

        Some((
            format!(
                "{} Setpoint Derivative (±{})",
                AXIS_NAMES[axis_index], format_max
            ),
            x_range,
            y_range,
            series,
            "Time (s)".to_string(),
            "Rate of Change (deg/s²)".to_string(),
        ))
    })
}
