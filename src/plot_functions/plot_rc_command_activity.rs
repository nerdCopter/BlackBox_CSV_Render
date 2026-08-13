// src/plot_functions/plot_rc_command_activity.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::types::{AllAxisPlotData2, AxisPlotData2};

use crate::constants::{
    COLOR_RC_COMMAND, COLOR_SETPOINT_MAIN, LINE_WIDTH_PLOT, RC_COMMAND_ACTIVITY_Y_AXIS_MIN,
    RC_STEP_BLOCKY_MEDIAN_PLATEAU_MS, RC_STEP_MIN_COUNT_FOR_ASSESSMENT, RC_STEP_MIN_JUMP_SIZE,
};
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};

/// Per-axis result of the RC Command step (stick-input smoothness) detection.
pub struct RcCommandStepResult {
    pub axis_name: String,
    pub step_count: usize,
    pub median_plateau_ms: Option<f64>,
    pub is_blocky: bool,
}

/// Detects discrete quantization steps in an RC Command time series by measuring the
/// median "plateau" duration between value changes: a smoothly-interpolated signal
/// changes almost every sample (short plateaus), while raw/unsmoothed RX-link input is
/// held flat for the RX update interval (long plateaus) — the signature of a visible
/// staircase in RC Command, and of jitter leaking into the Setpoint response.
fn detect_rc_command_steps(data: &AxisPlotData2, sample_rate: Option<f64>) -> RcCommandStepResult {
    let rc_points: Vec<(f64, f64)> = data
        .iter()
        .filter_map(|(t, _, rc)| rc.map(|r| (*t, r)))
        .collect();

    if rc_points.len() < 2 {
        return RcCommandStepResult {
            axis_name: String::new(),
            step_count: 0,
            median_plateau_ms: None,
            is_blocky: false,
        };
    }

    let mut plateau_samples: Vec<usize> = Vec::new();
    let mut plateau_len = 1usize;
    // Whether the transition that started the *current* run was itself a qualifying jump.
    let mut current_run_follows_qualifying_jump = false;

    for i in 1..rc_points.len() {
        let prev = rc_points[i - 1].1;
        let curr = rc_points[i].1;
        if (curr - prev).abs() < f64::EPSILON {
            plateau_len += 1;
        } else {
            let qualifies = (curr - prev).abs() >= RC_STEP_MIN_JUMP_SIZE;
            if qualifies {
                plateau_samples.push(plateau_len);
            }
            plateau_len = 1;
            current_run_follows_qualifying_jump = qualifies;
        }
    }
    // Include the trailing run, but only if the transition that started it was itself a
    // qualifying jump — a plateau following sub-threshold float noise isn't a real "held"
    // measurement, and a fully static series (no rc_command movement at all) must stay
    // unclassified (None), not "maximally blocky" from one giant plateau spanning the whole log.
    if current_run_follows_qualifying_jump {
        plateau_samples.push(plateau_len);
    }

    let step_count = plateau_samples.len();
    let median_plateau_ms = if step_count < RC_STEP_MIN_COUNT_FOR_ASSESSMENT {
        None
    } else {
        plateau_samples.sort_unstable();
        let mid = plateau_samples.len() / 2;
        let median_samples = if plateau_samples.len() % 2 == 0 {
            (plateau_samples[mid - 1] + plateau_samples[mid]) as f64 / 2.0
        } else {
            plateau_samples[mid] as f64
        };
        sample_rate
            .filter(|sr| *sr > 0.0)
            .map(|sr| (median_samples / sr) * 1000.0)
    };

    let is_blocky = median_plateau_ms
        .map(|v| v >= RC_STEP_BLOCKY_MEDIAN_PLATEAU_MS)
        .unwrap_or(false);

    RcCommandStepResult {
        axis_name: String::new(),
        step_count,
        median_plateau_ms,
        is_blocky,
    }
}

/// Generates the Stacked Setpoint vs RC Command Plot per axis (Roll, Pitch, Yaw).
///
/// Correlates stepped/blocky RC stick input against the resulting Setpoint, to visually
/// diagnose unfiltered or unsmoothed stick input. Also returns a per-axis step-detection
/// summary for the markdown report.
pub fn plot_rc_command_activity(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<Vec<RcCommandStepResult>, Box<dyn Error>> {
    let output_file_rc_command_activity = format!("{root_name}_RC_Command_Activity_stacked.png");
    let plot_type_name = "RC Command Activity";

    let mut axis_plot_data: AllAxisPlotData2 = Default::default();

    // Ensure AXIS_NAMES length matches the data array length to prevent out-of-bounds access
    if AXIS_NAMES.len() != axis_plot_data.len() {
        return Err(format!(
            "AXIS_NAMES length ({}) does not match axis_plot_data length ({})",
            AXIS_NAMES.len(),
            axis_plot_data.len()
        )
        .into());
    }

    // Collect Setpoint and RC Command data for each axis from log rows
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..axis_plot_data.len() {
                let setpoint = row.setpoint[axis_index];
                let rc_command = row.rc_command[axis_index];

                // Only add if at least one value exists
                if setpoint.is_some() || rc_command.is_some() {
                    axis_plot_data[axis_index].push((time, setpoint, rc_command));
                }
            }
        }
    }

    // Step-detection summary per axis, computed before draw_stacked_plot consumes axis_plot_data.
    let step_results: Vec<RcCommandStepResult> = (0..axis_plot_data.len())
        .map(|axis_index| {
            let mut result = detect_rc_command_steps(&axis_plot_data[axis_index], sample_rate);
            result.axis_name = if axis_index < AXIS_NAMES.len() {
                AXIS_NAMES[axis_index].to_string()
            } else {
                format!("Axis {axis_index}")
            };
            result
        })
        .collect();

    let color_setpoint: RGBColor = *COLOR_SETPOINT_MAIN;
    let color_rc_command: RGBColor = *COLOR_RC_COMMAND;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    // Pre-calculate percentile-based range across ALL axes for unified Y-axis scaling
    // Use 5th and 95th percentiles to ignore extreme outliers (crashes, hard landings)
    // that would compress normal flight data into an unreadable spectrum
    let mut all_values: Vec<f64> = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..axis_plot_data.len() {
        let data = &axis_plot_data[axis_index];
        for (_, setpoint, rc_command) in data {
            if let Some(s) = setpoint {
                all_values.push(*s);
            }
            if let Some(r) = rc_command {
                all_values.push(*r);
            }
        }
    }

    // Calculate 5th and 95th percentiles (removes 5% outliers - keeps 90% of data)
    let (global_val_min, global_val_max) = if all_values.is_empty() {
        (
            -RC_COMMAND_ACTIVITY_Y_AXIS_MIN,
            RC_COMMAND_ACTIVITY_Y_AXIS_MIN,
        )
    } else {
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = all_values.len();
        let p5_idx = (len as f64 * 0.05).floor() as usize;
        let p95_idx = (len as f64 * 0.95).ceil().min(len as f64 - 1.0) as usize;
        (all_values[p5_idx], all_values[p95_idx])
    };

    // Determine symmetric half-range with minimum scale
    let global_half = global_val_min.abs().max(global_val_max.abs());
    let half_range = global_half.max(RC_COMMAND_ACTIVITY_Y_AXIS_MIN);

    draw_stacked_plot(
        &output_file_rc_command_activity,
        root_name,
        plot_type_name,
        move |axis_index| {
            let data = &axis_plot_data[axis_index];
            if data.is_empty() {
                return None;
            }

            let mut setpoint_series_data: Vec<(f64, f64)> = Vec::new();
            let mut rc_command_series_data: Vec<(f64, f64)> = Vec::new();

            let mut time_min = f64::INFINITY;
            let mut time_max = f64::NEG_INFINITY;

            for (time, setpoint, rc_command) in data {
                time_min = time_min.min(*time);
                time_max = time_max.max(*time);

                if let Some(s) = setpoint {
                    setpoint_series_data.push((*time, *s));
                }
                if let Some(r) = rc_command {
                    rc_command_series_data.push((*time, *r));
                }
            }

            if setpoint_series_data.is_empty() && rc_command_series_data.is_empty() {
                return None;
            }

            // Use unified symmetric Y-axis range across all axes
            let x_range = time_min..time_max;
            let y_range = -half_range..half_range;

            let mut series = Vec::new();

            // Draw RC Command first (behind) — it is a stepped/staircase reference trace;
            // Setpoint is drawn last (on top) to stay visible where the two overlap.
            if !rc_command_series_data.is_empty() {
                series.push(PlotSeries {
                    data: rc_command_series_data,
                    label: "RC Command (stick position)".to_string(),
                    color: color_rc_command,
                    stroke_width: line_stroke_plot,
                });
            }
            if !setpoint_series_data.is_empty() {
                series.push(PlotSeries {
                    data: setpoint_series_data,
                    label: "Setpoint".to_string(),
                    color: color_setpoint,
                    stroke_width: line_stroke_plot,
                });
            }

            Some((
                {
                    if axis_index < AXIS_NAMES.len() {
                        format!("{} Setpoint vs RC Command", AXIS_NAMES[axis_index])
                    } else {
                        format!("Axis {} Setpoint vs RC Command", axis_index)
                    }
                },
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Value".to_string(),
            ))
        },
    )?;

    Ok(step_results)
}

// src/plot_functions/plot_rc_command_activity.rs
