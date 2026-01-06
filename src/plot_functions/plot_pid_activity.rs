// src/plot_functions/plot_pid_activity.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::types::AllAxisPlotData3;

use crate::constants::{
    COLOR_D_TERM_ACTIVITY, COLOR_I_TERM, COLOR_P_TERM, LINE_WIDTH_PLOT, PID_ACTIVITY_Y_AXIS_MIN,
};
use crate::data_input::log_data::LogRowData;
use crate::data_input::pid_metadata::parse_pid_metadata;
use crate::plot_framework::{draw_stacked_plot, PlotSeries, CUTOFF_LINE_DOTTED_PREFIX};

/// Generates the Stacked P, I, D Term Activity Plot showing all three PID terms over time
pub fn plot_pid_activity(
    log_data: &[LogRowData],
    root_name: &str,
    header_metadata: Option<&[(String, String)]>,
) -> Result<(), Box<dyn Error>> {
    let output_file_pid_activity = format!("{root_name}_PID_Activity_stacked.png");
    let plot_type_name = "P, I, D Activity";

    let mut axis_plot_data: AllAxisPlotData3 = Default::default();

    // Ensure AXIS_NAMES length matches the data array length to prevent out-of-bounds access
    if AXIS_NAMES.len() != axis_plot_data.len() {
        return Err(format!(
            "AXIS_NAMES length ({}) does not match axis_plot_data length ({})",
            AXIS_NAMES.len(),
            axis_plot_data.len()
        )
        .into());
    }

    // Collect P, I, D term data for each axis from log rows
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..axis_plot_data.len() {
                let p_term = row.p_term[axis_index];
                let i_term = row.i_term[axis_index];
                let d_term = row.d_term[axis_index];

                // Only add if at least one term exists
                if p_term.is_some() || i_term.is_some() || d_term.is_some() {
                    axis_plot_data[axis_index].push((time, p_term, i_term, d_term));
                }
            }
        }
    }

    let color_p_term: RGBColor = *COLOR_P_TERM;
    let color_i_term: RGBColor = *COLOR_I_TERM;
    let color_d_term: RGBColor = *COLOR_D_TERM_ACTIVITY;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    // Pre-calculate min/max across ALL axes for unified Y-axis scaling
    let mut global_val_min = f64::INFINITY;
    let mut global_val_max = f64::NEG_INFINITY;

    // Include I-term saturation limits in the calculation
    let iterm_limit = 400.0;

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..axis_plot_data.len() {
        let data = &axis_plot_data[axis_index];
        for (_, p_term, i_term, d_term) in data {
            if let Some(p) = p_term {
                global_val_min = global_val_min.min(*p);
                global_val_max = global_val_max.max(*p);
            }
            if let Some(i) = i_term {
                global_val_min = global_val_min.min(*i);
                global_val_max = global_val_max.max(*i);
            }
            if let Some(d) = d_term {
                global_val_min = global_val_min.min(*d);
                global_val_max = global_val_max.max(*d);
            }
        }
    }

    // Include I-term saturation limits in the range
    global_val_min = global_val_min.min(-iterm_limit);
    global_val_max = global_val_max.max(iterm_limit);

    // Determine symmetric half-range with minimum scale
    let global_half = global_val_min.abs().max(global_val_max.abs());
    let half_range = global_half.max(PID_ACTIVITY_Y_AXIS_MIN);

    // Parse PID metadata for optional display in titles
    let pid_data = header_metadata.map(parse_pid_metadata);

    draw_stacked_plot(
        &output_file_pid_activity,
        root_name,
        plot_type_name,
        move |axis_index| {
            let data = &axis_plot_data[axis_index];
            if data.is_empty() {
                return None;
            }

            let mut p_term_series_data: Vec<(f64, f64)> = Vec::new();
            let mut i_term_series_data: Vec<(f64, f64)> = Vec::new();
            let mut d_term_series_data: Vec<(f64, f64)> = Vec::new();

            let mut time_min = f64::INFINITY;
            let mut time_max = f64::NEG_INFINITY;

            // Track min/max/avg for each term for legend display
            // Average is included because:
            // - P-term avg: Shows net proportional correction (persistent offset = tuning issue)
            // - I-term avg: Shows integrator wind direction (fighting disturbance = bias or wind)
            // - D-term avg: Should stay ~0 (non-zero avg = lag or phase shift issues)
            // Min/max show actual operating range; avg contextualizes the data trend

            // Track min/max for each term for legend display
            let mut p_min = f64::INFINITY;
            let mut p_max = f64::NEG_INFINITY;
            let mut p_sum = 0.0;
            let mut p_count = 0;

            let mut i_min = f64::INFINITY;
            let mut i_max = f64::NEG_INFINITY;
            let mut i_sum = 0.0;
            let mut i_count = 0;

            let mut d_min = f64::INFINITY;
            let mut d_max = f64::NEG_INFINITY;
            let mut d_sum = 0.0;
            let mut d_count = 0;

            // Collect P, I, D term data and track time range, min/max/sum
            for (time, p_term, i_term, d_term) in data {
                time_min = time_min.min(*time);
                time_max = time_max.max(*time);

                if let Some(p) = p_term {
                    p_term_series_data.push((*time, *p));
                    p_min = p_min.min(*p);
                    p_max = p_max.max(*p);
                    p_sum += p;
                    p_count += 1;
                }
                if let Some(i) = i_term {
                    i_term_series_data.push((*time, *i));
                    i_min = i_min.min(*i);
                    i_max = i_max.max(*i);
                    i_sum += i;
                    i_count += 1;
                }
                if let Some(d) = d_term {
                    d_term_series_data.push((*time, *d));
                    d_min = d_min.min(*d);
                    d_max = d_max.max(*d);
                    d_sum += d;
                    d_count += 1;
                }
            }

            if p_term_series_data.is_empty()
                && i_term_series_data.is_empty()
                && d_term_series_data.is_empty()
            {
                return None;
            }

            // Create constant-value series for I-term saturation limits
            let mut limit_pos_series_data: Vec<(f64, f64)> = Vec::new();
            let mut limit_neg_series_data: Vec<(f64, f64)> = Vec::new();

            limit_pos_series_data.push((time_min, iterm_limit));
            limit_pos_series_data.push((time_max, iterm_limit));

            limit_neg_series_data.push((time_min, -iterm_limit));
            limit_neg_series_data.push((time_max, -iterm_limit));

            // Use unified symmetric Y-axis range across all axes
            let x_range = time_min..time_max;
            let y_range = -half_range..half_range;

            let mut series = Vec::new();

            // Add D-term data series first (drawn first = behind)
            if !d_term_series_data.is_empty() {
                let d_avg = if d_count > 0 {
                    d_sum / d_count as f64
                } else {
                    0.0
                };
                // Calculate lag ratio: |avg| / |max| shows phase shift tendency
                let d_range = (d_max - d_min).abs();
                let d_lag_ratio = if d_range.abs() > 1e-6 {
                    (d_avg.abs() / d_range) * 100.0
                } else {
                    0.0
                };
                series.push(PlotSeries {
                    data: d_term_series_data,
                    label: format!(
                        "D-term (Derivative): min={:.0}, avg={:.1}, max={:.0} (lag ratio: {:.1}%)",
                        d_min, d_avg, d_max, d_lag_ratio
                    ),
                    color: color_d_term,
                    stroke_width: line_stroke_plot,
                });
            }

            // Add P-term data series (drawn second)
            if !p_term_series_data.is_empty() {
                let p_avg = if p_count > 0 {
                    p_sum / p_count as f64
                } else {
                    0.0
                };
                series.push(PlotSeries {
                    data: p_term_series_data,
                    label: format!(
                        "P-term (Proportional): min={:.0}, avg={:.1}, max={:.0}",
                        p_min, p_avg, p_max
                    ),
                    color: color_p_term,
                    stroke_width: line_stroke_plot,
                });
            }

            // Add I-term data series last (drawn last = on top, most visible)
            if !i_term_series_data.is_empty() {
                let i_avg = if i_count > 0 {
                    i_sum / i_count as f64
                } else {
                    0.0
                };
                series.push(PlotSeries {
                    data: i_term_series_data,
                    label: format!(
                        "I-term (Integral): min={:.0}, avg={:.1}, max={:.0}",
                        i_min, i_avg, i_max
                    ),
                    color: color_i_term,
                    stroke_width: line_stroke_plot,
                });
            }

            // Add I-term saturation reference lines (dashed)
            series.push(PlotSeries {
                data: limit_pos_series_data,
                label: format!(
                    "{}I-term Limit (+{})",
                    CUTOFF_LINE_DOTTED_PREFIX, iterm_limit as i32
                ),
                color: RGBColor(200, 0, 0),
                stroke_width: 2,
            });

            series.push(PlotSeries {
                data: limit_neg_series_data,
                label: format!(
                    "{}I-term Limit (-{})",
                    CUTOFF_LINE_DOTTED_PREFIX, iterm_limit as i32
                ),
                color: RGBColor(200, 0, 0),
                stroke_width: 2,
            });

            Some((
                {
                    if axis_index < AXIS_NAMES.len() {
                        let mut title = format!("{} P, I, D Activity", AXIS_NAMES[axis_index]);
                        // Append PID values if available
                        if let Some(ref pid) = pid_data {
                            let pid_values = match axis_index {
                                0 => &pid.roll,
                                1 => &pid.pitch,
                                2 => &pid.yaw,
                                _ => &pid.roll, // fallback
                            };
                            let mut pid_str = String::new();
                            if let Some(p) = pid_values.p {
                                pid_str.push_str(&format!("P={}", p));
                            }
                            if let Some(i) = pid_values.i {
                                if !pid_str.is_empty() {
                                    pid_str.push_str(", ");
                                }
                                pid_str.push_str(&format!("I={}", i));
                            }
                            if let Some(d) = pid_values.d {
                                if !pid_str.is_empty() {
                                    pid_str.push_str(", ");
                                }
                                pid_str.push_str(&format!("D={}", d));
                            }
                            if !pid_str.is_empty() {
                                title.push_str(&format!(" ({})", pid_str));
                            }
                        }
                        title
                    } else {
                        format!("Axis {} P, I, D Activity", axis_index)
                    }
                },
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Term Value".to_string(),
            ))
        },
    )
}

// src/plot_functions/plot_pid_activity.rs
