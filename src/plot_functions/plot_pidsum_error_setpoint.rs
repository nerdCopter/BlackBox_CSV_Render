// src/plot_functions/plot_pidsum_error_setpoint.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::types::AllAxisPlotData3;

use crate::constants::{
    COLOR_PIDERROR_MAIN, COLOR_PIDSUM_MAIN, COLOR_SETPOINT_MAIN, LINE_WIDTH_PLOT,
};
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{calculate_range, draw_stacked_plot, PlotSeries};

/// Generates the Stacked PIDsum vs PID Error vs Setpoint Plot (Green, Blue, Yellow)
pub fn plot_pidsum_error_setpoint(
    log_data: &[LogRowData],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file_pidsum_error = format!("{root_name}_PIDsum_PIDerror_Setpoint_stacked.png");
    let plot_type_name = "PIDsum/PIDerror/Setpoint";

    let mut axis_plot_data: AllAxisPlotData3 = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..3 {
                let pidsum = row.p_term[axis_index].and_then(|p| {
                    row.i_term[axis_index].and_then(|i| row.d_term[axis_index].map(|d| p + i + d))
                });
                axis_plot_data[axis_index].push((
                    time,
                    row.setpoint[axis_index],
                    row.gyro[axis_index],
                    pidsum,
                ));
            }
        }
    }

    let color_pidsum: RGBColor = *COLOR_PIDSUM_MAIN;
    let color_pid_error: RGBColor = *COLOR_PIDERROR_MAIN;
    let color_setpoint: RGBColor = *COLOR_SETPOINT_MAIN;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_pidsum_error,
        root_name,
        plot_type_name,
        move |axis_index| {
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

            if pidsum_series_data.is_empty()
                && setpoint_series_data.is_empty()
                && pid_error_series_data.is_empty()
            {
                return None;
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !pidsum_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pidsum_series_data,
                    label: "PIDsum (P+I+D)".to_string(),
                    color: color_pidsum,
                    stroke_width: line_stroke_plot,
                });
            }
            if !pid_error_series_data.is_empty() {
                series.push(PlotSeries {
                    data: pid_error_series_data,
                    label: "PID Error (Setpoint - GyroADC)".to_string(),
                    color: color_pid_error,
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
                    let axis_names = ["Roll", "Pitch", "Yaw"];
                    format!("{} PIDsum vs PID Error vs Setpoint", axis_names[axis_index])
                },
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Value".to_string(),
            ))
        },
    )
}

// src/plot_functions/plot_pidsum_error_setpoint.rs
