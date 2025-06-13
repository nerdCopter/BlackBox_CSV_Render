// src/plot_functions/plot_setpoint_vs_gyro.rs

use std::error::Error;
use plotters::style::RGBColor;

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries, calculate_range};
use crate::constants::{
    COLOR_SETPOINT_VS_GYRO_SP, COLOR_SETPOINT_VS_GYRO_GYRO,
    LINE_WIDTH_PLOT,
};
use crate::data_analysis::filter_delay;

/// Generates the Stacked Setpoint vs Gyro Plot (Orange, Blue)
pub fn plot_setpoint_vs_gyro(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file_setpoint_gyro = format!("{}_SetpointVsGyro_stacked.png", root_name);
    let plot_type_name = "Setpoint/Gyro";

    // Calculate filtering delay
    let average_delay_ms = if let Some(sr) = sample_rate {
        filter_delay::calculate_average_filtering_delay(log_data, sr)
    } else {
        None
    };

    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
     for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.setpoint[axis_index], row.gyro[axis_index]));
             }
         }
     }

    let color_sp: RGBColor = *COLOR_SETPOINT_VS_GYRO_SP;
    let color_gyro: RGBColor = *COLOR_SETPOINT_VS_GYRO_GYRO;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_setpoint_gyro,
        root_name,
        plot_type_name,
        move |axis_index| {
             let data = &axis_plot_data[axis_index];
             if data.is_empty() {
                 return None;
             }

             let mut setpoint_series_data: Vec<(f64, f64)> = Vec::new();
             let mut gyro_series_data: Vec<(f64, f64)> = Vec::new();

             let mut time_min = f64::INFINITY;
             let mut time_max = f64::NEG_INFINITY;
             let mut val_min = f64::INFINITY;
             let mut val_max = f64::NEG_INFINITY;

             for (time, setpoint, gyro_filt) in data {
                 time_min = time_min.min(*time);
                 time_max = time_max.max(*time);

                 if let Some(s) = setpoint {
                     setpoint_series_data.push((*time, *s));
                     val_min = val_min.min(*s);
                     val_max = val_max.max(*s);
                 }
                 if let Some(g) = gyro_filt {
                     gyro_series_data.push((*time, *g));
                     val_min = val_min.min(*g);
                     val_max = val_max.max(*g);
                 }
             }

            if setpoint_series_data.is_empty() && gyro_series_data.is_empty() {
                 return None;
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !gyro_series_data.is_empty() {
                let gyro_label = if let Some(delay) = average_delay_ms {
                    format!("Gyro (gyroADC) - Delay: {:.1}ms", delay)
                } else {
                    "Gyro (gyroADC)".to_string()
                };
                 series.push(PlotSeries {
                     data: gyro_series_data,
                     label: gyro_label,
                     color: color_gyro,
                     stroke_width: line_stroke_plot,
                 });
            }
            if !setpoint_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: setpoint_series_data,
                     label: "Setpoint".to_string(),
                     color: color_sp,
                     stroke_width: line_stroke_plot,
                 });
             }

            Some((
                format!("Axis {} Setpoint vs Gyro", axis_index),
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Value".to_string(),
            ))
        },
    )
}

// src/plot_functions/plot_setpoint_vs_gyro.rs