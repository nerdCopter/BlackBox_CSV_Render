// src/plot_functions/plot_setpoint_vs_gyro.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::types::AllAxisPlotData2;

use crate::constants::{
    COLOR_SETPOINT_VS_GYRO_GYRO, COLOR_SETPOINT_VS_GYRO_SP, LINE_WIDTH_PLOT,
    UNIFIED_Y_AXIS_HEADROOM_SCALE, UNIFIED_Y_AXIS_MIN_SCALE, UNIFIED_Y_AXIS_PERCENTILE,
};
use crate::data_analysis::filter_delay;
use crate::data_analysis::filter_delay::DelayAnalysisResult;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};

/// Generates the Stacked Setpoint vs Gyro Plot (Orange, Blue)
pub fn plot_setpoint_vs_gyro(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file_setpoint_gyro = format!("{root_name}_SetpointVsGyro_stacked.png");
    let plot_type_name = "Setpoint/Gyro";

    // Calculate filtering delay using enhanced cross-correlation
    let delay_analysis = if let Some(sr) = sample_rate {
        filter_delay::calculate_average_filtering_delay_comparison(log_data, sr)
    } else {
        DelayAnalysisResult {
            average_delay: None,
            results: Vec::new(),
        }
    };

    let delay_comparison_results = if !delay_analysis.results.is_empty() {
        Some(delay_analysis.results)
    } else {
        None
    };

    let mut axis_plot_data: AllAxisPlotData2 = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..AXIS_NAMES.len() {
                axis_plot_data[axis_index].push((
                    time,
                    row.setpoint[axis_index],
                    row.gyro[axis_index],
                ));
            }
        }
    }

    let color_sp: RGBColor = *COLOR_SETPOINT_VS_GYRO_SP;
    let color_gyro: RGBColor = *COLOR_SETPOINT_VS_GYRO_GYRO;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    // Collect all absolute values for percentile calculation (issue #125)
    let mut all_abs_vals = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..AXIS_NAMES.len() {
        let data = &axis_plot_data[axis_index];
        for (_, setpoint, gyro_filt) in data {
            if let Some(s) = setpoint {
                all_abs_vals.push(s.abs());
            }
            if let Some(g) = gyro_filt {
                all_abs_vals.push(g.abs());
            }
        }
    }

    // Calculate 95th percentile for Y-axis scaling
    // This provides better visualization by not letting outliers (crashes, hard landings)
    // compress the normal flight data. Analysis of 148 logs shows P95 is typically
    // only 27% of absolute max, meaning outliers dominate current scaling.
    let half_range = if !all_abs_vals.is_empty() {
        all_abs_vals.sort_by(|a, b| a.total_cmp(b));
        let p95_idx =
            ((all_abs_vals.len() - 1) as f64 * UNIFIED_Y_AXIS_PERCENTILE).floor() as usize;
        let p95_val = all_abs_vals[p95_idx];
        let scaled_p95 = p95_val * UNIFIED_Y_AXIS_HEADROOM_SCALE;
        scaled_p95.max(UNIFIED_Y_AXIS_MIN_SCALE)
    } else {
        UNIFIED_Y_AXIS_MIN_SCALE
    };

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

            for (time, setpoint, gyro_filt) in data {
                time_min = time_min.min(*time);
                time_max = time_max.max(*time);

                if let Some(s) = setpoint {
                    setpoint_series_data.push((*time, *s));
                }
                if let Some(g) = gyro_filt {
                    gyro_series_data.push((*time, *g));
                }
            }

            if setpoint_series_data.is_empty() && gyro_series_data.is_empty() {
                return None;
            }

            // Use unified symmetric Y-axis range across all axes
            let x_range = time_min..time_max;
            let y_range = -half_range..half_range;

            let mut series = Vec::new();
            if !gyro_series_data.is_empty() {
                let gyro_label = if let Some(ref results) = delay_comparison_results {
                    // Show comparison of both methods if available - NO AVERAGING
                    let mut method_strings = Vec::new();
                    for result in results.iter() {
                        if let Some(freq) = result.frequency_hz {
                            method_strings.push(format!(
                                "{}: {:.1}ms@{:.0}Hz(c:{:.0}%)",
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown",
                                },
                                result.delay_ms,
                                freq,
                                result.confidence * 100.0
                            ));
                        } else {
                            method_strings.push(format!(
                                "{}: {:.1}ms(c:{:.0}%)",
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown",
                                },
                                result.delay_ms,
                                result.confidence * 100.0
                            ));
                        }
                    }
                    if method_strings.is_empty() {
                        "Gyro (gyroADC)".to_string()
                    } else {
                        format!("Gyro (gyroADC) - {}", method_strings.join(" vs "))
                    }
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
                { format!("{} Setpoint vs Gyro", AXIS_NAMES[axis_index]) },
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
