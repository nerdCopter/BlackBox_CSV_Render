// src/plot_functions/plot_gyro_vs_unfilt.rs

use std::error::Error;
use plotters::style::RGBColor;

use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries, calculate_range};
use crate::constants::{
    COLOR_GYRO_VS_UNFILT_UNFILT, COLOR_GYRO_VS_UNFILT_FILT,
    LINE_WIDTH_PLOT,
};
use crate::data_analysis::filter_delay;
use crate::data_analysis::filter_delay::DelayAnalysisResult;

/// Generates the Stacked Gyro vs Unfiltered Gyro Plot (Purple, Orange)
pub fn plot_gyro_vs_unfilt(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
    let plot_type_name = "Gyro/UnfiltGyro";

    // Calculate filtering delay using enhanced cross-correlation
    let delay_analysis = if let Some(sr) = sample_rate {
        filter_delay::calculate_average_filtering_delay_comparison(log_data, sr)
    } else {
        DelayAnalysisResult {
            average_delay: None,
            results: Vec::new(),
        }
    };
    
    let average_delay_ms = delay_analysis.average_delay;
    let delay_comparison_results = if !delay_analysis.results.is_empty() {
        Some(delay_analysis.results)
    } else {
        None
    };

    let mut axis_plot_data: [Vec<(f64, Option<f64>, Option<f64>)>; 3] = Default::default();
    for row in log_data {
         if let Some(time) = row.time_sec {
             for axis_index in 0..3 {
                 axis_plot_data[axis_index].push((time, row.gyro[axis_index], row.gyro_unfilt[axis_index]));
             }
         }
     }

    let color_gyro_unfilt: RGBColor = *COLOR_GYRO_VS_UNFILT_UNFILT;
    let color_gyro_filt: RGBColor = *COLOR_GYRO_VS_UNFILT_FILT;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    draw_stacked_plot(
        &output_file_gyro,
        root_name,
        plot_type_name,
        move |axis_index| {
             let data = &axis_plot_data[axis_index];
             if data.is_empty() {
                 return None;
             }

             let mut filt_series_data: Vec<(f64, f64)> = Vec::new();
             let mut unfilt_series_data: Vec<(f64, f64)> = Vec::new();

             let mut time_min = f64::INFINITY;
             let mut time_max = f64::NEG_INFINITY;
             let mut val_min = f64::INFINITY;
             let mut val_max = f64::NEG_INFINITY;

             for (time, gyro_filt, gyro_unfilt) in data {
                 time_min = time_min.min(*time);
                 time_max = time_max.max(*time);

                 if let Some(gf) = gyro_filt {
                     filt_series_data.push((*time, *gf));
                     val_min = val_min.min(*gf);
                     val_max = val_max.max(*gf);
                 }
                 if let Some(gu) = gyro_unfilt {
                     unfilt_series_data.push((*time, *gu));
                     val_min = val_min.min(*gu);
                     val_max = val_max.max(*gu);
                 }
             }

            if filt_series_data.is_empty() && unfilt_series_data.is_empty() {
                 return None;
            }

            let (final_value_min, final_value_max) = calculate_range(val_min, val_max);
            let x_range = time_min..time_max;
            let y_range = final_value_min..final_value_max;

            let mut series = Vec::new();
            if !unfilt_series_data.is_empty() {
                 series.push(PlotSeries {
                     data: unfilt_series_data,
                     label: "Unfiltered Gyro (gyroUnfilt/debug)".to_string(),
                     color: color_gyro_unfilt,
                     stroke_width: line_stroke_plot,
                 });
            }
            if !filt_series_data.is_empty() {
                let filtered_label = if let Some(ref results) = delay_comparison_results {
                    // Show comparison of both methods if available - NO AVERAGING
                    let mut method_strings = Vec::new();
                    for result in results.iter() {
                        if let Some(freq) = result.frequency_hz {
                            method_strings.push(format!("{}: {:.1}ms@{:.0}Hz(c:{:.0}%)", 
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown"
                                },
                                result.delay_ms, freq, result.confidence * 100.0));
                        } else {
                            method_strings.push(format!("{}: {:.1}ms(c:{:.0}%)", 
                                match result.method.as_str() {
                                    "Enhanced Cross-Correlation" => "Delay",
                                    _ => "Unknown"
                                },
                                result.delay_ms, result.confidence * 100.0));
                        }
                    }
                    if method_strings.is_empty() {
                        "Filtered Gyro (gyroADC) - No delay calculated".to_string()
                    } else if method_strings.len() == 1 {
                        format!("Filtered Gyro (gyroADC) - {}", method_strings[0])
                    } else {
                        format!("Filtered Gyro (gyroADC) - {}", method_strings.join(" vs "))
                    }
                } else if let Some(delay) = average_delay_ms {
                    // Fallback to single method display
                    format!("Filtered Gyro (gyroADC) - Delay: {:.1}ms", delay)
                } else {
                    "Filtered Gyro (gyroADC)".to_string()
                };
                 series.push(PlotSeries {
                     data: filt_series_data,
                     label: filtered_label,
                     color: color_gyro_filt,
                     stroke_width: line_stroke_plot,
                 });
            }

            Some((
                format!("Axis {} Filtered vs Unfiltered Gyro", axis_index),
                x_range,
                y_range,
                series,
                "Time (s)".to_string(),
                "Gyro Value".to_string(),
            ))
        },
    )
}

// src/plot_functions/plot_gyro_vs_unfilt.rs