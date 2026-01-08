// src/plot_functions/plot_gyro_vs_unfilt.rs

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_GYRO_VS_UNFILT_FILT, COLOR_GYRO_VS_UNFILT_UNFILT, LINE_WIDTH_PLOT,
    UNIFIED_Y_AXIS_MIN_SCALE, UNIFIED_Y_AXIS_PERCENTILE, UNIFIED_Y_AXIS_PERCENTILE_SCALE,
};
use crate::data_analysis::filter_delay;
use crate::data_analysis::filter_delay::DelayAnalysisResult;
use crate::data_input::log_data::LogRowData;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};
use crate::types::AllAxisPlotData2;

/// Generates the Stacked Gyro vs Unfiltered Gyro Plot (Purple, Orange)
pub fn plot_gyro_vs_unfilt(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
    using_debug_fallback: bool,
    debug_mode_name: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let output_file_gyro = format!("{root_name}_GyroVsUnfilt_stacked.png");
    let plot_type_name = "Gyro/UnfiltGyro";

    // Clone debug mode name to move into closure
    let debug_mode_name_owned = debug_mode_name.map(|s| s.to_string());

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

    let mut axis_plot_data: AllAxisPlotData2 = Default::default();
    for row in log_data {
        if let Some(time) = row.time_sec {
            #[allow(clippy::needless_range_loop)]
            for axis_index in 0..AXIS_NAMES.len() {
                axis_plot_data[axis_index].push((
                    time,
                    row.gyro[axis_index],
                    row.gyro_unfilt[axis_index],
                ));
            }
        }
    }

    let color_gyro_unfilt: RGBColor = *COLOR_GYRO_VS_UNFILT_UNFILT;
    let color_gyro_filt: RGBColor = *COLOR_GYRO_VS_UNFILT_FILT;
    let line_stroke_plot = LINE_WIDTH_PLOT;

    // Collect all absolute values for percentile calculation (issue #125)
    let mut all_abs_vals = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for axis_index in 0..AXIS_NAMES.len() {
        let data = &axis_plot_data[axis_index];
        for (_, gyro_filt, gyro_unfilt) in data {
            if let Some(gf) = gyro_filt {
                all_abs_vals.push(gf.abs());
            }
            if let Some(gu) = gyro_unfilt {
                all_abs_vals.push(gu.abs());
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
        let scaled_p95 = p95_val * UNIFIED_Y_AXIS_PERCENTILE_SCALE;
        scaled_p95.max(UNIFIED_Y_AXIS_MIN_SCALE)
    } else {
        UNIFIED_Y_AXIS_MIN_SCALE
    };

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

            for (time, gyro_filt, gyro_unfilt) in data {
                time_min = time_min.min(*time);
                time_max = time_max.max(*time);

                if let Some(gf) = gyro_filt {
                    filt_series_data.push((*time, *gf));
                }
                if let Some(gu) = gyro_unfilt {
                    unfilt_series_data.push((*time, *gu));
                }
            }

            if filt_series_data.is_empty() && unfilt_series_data.is_empty() {
                return None;
            }

            // Use unified Y-axis range across all axes
            let x_range = time_min..time_max;
            let y_range = -half_range..half_range;

            let mut series = Vec::new();
            if !unfilt_series_data.is_empty() {
                // Create label with debug mode annotation if using debug fallback
                let unfilt_label = super::format_debug_suffix(
                    "Unfiltered Gyro (gyroUnfilt)",
                    using_debug_fallback,
                    debug_mode_name_owned.as_deref(),
                );

                series.push(PlotSeries {
                    data: unfilt_series_data,
                    label: unfilt_label,
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
                        "Filtered Gyro (gyroADC) - No delay calculated".to_string()
                    } else if method_strings.len() == 1 {
                        format!("Filtered Gyro (gyroADC) - {}", method_strings[0])
                    } else {
                        format!("Filtered Gyro (gyroADC) - {}", method_strings.join(" vs "))
                    }
                } else if let Some(delay) = average_delay_ms {
                    // Fallback to single method display
                    format!("Filtered Gyro (gyroADC) - Delay: {delay:.1}ms")
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
                { format!("{} Filtered vs Unfiltered Gyro", AXIS_NAMES[axis_index]) },
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
