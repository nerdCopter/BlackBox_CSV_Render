// src/plot_functions/plot_eso.rs
// Time-domain plot of ESO estimated output (omega_hat) vs measured gyro (omega_meas),
// plus scaled disturbance estimate (f_hat) — one stacked subplot per axis.

use plotters::style::RGBColor;
use std::error::Error;

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    COLOR_ESO_FHAT, COLOR_ESO_HAT, COLOR_ESO_MEAS, ESO_FHAT_Y_FRACTION, LINE_WIDTH_PLOT,
    UNIFIED_Y_AXIS_HEADROOM_SCALE, UNIFIED_Y_AXIS_MIN_SCALE, UNIFIED_Y_AXIS_PERCENTILE,
};
use crate::eso::EsoResult;
use crate::plot_framework::{draw_stacked_plot, PlotSeries};

/// Generates the stacked ESO estimated-output plot.
///
/// Each axis subplot shows:
///   - omega_meas: the measured (filtered) gyro rate (blue)
///   - omega_hat: the ESO estimated rate (orange)
///   - f_hat: the disturbance estimate, scaled to ±50% of the omega range (green)
///
/// Only axes with a valid EsoResult are rendered; others show the unavailable message.
pub fn plot_eso_output(
    eso_results: &[Option<EsoResult>; 3],
    root_name: &str,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_ESO_output_stacked.png");
    let plot_type_name = "ESO Output";

    let color_meas: RGBColor = *COLOR_ESO_MEAS;
    let color_hat: RGBColor = *COLOR_ESO_HAT;
    let color_fhat: RGBColor = *COLOR_ESO_FHAT;
    let stroke = LINE_WIDTH_PLOT;

    // Pre-compute per-axis plot data so the closure can own it.
    let mut axis_data: [Option<AxisEsoData>; 3] = [None, None, None];
    for (i, result) in eso_results.iter().enumerate() {
        if let Some(eso) = result {
            axis_data[i] = Some(build_axis_data(eso));
        }
    }

    // Collect all |omega_meas| values across all axes for a unified Y range.
    let mut all_abs: Vec<f64> = Vec::new();
    for data in axis_data.iter().flatten() {
        for &(_, m, _) in &data.meas_hat {
            all_abs.push(m.abs());
        }
    }
    let half_range = if !all_abs.is_empty() {
        all_abs.sort_by(|a, b| a.total_cmp(b));
        let p95_idx = ((all_abs.len() - 1) as f64 * UNIFIED_Y_AXIS_PERCENTILE).floor() as usize;
        (all_abs[p95_idx] * UNIFIED_Y_AXIS_HEADROOM_SCALE).max(UNIFIED_Y_AXIS_MIN_SCALE)
    } else {
        UNIFIED_Y_AXIS_MIN_SCALE
    };

    draw_stacked_plot(&output_file, root_name, plot_type_name, move |axis_index| {
        let data = axis_data[axis_index].as_ref()?;
        if data.meas_hat.is_empty() {
            return None;
        }

        let time_min = data.meas_hat.first().map(|&(t, _, _)| t).unwrap_or(0.0);
        let time_max = data.meas_hat.last().map(|&(t, _, _)| t).unwrap_or(0.0);
        if time_min >= time_max {
            return None;
        }

        let x_range = time_min..time_max;
        let y_range = -half_range..half_range;

        let meas_data: Vec<(f64, f64)> = data.meas_hat.iter().map(|&(t, m, _)| (t, m)).collect();
        let hat_data: Vec<(f64, f64)> = data.meas_hat.iter().map(|&(t, _, h)| (t, h)).collect();

        // Draw hat first (thick, background) then meas on top (thin, foreground).
        // Since a well-tuned ESO tracks closely, the orange will only visibly peek
        // out at transients where the observer briefly diverges from the measurement.
        let mut series = vec![
            PlotSeries {
                data: hat_data,
                label: format!(
                    "ESO estimate (omega_hat, omega0={:.0} rad/s, b0={:.2})",
                    data.omega0_opt, data.b0
                ),
                color: color_hat,
                stroke_width: stroke + 1,
            },
            PlotSeries {
                data: meas_data,
                label: "Measured gyro (omega_meas)".to_string(),
                color: color_meas,
                stroke_width: stroke,
            },
        ];

        // Scale f_hat to fit ±50% of the omega Y range for visual interpretability.
        if data.fhat_max_abs > 1e-12 {
            let scale = (half_range * ESO_FHAT_Y_FRACTION) / data.fhat_max_abs;
            let fhat_data: Vec<(f64, f64)> =
                data.fhat.iter().map(|&(t, f)| (t, f * scale)).collect();
            series.push(PlotSeries {
                data: fhat_data,
                label: format!("f_hat x{scale:.2e} (disturbance est., scaled)"),
                color: color_fhat,
                stroke_width: stroke,
            });
        }

        Some((
            format!("{} ESO Output", AXIS_NAMES[axis_index]),
            x_range,
            y_range,
            series,
            "Time (s)".to_string(),
            "Rate (deg/s)".to_string(),
        ))
    })
}

// Internal per-axis pre-computed data.
struct AxisEsoData {
    omega0_opt: f64,
    b0: f64,
    /// (time, omega_meas, omega_hat) triples
    meas_hat: Vec<(f64, f64, f64)>,
    /// (time, f_hat_raw) before visual scaling
    fhat: Vec<(f64, f64)>,
    /// Maximum absolute f_hat value; used to compute scale inside draw_stacked_plot.
    fhat_max_abs: f64,
}

fn build_axis_data(eso: &EsoResult) -> AxisEsoData {
    let n = eso
        .timestamps
        .len()
        .min(eso.omega_meas_trace.len())
        .min(eso.omega_hat_trace.len())
        .min(eso.f_hat_trace.len());

    let meas_hat: Vec<(f64, f64, f64)> = (0..n)
        .map(|k| {
            (
                eso.timestamps[k],
                eso.omega_meas_trace[k],
                eso.omega_hat_trace[k],
            )
        })
        .collect();

    let fhat: Vec<(f64, f64)> = (0..n)
        .map(|k| (eso.timestamps[k], eso.f_hat_trace[k]))
        .collect();

    let fhat_max_abs = fhat.iter().map(|&(_, f)| f.abs()).fold(0.0_f64, f64::max);

    AxisEsoData {
        omega0_opt: eso.omega0_opt,
        b0: eso.b0,
        meas_hat,
        fhat,
        fhat_max_abs,
    }
}
