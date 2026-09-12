// src/plot_functions/plot_motor_erpm.rs

use plotters::prelude::*;
use std::error::Error;

use crate::constants::{
    LINE_WIDTH_PLOT, MOTOR_ERPM_EVENT_MARKER_WIDTH, MOTOR_ERPM_MIN_RANGE,
    MOTOR_ERPM_PLOT_Y_AXIS_MAX, MOTOR_ERPM_PLOT_Y_AXIS_MIN, PLOT_HEIGHT, PLOT_WIDTH,
};
use crate::data_input::log_data::LogRowData;
use crate::plot_functions::motor_desync::MotorDesyncResult;

/// Per-motor colors, distinct from `plot_motor_spectrums.rs`'s palette so the two plot types
/// don't visually imply a relationship they don't have.
const MOTOR_LINE_COLOR: RGBColor = RGBColor(0, 102, 204);
const ERPM_LINE_COLOR: RGBColor = RGBColor(255, 102, 0);
const EVENT_MARKER_COLOR: RGBColor = RGBColor(200, 0, 0);

/// Generates a stacked plot, one row per motor, showing `motor[N]` (commanded output) and
/// `eRPM[N]` (RPM telemetry) together on the same axes — each normalized to its own 0-100% of
/// its observed range in this log, since raw motor and eRPM units are not on a comparable scale
/// (see `motor_desync.rs`'s self-relative design notes for why). A motor/ESC failing to respond
/// shows up directly here: the two traces separate, one near its ceiling while the other stays
/// low. Vertical markers mark every `MotorDesyncResult` event timestamp for that motor,
/// regardless of confidence tier, so a flagged moment can be located on sight.
///
/// Skips (writes no file) when no motor in this log has eRPM telemetry at all — this plot has
/// nothing to show without it, consistent with `Motor Desync Detection`'s own "Insufficient
/// signal" handling.
pub fn plot_motor_erpm(
    log_data: &[LogRowData],
    root_name: &str,
    motor_desync_results: &[MotorDesyncResult],
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_Motor_vs_eRPM_stacked.png");

    let motor_count = log_data.first().map(|row| row.motors.len()).unwrap_or(0);
    if motor_count == 0 {
        println!("\nINFO: Skipping Motor vs eRPM Plot: No motor data available.");
        return Ok(());
    }

    // Per-motor (time, motor, eRPM) samples, only where both exist for that row — mirrors
    // motor_desync.rs's collect_motor_samples so the plotted data matches what was analyzed.
    let mut per_motor_samples: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); motor_count];
    for row in log_data {
        let Some(t) = row.time_sec else { continue };
        for (motor_idx, motor_samples) in per_motor_samples.iter_mut().enumerate().take(motor_count)
        {
            if let (Some(Some(m)), Some(Some(e))) = (
                row.motors.get(motor_idx).copied(),
                row.erpms.get(motor_idx).copied(),
            ) {
                motor_samples.push((t, m, e));
            }
        }
    }

    if per_motor_samples.iter().all(Vec::is_empty) {
        println!("\nINFO: Skipping Motor vs eRPM Plot: No eRPM telemetry in this log.");
        return Ok(());
    }

    // A motor row is only plottable when both its command and eRPM vary enough to normalize to
    // 0-100% (see the per-row range gate below) — check up front so a log where every motor
    // fails that gate (e.g. eRPM present but every motor near-constant, a ground test) skips
    // entirely instead of writing a blank canvas with no data.
    let plottable = |samples: &Vec<(f64, f64, f64)>| -> bool {
        if samples.len() < 2 {
            return false;
        }
        let motor_min = samples
            .iter()
            .map(|(_, m, _)| *m)
            .fold(f64::INFINITY, f64::min);
        let motor_max = samples
            .iter()
            .map(|(_, m, _)| *m)
            .fold(f64::NEG_INFINITY, f64::max);
        let erpm_min = samples
            .iter()
            .map(|(_, _, e)| *e)
            .fold(f64::INFINITY, f64::min);
        let erpm_max = samples
            .iter()
            .map(|(_, _, e)| *e)
            .fold(f64::NEG_INFINITY, f64::max);
        motor_max - motor_min >= MOTOR_ERPM_MIN_RANGE && erpm_max - erpm_min >= MOTOR_ERPM_MIN_RANGE
    };
    if !per_motor_samples.iter().any(plottable) {
        println!(
            "\nINFO: Skipping Motor vs eRPM Plot: No motor has enough command/eRPM variation to plot."
        );
        return Ok(());
    }

    println!(
        "\n--- Generating Motor vs eRPM Plot ({} motor{}) ---",
        motor_count,
        if motor_count == 1 { "" } else { "s" }
    );

    let root_area = BitMapBackend::new(&output_file, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE)?;
    root_area.draw(&Text::new(
        root_name,
        (10, 10),
        crate::font_config::FONT_TUPLE_MAIN_TITLE
            .into_font()
            .color(&BLACK),
    ))?;

    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((motor_count, 1));

    let y_range = MOTOR_ERPM_PLOT_Y_AXIS_MIN..MOTOR_ERPM_PLOT_Y_AXIS_MAX;

    for (motor_idx, samples) in per_motor_samples.iter().enumerate() {
        let area = &sub_plot_areas[motor_idx];

        if samples.len() < 2 {
            continue;
        }

        let motor_min = samples
            .iter()
            .map(|(_, m, _)| *m)
            .fold(f64::INFINITY, f64::min);
        let motor_max = samples
            .iter()
            .map(|(_, m, _)| *m)
            .fold(f64::NEG_INFINITY, f64::max);
        let erpm_min = samples
            .iter()
            .map(|(_, _, e)| *e)
            .fold(f64::INFINITY, f64::min);
        let erpm_max = samples
            .iter()
            .map(|(_, _, e)| *e)
            .fold(f64::NEG_INFINITY, f64::max);
        let motor_range = motor_max - motor_min;
        let erpm_range = erpm_max - erpm_min;

        // A near-constant command or eRPM can't be meaningfully normalized to 0-100% — nothing
        // useful to show for this motor, so leave its row blank rather than divide by ~zero.
        if motor_range < MOTOR_ERPM_MIN_RANGE || erpm_range < MOTOR_ERPM_MIN_RANGE {
            continue;
        }

        let t_min = samples.first().map(|(t, _, _)| *t).unwrap_or(0.0);
        let t_max = samples.last().map(|(t, _, _)| *t).unwrap_or(1.0);
        let x_range = t_min..t_max;

        let motor_series: Vec<(f64, f64)> = samples
            .iter()
            .map(|(t, m, _)| (*t, (m - motor_min) / motor_range * 100.0))
            .collect();
        let erpm_series: Vec<(f64, f64)> = samples
            .iter()
            .map(|(t, _, e)| (*t, (e - erpm_min) / erpm_range * 100.0))
            .collect();

        let chart_title = format!("Motor {motor_idx}: Command vs eRPM (% of own range)");
        let mut chart = ChartBuilder::on(area)
            .caption(&chart_title, crate::font_config::FONT_TUPLE_CHART_TITLE)
            .margin(5)
            .x_label_area_size(50)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range.clone())?;

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("% of own range")
            .x_labels(20)
            .y_labels(10)
            .light_line_style(WHITE.mix(0.7))
            .label_style(crate::font_config::FONT_TUPLE_AXIS_LABEL)
            .draw()?;

        // Mark every flagged event for this motor, any confidence tier — a single vertical
        // line so a reader can go straight to the moment the detector flagged. Drawn before
        // the motor/eRPM traces so the traces render on top of the marker, not the other way
        // around.
        if let Some(result) = motor_desync_results
            .iter()
            .find(|r| r.motor_idx == motor_idx)
        {
            for (event_idx, event) in result.events.iter().enumerate() {
                let series = chart.draw_series(std::iter::once(PathElement::new(
                    vec![
                        (event.time_s, MOTOR_ERPM_PLOT_Y_AXIS_MIN),
                        (event.time_s, MOTOR_ERPM_PLOT_Y_AXIS_MAX),
                    ],
                    ShapeStyle::from(EVENT_MARKER_COLOR)
                        .stroke_width(MOTOR_ERPM_EVENT_MARKER_WIDTH),
                )))?;
                // Only the first marker gets a legend entry — one line per row, not one per event.
                if event_idx == 0 {
                    series.label("Desync event").legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)],
                            ShapeStyle::from(EVENT_MARKER_COLOR)
                                .stroke_width(MOTOR_ERPM_EVENT_MARKER_WIDTH),
                        )
                    });
                }
            }
        }

        chart
            .draw_series(LineSeries::new(
                motor_series,
                ShapeStyle::from(MOTOR_LINE_COLOR).stroke_width(LINE_WIDTH_PLOT),
            ))?
            .label("Motor command")
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(MOTOR_LINE_COLOR).stroke_width(LINE_WIDTH_PLOT),
                )
            });

        chart
            .draw_series(LineSeries::new(
                erpm_series,
                ShapeStyle::from(ERPM_LINE_COLOR).stroke_width(LINE_WIDTH_PLOT),
            ))?
            .label("eRPM")
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(ERPM_LINE_COLOR).stroke_width(LINE_WIDTH_PLOT),
                )
            });

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(crate::font_config::FONT_TUPLE_LEGEND)
            .draw()?;
    }

    root_area.present()?;
    println!("  Stacked plot saved as '{output_file}'.");

    Ok(())
}
