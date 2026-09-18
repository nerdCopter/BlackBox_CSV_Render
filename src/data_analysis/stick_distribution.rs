// src/data_analysis/stick_distribution.rs

use crate::axis_names::AXIS_NAMES;
use crate::constants::{
    RATIO_TO_PERCENT, STICK_DIST_CENTER_THRESHOLD_PCT, STICK_DIST_HIGH_THRESHOLD_PCT,
    STICK_DIST_SATURATION_THRESHOLD_PCT, UNIFIED_Y_AXIS_PERCENTILE,
};
use crate::data_analysis::rate_curve::{configured_max_rate, parse_rate_curve_config};
use crate::data_analysis::threshold_events::count_rising_edge_events;
use crate::data_input::log_data::LogRowData;

/// Per-axis stick position distribution and rate-utilization statistics.
/// Zone percentages are relative to this flight's own peak `rc_command` magnitude for the
/// axis — a data-driven reference independent of `configured_max_rate` below.
pub struct StickDistributionResult {
    pub axis_name: String,
    /// Peak |rc_command| observed for this axis — the 100% reference for the zone percentages.
    /// `None` when the axis has fewer than two RC Command samples.
    pub peak_stick: Option<f64>,
    pub center_pct: f64,
    pub mid_pct: f64,
    pub high_pct: f64,
    pub saturation_time_s: f64,
    /// Count of discrete transitions into Saturation (>95%) from below — how many separate
    /// times the stick hit the extreme, not how long each one lasted (`saturation_time_s`).
    pub saturation_event_count: u32,
    /// 95th percentile of |setpoint|/|gyro| for this axis, not the raw maximum — a single
    /// crash/tumble sample can put raw max gyro rate an order of magnitude above every other
    /// sample in the flight (same outlier problem `plot_setpoint_vs_gyro`'s Y-axis scaling
    /// guards against).
    pub p95_setpoint: Option<f64>,
    pub p95_gyro: Option<f64>,
    /// Sign-reversal rate of RC Command while inside the Center zone, in Hz — reversal count
    /// divided by time actually spent in the Center zone, not by total flight time, so it
    /// isn't diluted by however much of the flight was spent outside Center.
    /// `None` when the axis spent no time in the Center zone.
    pub center_reversal_rate_hz: Option<f64>,
    /// Setpoint at full stick deflection, computed from the header's `rates_type`/`rc_rates`/
    /// `rc_expo`/`rates`/`rate_limits` rate-curve configuration (see
    /// `data_analysis::rate_curve`). `None` when the header lacks a complete rate-curve config.
    pub configured_max_rate: Option<f64>,
    /// `p95_setpoint / configured_max_rate * 100` — how much of the configured ceiling this
    /// flight's P95 setpoint reached. `None` when either input is `None`, or
    /// `configured_max_rate` is not a positive value.
    pub rate_headroom_pct: Option<f64>,
}

#[derive(Default)]
struct StickZoneStats {
    peak_stick: Option<f64>,
    center_pct: f64,
    mid_pct: f64,
    high_pct: f64,
    saturation_time_s: f64,
    saturation_event_count: u32,
    center_reversal_rate_hz: Option<f64>,
}

/// Walks one axis's (time, rc_command) samples and computes zone-time percentages and the
/// Center-zone reversal rate.
///
/// Each interval `[points[i], points[i+1])` is classified by the value at its start (the
/// same sample-and-hold weighting `detect_rc_command_steps` uses for plateau timing), and its
/// duration credited to that zone — so the result is time-weighted, not sample-count-weighted.
fn analyze_stick_zones(points: &[(f64, f64)]) -> StickZoneStats {
    if points.len() < 2 {
        return StickZoneStats::default();
    }

    let peak = points.iter().fold(0.0_f64, |acc, (_, v)| acc.max(v.abs()));
    if peak <= 0.0 {
        return StickZoneStats {
            peak_stick: Some(peak),
            ..Default::default()
        };
    }

    // Normalized (time, % of peak) series, shared by the zone/reversal loop below and by the
    // Saturation Events count, which delegates to the generic level-crossing counter rather
    // than duplicating its own rising-edge state machine.
    let pct_points: Vec<(f64, f64)> = points
        .iter()
        .map(|(t, v)| (*t, (v.abs() / peak) * RATIO_TO_PERCENT))
        .collect();
    let saturation_event_count =
        count_rising_edge_events(&pct_points, STICK_DIST_SATURATION_THRESHOLD_PCT);

    let mut center_time = 0.0_f64;
    let mut mid_time = 0.0_f64;
    let mut high_time = 0.0_f64;
    let mut saturation_time = 0.0_f64;
    let mut total_time = 0.0_f64;
    let mut reversal_count: u64 = 0;
    let mut prev_sign: Option<f64> = None;

    for window in points.windows(2) {
        let (t0, v0) = window[0];
        let (t1, _) = window[1];
        let dt = t1 - t0;
        if !dt.is_finite() || dt <= 0.0 {
            continue;
        }
        total_time += dt;

        let pct = (v0.abs() / peak) * RATIO_TO_PERCENT;
        if pct < STICK_DIST_CENTER_THRESHOLD_PCT {
            center_time += dt;
        } else if pct < STICK_DIST_HIGH_THRESHOLD_PCT {
            mid_time += dt;
        } else {
            high_time += dt;
        }
        if pct >= STICK_DIST_SATURATION_THRESHOLD_PCT {
            saturation_time += dt;
        }

        // Reversal = a sign change of rc_command while continuously inside the Center zone.
        // Leaving the Center zone resets tracking, so a real transit out to Mid/High and back
        // is never miscounted as a center-jitter reversal. An exact-zero sample is ambiguous
        // (no sign) and leaves the tracked sign unchanged rather than breaking continuity.
        if pct >= STICK_DIST_CENTER_THRESHOLD_PCT {
            prev_sign = None;
        } else if v0 != 0.0 {
            let sign = v0.signum();
            if let Some(prev) = prev_sign {
                if prev != sign {
                    reversal_count += 1;
                }
            }
            prev_sign = Some(sign);
        }
    }

    if total_time <= 0.0 {
        return StickZoneStats {
            peak_stick: Some(peak),
            ..Default::default()
        };
    }

    StickZoneStats {
        peak_stick: Some(peak),
        center_pct: (center_time / total_time) * RATIO_TO_PERCENT,
        mid_pct: (mid_time / total_time) * RATIO_TO_PERCENT,
        high_pct: (high_time / total_time) * RATIO_TO_PERCENT,
        saturation_time_s: saturation_time,
        saturation_event_count,
        // Density of reversals within Center-zone dwell time, not diluted by time spent
        // outside it — two flights with identical center-jitter behavior but different
        // center-zone occupancy should report the same rate here.
        center_reversal_rate_hz: (center_time > 0.0).then(|| reversal_count as f64 / center_time),
    }
}

/// 95th percentile of `values`, sorting in place. Mirrors the Y-axis scaling percentile used
/// by `plot_setpoint_vs_gyro`/`plot_gyro_vs_unfilt`. `None` when `values` is empty.
fn percentile_95(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f64 * UNIFIED_Y_AXIS_PERCENTILE).floor() as usize;
    Some(values[idx])
}

/// Computes per-axis stick position distribution and rate-utilization statistics for the
/// markdown report. Report-only — no plot is generated; `rc_command` (the pre-rate-curve stick
/// input) is already visualized over time by `plot_rc_command_activity`, and a deflection
/// histogram/CDF plot tried for this feature added confusion without adding information the
/// report table doesn't already state directly.
pub fn analyze_stick_distribution(
    log_data: &[LogRowData],
    header_metadata: Option<&[(String, String)]>,
) -> Vec<StickDistributionResult> {
    let axis_count = AXIS_NAMES.len();
    let rate_curve_config = header_metadata.and_then(parse_rate_curve_config);

    let mut rc_points: Vec<Vec<(f64, f64)>> = vec![Vec::new(); axis_count];
    let mut setpoint_abs: Vec<Vec<f64>> = vec![Vec::new(); axis_count];
    let mut gyro_abs: Vec<Vec<f64>> = vec![Vec::new(); axis_count];

    for row in log_data {
        let Some(time) = row.time_sec else {
            continue;
        };
        #[allow(clippy::needless_range_loop)]
        for axis in 0..axis_count {
            if let Some(rc) = row.rc_command[axis] {
                rc_points[axis].push((time, rc));
            }
            if let Some(sp) = row.setpoint[axis] {
                setpoint_abs[axis].push(sp.abs());
            }
            if let Some(g) = row.gyro[axis] {
                gyro_abs[axis].push(g.abs());
            }
        }
    }

    let mut results = Vec::with_capacity(axis_count);

    for axis in 0..axis_count {
        let stats = analyze_stick_zones(&rc_points[axis]);
        let p95_setpoint = percentile_95(&mut setpoint_abs[axis]);
        let p95_gyro = percentile_95(&mut gyro_abs[axis]);

        let max_rate = rate_curve_config
            .as_ref()
            .and_then(|config| configured_max_rate(config, axis));
        // Headroom is the unused portion of the configured range, not the used portion.
        let rate_headroom_pct = match (p95_setpoint, max_rate) {
            (Some(sp), Some(max)) if max > 0.0 => {
                Some(RATIO_TO_PERCENT - (sp.abs() / max) * RATIO_TO_PERCENT)
            }
            _ => None,
        };

        results.push(StickDistributionResult {
            axis_name: AXIS_NAMES[axis].to_string(),
            peak_stick: stats.peak_stick,
            center_pct: stats.center_pct,
            mid_pct: stats.mid_pct,
            high_pct: stats.high_pct,
            saturation_time_s: stats.saturation_time_s,
            saturation_event_count: stats.saturation_event_count,
            p95_setpoint,
            p95_gyro,
            center_reversal_rate_hz: stats.center_reversal_rate_hz,
            configured_max_rate: max_rate,
            rate_headroom_pct,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points_from(values: &[f64]) -> Vec<(f64, f64)> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 * 0.01, v))
            .collect()
    }

    #[test]
    fn all_center_gives_full_center_pct_and_no_saturation() {
        // Peak (1.0) only ever appears as the trailing endpoint of the last interval, so
        // every timed interval is classified by the 0.1 (10% of peak) value that starts it.
        let points = points_from(&[0.1, 0.1, 0.1, 0.1, 0.1, 1.0]);
        let stats = analyze_stick_zones(&points);
        assert_eq!(stats.peak_stick, Some(1.0));
        assert!((stats.center_pct - 100.0).abs() < 1e-9);
        assert_eq!(stats.mid_pct, 0.0);
        assert_eq!(stats.high_pct, 0.0);
        assert_eq!(stats.saturation_time_s, 0.0);
    }

    #[test]
    fn full_stick_is_saturation() {
        // Peak sample itself (pct=100%) is excluded from timed intervals (windows(2) uses it
        // only as the endpoint of the prior interval), so hold it for two samples to measure it.
        let points = points_from(&[100.0, 100.0, 0.0]);
        let stats = analyze_stick_zones(&points);
        assert!(stats.saturation_time_s > 0.0);
        assert!(stats.high_pct > 0.0);
        assert_eq!(stats.saturation_event_count, 1);
    }

    #[test]
    fn saturation_events_count_separate_excursions_not_samples() {
        // Two separate trips into saturation (>95%), each held for two samples, separated by a
        // return to center — must count as 2 events, not 4 (one per saturated sample).
        let points = points_from(&[100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 0.0]);
        let stats = analyze_stick_zones(&points);
        assert_eq!(stats.saturation_event_count, 2);
    }

    #[test]
    fn reversal_counted_only_inside_center_zone() {
        // +1 -> -1 while at peak (pct=100%, outside Center) must not count.
        // +0.1 -> -0.1 relative to a peak of 1.0 (pct=10%, inside Center) must count.
        let points = points_from(&[1.0, -1.0, 0.1, -0.1, 0.1]);
        let stats = analyze_stick_zones(&points);
        assert_eq!(stats.center_reversal_rate_hz.map(|r| r > 0.0), Some(true));
    }

    #[test]
    fn zero_deflection_log_has_no_peak() {
        let points = points_from(&[0.0, 0.0, 0.0]);
        let stats = analyze_stick_zones(&points);
        assert_eq!(stats.peak_stick, Some(0.0));
        assert_eq!(stats.center_pct, 0.0);
    }

    #[test]
    fn fewer_than_two_samples_returns_default() {
        let stats = analyze_stick_zones(&points_from(&[1.0]));
        assert_eq!(stats.peak_stick, None);
    }
}
