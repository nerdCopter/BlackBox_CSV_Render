// src/data_analysis/threshold_events.rs

/// Counts discrete rising-edge crossings above `threshold` in a time-ordered `(time, value)`
/// series — how many separate times the value entered the "above threshold" state, not how
/// long it stayed there. Each interval `[points[i], points[i+1])` is classified by the value at
/// its start (sample-and-hold), the same convention `analyze_stick_zones`/
/// `detect_rc_command_steps` use elsewhere in this codebase for interval timing.
///
/// This is a plain level-crossing counter — it does not weight by duration, self-relative
/// baselines, or a delta-over-window trigger. `motor_desync.rs`'s fallback-event detection
/// (windowed average + self-relative percentile + refractory debounce) and
/// `torque_inertia_profiler.rs`'s throttle-punch detection (delta over a fixed time window) are
/// genuinely different algorithms solving different problems, not variants of this one — they
/// are intentionally not built on this helper.
///
/// Returns 0 for fewer than 2 points.
pub fn count_rising_edge_events(points: &[(f64, f64)], threshold: f64) -> u32 {
    if points.len() < 2 {
        return 0;
    }

    let mut count = 0_u32;
    let mut was_above = false;
    for window in points.windows(2) {
        let (t0, v0) = window[0];
        let (t1, _) = window[1];
        let dt = t1 - t0;
        if !dt.is_finite() || dt <= 0.0 {
            continue;
        }
        let is_above = v0 >= threshold;
        if is_above && !was_above {
            count += 1;
        }
        was_above = is_above;
    }
    count
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
    fn fewer_than_two_points_returns_zero() {
        assert_eq!(count_rising_edge_events(&[], 95.0), 0);
        assert_eq!(count_rising_edge_events(&points_from(&[100.0]), 95.0), 0);
    }

    #[test]
    fn single_excursion_counts_once_regardless_of_duration() {
        // Peak sample itself never becomes a window start (windows(2) only uses it as the
        // trailing endpoint of the prior interval) — hold it for two samples to measure it,
        // same convention as the rest of this codebase.
        let points = points_from(&[100.0, 100.0, 0.0]);
        assert_eq!(count_rising_edge_events(&points, 95.0), 1);
    }

    #[test]
    fn two_separate_excursions_count_as_two_not_four() {
        let points = points_from(&[100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 0.0]);
        assert_eq!(count_rising_edge_events(&points, 95.0), 2);
    }

    #[test]
    fn never_crossing_threshold_counts_zero() {
        let points = points_from(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(count_rising_edge_events(&points, 95.0), 0);
    }

    #[test]
    fn starting_above_threshold_counts_as_one_event() {
        let points = points_from(&[100.0, 100.0, 0.0]);
        assert_eq!(count_rising_edge_events(&points, 95.0), 1);
    }

    #[test]
    fn exactly_at_threshold_counts_as_above() {
        let points = points_from(&[95.0, 95.0, 0.0]);
        assert_eq!(count_rising_edge_events(&points, 95.0), 1);
    }

    #[test]
    fn non_positive_time_delta_is_skipped_not_counted() {
        // Every timestamp identical (duplicate/frozen clock) — every window has dt <= 0 and
        // must be skipped entirely, not miscounted.
        let points = vec![(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)];
        assert_eq!(count_rising_edge_events(&points, 95.0), 0);
    }

    #[test]
    fn mid_series_duplicate_timestamp_does_not_reset_ongoing_excursion() {
        // A duplicate timestamp mid-series, while still above threshold on both sides, must not
        // reset `was_above` — otherwise the excursion continuing past the skipped window would
        // be miscounted as a second rising edge (2) instead of the single excursion it is (1).
        let points = vec![
            (0.0, 100.0),
            (0.01, 0.0),
            (0.01, 100.0), // duplicate timestamp with the previous point (dt == 0, skipped)
            (0.02, 100.0),
            (0.03, 0.0),
        ];
        assert_eq!(count_rising_edge_events(&points, 95.0), 1);
    }
}
