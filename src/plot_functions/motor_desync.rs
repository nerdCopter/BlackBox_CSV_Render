// src/plot_functions/motor_desync.rs

use crate::constants::{
    MOTOR_DESYNC_ERPM_JUMP_THRESHOLD_PERCENT, MOTOR_DESYNC_EVENT_REFRACTORY_S,
    MOTOR_DESYNC_MIN_ARMED_PERCENT, MOTOR_DESYNC_MIN_ERPM_RANGE,
    MOTOR_DESYNC_MOTOR_STABLE_THRESHOLD_PERCENT,
};
use crate::data_input::log_data::LogRowData;

/// One flagged motor/eRPM divergence: eRPM changed sharply between two samples while the
/// commanded motor output barely moved over the same interval.
pub struct MotorDesyncEvent {
    pub time_s: f64,
    pub erpm_delta_percent: f64,
}

/// Per-motor desync-detection summary.
pub struct MotorDesyncResult {
    pub motor_idx: usize,
    /// False when this motor's eRPM signal never varied enough to analyze — missing
    /// telemetry, or the motor never spun meaningfully during this log.
    pub erpm_signal_available: bool,
    pub events: Vec<MotorDesyncEvent>,
}

/// Flags candidate motor-desync events by comparing each motor's commanded output
/// (`motor[N]`) against its eRPM telemetry (`eRPM[N]`) sample-to-sample.
///
/// A candidate is a sample where eRPM jumps by a large fraction of its own observed range
/// while the commanded output over the same interval barely moves, with the motor armed
/// well above idle — the ESC/eRPM link losing track of the commanded speed rather than
/// following a real throttle change. Both signals are normalized to each motor's own
/// min/max range within this log, since raw motor and eRPM units aren't a fixed absolute
/// scale across protocols or logs.
///
/// This is a divergence heuristic, not a certainty: it has not been validated against a
/// confirmed real desync event (none was available in calibration data), only checked to
/// produce zero candidates across several apparently-clean flight logs. Treat a flagged
/// event as something to cross-check against gyro/accelerometer disturbance at the same
/// timestamp, not as a standalone diagnosis.
pub fn detect_motor_desync(log_data: &[LogRowData]) -> Vec<MotorDesyncResult> {
    let motor_count = log_data
        .iter()
        .map(|row| row.motors.len())
        .max()
        .unwrap_or(0);

    let mut results = Vec::with_capacity(motor_count);

    for motor_idx in 0..motor_count {
        let samples: Vec<(f64, f64, f64)> = log_data
            .iter()
            .filter_map(|row| {
                let time = row.time_sec?;
                let motor = *row.motors.get(motor_idx)?;
                let erpm = *row.erpms.get(motor_idx)?;
                Some((time, motor?, erpm?))
            })
            .collect();

        if samples.len() < 2 {
            results.push(MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                events: Vec::new(),
            });
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

        if motor_range <= 0.0 || erpm_range < MOTOR_DESYNC_MIN_ERPM_RANGE {
            results.push(MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                events: Vec::new(),
            });
            continue;
        }

        let mut events = Vec::new();
        let mut last_event_time: Option<f64> = None;

        for pair in samples.windows(2) {
            let (t0, m0, e0) = pair[0];
            let (t1, m1, e1) = pair[1];
            if t1 <= t0 {
                continue;
            }

            let motor_delta_pct = (m1 - m0).abs() / motor_range * 100.0;
            let erpm_delta_pct = (e1 - e0).abs() / erpm_range * 100.0;
            let armed_pct = (m1 - motor_min) / motor_range * 100.0;

            let is_candidate = erpm_delta_pct >= MOTOR_DESYNC_ERPM_JUMP_THRESHOLD_PERCENT
                && motor_delta_pct <= MOTOR_DESYNC_MOTOR_STABLE_THRESHOLD_PERCENT
                && armed_pct >= MOTOR_DESYNC_MIN_ARMED_PERCENT;

            if is_candidate {
                let debounced = last_event_time
                    .map(|last| t1 - last >= MOTOR_DESYNC_EVENT_REFRACTORY_S)
                    .unwrap_or(true);
                if debounced {
                    events.push(MotorDesyncEvent {
                        time_s: t1,
                        erpm_delta_percent: erpm_delta_pct,
                    });
                    last_event_time = Some(t1);
                }
            }
        }

        results.push(MotorDesyncResult {
            motor_idx,
            erpm_signal_available: true,
            events,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(time_s: f64, motor_count: usize, motor_idx: usize, motor: f64, erpm: f64) -> LogRowData {
        let mut r = LogRowData {
            time_sec: Some(time_s),
            ..Default::default()
        };
        r.motors = vec![Some(0.0); motor_count];
        r.erpms = vec![Some(0.0); motor_count];
        r.motors[motor_idx] = Some(motor);
        r.erpms[motor_idx] = Some(erpm);
        r
    }

    #[test]
    fn erpm_tracking_motor_smoothly_flags_nothing() {
        // eRPM scales in lockstep with motor command (typical clean flight) — no candidate
        // should ever satisfy "eRPM jumps while motor stays flat".
        let mut data = Vec::new();
        for i in 0..500 {
            let t = i as f64 / 1000.0;
            let motor = 1000.0 + (i as f64 % 200.0) * 2.0;
            let erpm = motor * 1.5;
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert_eq!(results.len(), 1);
        assert!(results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn erpm_drop_under_steady_command_is_flagged() {
        // Motor command holds essentially flat (a tiny drift keeps motor_range > 0, required
        // for the low-signal guard) at a well-armed level while eRPM plummets for one sample
        // then recovers — the textbook desync signature this detector targets.
        let mut data = Vec::new();
        for i in 0..100u32 {
            let t = i as f64 / 1000.0;
            let motor = 1500.0 + i as f64 * 0.01;
            let erpm = if i == 50 { 200.0 } else { 1000.0 };
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert_eq!(results[0].events.len(), 1);
        assert!(
            results[0].events[0].erpm_delta_percent >= MOTOR_DESYNC_ERPM_JUMP_THRESHOLD_PERCENT
        );
    }

    #[test]
    fn low_signal_motor_is_not_analyzed() {
        // eRPM never varies beyond MOTOR_DESYNC_MIN_ERPM_RANGE (e.g. disconnected/unpowered
        // motor) — must be reported as unavailable, not flooded with false candidates.
        let mut data = Vec::new();
        for i in 0..200 {
            let t = i as f64 / 1000.0;
            let motor = 200.0 + (i as f64 % 100.0) * 10.0;
            data.push(row(t, 1, 0, motor, 12.0));
        }

        let results = detect_motor_desync(&data);
        assert!(!results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn near_idle_erpm_swing_is_not_flagged() {
        // eRPM swings sharply while the motor command stays perfectly flat at idle (0% of
        // its own range) — isolates the armed_pct guard from the stable-command guard. A
        // separate, later high-throttle block only exists to establish a non-zero motor
        // range; it has no internal transitions of its own to flag.
        let mut data = Vec::new();
        for i in 0..100u32 {
            let t = i as f64 / 1000.0;
            let erpm = if i == 50 { 100.0 } else { 1000.0 };
            data.push(row(t, 1, 0, 1000.0, erpm));
        }
        for i in 100..150u32 {
            let t = i as f64 / 1000.0;
            data.push(row(t, 1, 0, 2000.0, 1500.0));
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn repeated_glitch_samples_debounce_into_one_event() {
        // Entry into the glitch (sharp drop) and exit from it (sharp recovery) both satisfy
        // the candidate condition, 5ms apart — well under the 50ms refractory — so they must
        // collapse into one reported event, not two. Motor carries a tiny drift throughout
        // (motor_range > 0 is required by the low-signal guard) while staying under the 5%
        // stable-command threshold at every transition.
        let mut data = Vec::new();
        for i in 0..80u32 {
            let t = i as f64 / 1000.0;
            let motor = 1500.0 + i as f64 * 0.01;
            let erpm = if (20..25).contains(&i) {
                100.0 + i as f64
            } else {
                1000.0
            };
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert_eq!(results[0].events.len(), 1);
    }
}

// src/plot_functions/motor_desync.rs
