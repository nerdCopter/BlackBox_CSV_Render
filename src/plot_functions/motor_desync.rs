// src/plot_functions/motor_desync.rs

use crate::constants::{
    MOTOR_DESYNC_ERPM_REF_PERCENTILE, MOTOR_DESYNC_EVENT_REFRACTORY_S,
    MOTOR_DESYNC_HIGH_CMD_PERCENTILE, MOTOR_DESYNC_MIN_BASELINE_SAMPLES,
    MOTOR_DESYNC_MIN_ERPM_RANGE, MOTOR_DESYNC_MIN_MOTOR_RANGE, MOTOR_DESYNC_NOISE_MULTIPLIER,
    MOTOR_DESYNC_POSSIBLE_CEILING_FRACTION, MOTOR_DESYNC_POSSIBLE_HIGH_CMD_PERCENTILE,
    MOTOR_DESYNC_POSSIBLE_SUSTAIN_S, MOTOR_DESYNC_RESPONSE_FLOOR_FRACTION, MOTOR_DESYNC_SUSTAIN_S,
};
use crate::data_input::log_data::LogRowData;

/// Confidence tier of a flagged event — see `detect_motor_desync` for how each is derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesyncConfidence {
    /// This motor has enough of its own high-command flying history elsewhere in the log to
    /// compare against — the flagged window's eRPM response is a self-relative outlier
    /// against that motor's own established behavior at similarly high command.
    DeFacto,
    /// Too little high-command history exists elsewhere in the log to build the self-relative
    /// baseline `DeFacto` needs (e.g. a short flight whose only high-throttle moment is the
    /// event itself) — flagged instead against a looser, still self-relative check (this
    /// motor's own overall eRPM level), so it carries less confidence.
    Possible,
}

/// One flagged motor/eRPM anomaly.
pub struct MotorDesyncEvent {
    pub time_s: f64,
    pub confidence: DesyncConfidence,
}

/// Per-motor desync-detection summary.
pub struct MotorDesyncResult {
    pub motor_idx: usize,
    /// False when this motor's command or eRPM signal never varied enough to analyze —
    /// missing telemetry, near-constant command, or the motor never spun meaningfully.
    pub erpm_signal_available: bool,
    pub events: Vec<MotorDesyncEvent>,
}

struct MotorSamples {
    times: Vec<f64>,
    motor: Vec<f64>,
    erpm: Vec<f64>,
}

fn collect_motor_samples(log_data: &[LogRowData], motor_idx: usize) -> MotorSamples {
    let mut times = Vec::new();
    let mut motor = Vec::new();
    let mut erpm = Vec::new();
    for row in log_data {
        if let (Some(t), Some(Some(m)), Some(Some(e))) = (
            row.time_sec,
            row.motors.get(motor_idx).copied(),
            row.erpms.get(motor_idx).copied(),
        ) {
            times.push(t);
            motor.push(m);
            erpm.push(e);
        }
    }
    MotorSamples { times, motor, erpm }
}

fn percentile(sorted_vals: &[f64], p: f64) -> f64 {
    if sorted_vals.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_vals.len() as f64 * p / 100.0) as usize).min(sorted_vals.len() - 1);
    sorted_vals[idx]
}

fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn median(vals: &[f64]) -> f64 {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn population_stdev(vals: &[f64]) -> f64 {
    let m = mean(vals);
    (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64).sqrt()
}

/// Sample-index window length covering `duration_s`, given this motor's own average sample
/// interval — at least 2 samples so a window always spans a real interval.
fn window_len(times: &[f64], duration_s: f64) -> usize {
    let dt = (times[times.len() - 1] - times[0]) / (times.len() - 1) as f64;
    if dt <= 0.0 {
        return 2;
    }
    ((duration_s / dt).round() as usize).max(2)
}

/// `DeFacto` check: this motor has enough of its own high-command history elsewhere in the
/// flight to know what its eRPM normally does there. Slide a window and flag any stretch
/// where the command stays in that motor's own top quartile for the whole window while the
/// window's eRPM median falls far below, or its spread far exceeds, that established norm.
fn detect_de_facto(samples: &MotorSamples, motor_min: f64, motor_range: f64) -> Vec<f64> {
    let high_thresh = motor_min + motor_range * MOTOR_DESYNC_HIGH_CMD_PERCENTILE / 100.0;
    let baseline: Vec<f64> = samples
        .motor
        .iter()
        .zip(samples.erpm.iter())
        .filter(|(m, _)| **m >= high_thresh)
        .map(|(_, e)| *e)
        .collect();
    if baseline.len() < MOTOR_DESYNC_MIN_BASELINE_SAMPLES {
        return Vec::new();
    }
    let base_median = median(&baseline);
    let base_stdev = population_stdev(&baseline);

    let win = window_len(&samples.times, MOTOR_DESYNC_SUSTAIN_S);
    let mut events = Vec::new();
    let mut last_event_time: Option<f64> = None;
    let mut i = 0;
    while i + win <= samples.motor.len() {
        let mseg = &samples.motor[i..i + win];
        if mseg.iter().cloned().fold(f64::INFINITY, f64::min) >= high_thresh {
            let eseg = &samples.erpm[i..i + win];
            let window_median = median(eseg);
            let window_stdev = population_stdev(eseg);
            let low_response = window_median < base_median * MOTOR_DESYNC_RESPONSE_FLOOR_FRACTION;
            let noisy = base_stdev > f64::EPSILON
                && window_stdev > base_stdev * MOTOR_DESYNC_NOISE_MULTIPLIER;
            if low_response || noisy {
                let t = samples.times[i];
                if last_event_time.map_or(true, |last| t - last >= MOTOR_DESYNC_EVENT_REFRACTORY_S)
                {
                    events.push(t);
                    last_event_time = Some(t);
                }
            }
        }
        i += (win / 2).max(1);
    }
    events
}

/// `Possible` check: used only where `detect_de_facto` couldn't build a baseline (too little
/// high-command history in this flight). Falls back to a shorter window and a looser
/// self-relative reference — this motor's own 90th-percentile eRPM over the whole flight,
/// robust to a few noisy outlier samples in a way a raw max isn't. Flags a brief stretch of
/// near-ceiling command whose eRPM never approaches that reference at all.
fn detect_possible(samples: &MotorSamples, motor_min: f64, motor_range: f64) -> Vec<f64> {
    let high_thresh = motor_min + motor_range * MOTOR_DESYNC_POSSIBLE_HIGH_CMD_PERCENTILE / 100.0;
    let mut erpm_sorted = samples.erpm.clone();
    erpm_sorted.sort_by(|a, b| a.total_cmp(b));
    let erpm_ref = percentile(&erpm_sorted, MOTOR_DESYNC_ERPM_REF_PERCENTILE);
    let ceiling = erpm_ref * MOTOR_DESYNC_POSSIBLE_CEILING_FRACTION;

    let win = window_len(&samples.times, MOTOR_DESYNC_POSSIBLE_SUSTAIN_S);
    let mut events = Vec::new();
    let mut last_event_time: Option<f64> = None;
    let mut i = 0;
    while i + win <= samples.motor.len() {
        let mseg = &samples.motor[i..i + win];
        if mseg.iter().cloned().fold(f64::INFINITY, f64::min) >= high_thresh {
            let eseg = &samples.erpm[i..i + win];
            let window_max = eseg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if window_max < ceiling {
                let t = samples.times[i];
                if last_event_time.map_or(true, |last| t - last >= MOTOR_DESYNC_EVENT_REFRACTORY_S)
                {
                    events.push(t);
                    last_event_time = Some(t);
                }
            }
        }
        i += (win / 2).max(1);
    }
    events
}

/// Flags candidate motor-desync events by comparing each motor's commanded output
/// (`motor[N]`) against its eRPM telemetry (`eRPM[N]`), self-relative to that same motor's
/// own behavior elsewhere in this same flight — never a fixed cross-aircraft threshold, since
/// raw motor/eRPM units and normal spin-up/response behavior both vary by protocol, pole
/// count, and airframe.
///
/// Two confidence tiers, see `DesyncConfidence`. Calibrated and validated against three
/// confirmed real crash logs (each ending mid-tumble, log terminating at the crash) and
/// checked for false positives across roughly 500 seconds of otherwise-normal flight in
/// those same logs plus eight further real flights: `DeFacto` caught two of the three crashes
/// cleanly with zero false positives; the third crash's motor had too little other
/// high-command history in that short flight for `DeFacto`'s baseline, but was caught by
/// `Possible`, which also produced zero false positives across the same data. Still a
/// heuristic, not a certainty — cross-check any flagged time against gyro/accelerometer
/// disturbance at the same timestamp.
pub fn detect_motor_desync(log_data: &[LogRowData]) -> Vec<MotorDesyncResult> {
    let motor_count = log_data
        .iter()
        .map(|row| row.motors.len())
        .max()
        .unwrap_or(0);

    let mut results = Vec::with_capacity(motor_count);

    for motor_idx in 0..motor_count {
        let samples = collect_motor_samples(log_data, motor_idx);

        if samples.motor.len() < 2 {
            results.push(MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                events: Vec::new(),
            });
            continue;
        }

        let motor_min = samples.motor.iter().cloned().fold(f64::INFINITY, f64::min);
        let motor_max = samples
            .motor
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let erpm_min = samples.erpm.iter().cloned().fold(f64::INFINITY, f64::min);
        let erpm_max = samples
            .erpm
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let motor_range = motor_max - motor_min;
        let erpm_range = erpm_max - erpm_min;

        if motor_range < MOTOR_DESYNC_MIN_MOTOR_RANGE || erpm_range < MOTOR_DESYNC_MIN_ERPM_RANGE {
            results.push(MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                events: Vec::new(),
            });
            continue;
        }

        let de_facto = detect_de_facto(&samples, motor_min, motor_range);
        let mut events: Vec<MotorDesyncEvent> = de_facto
            .into_iter()
            .map(|t| MotorDesyncEvent {
                time_s: t,
                confidence: DesyncConfidence::DeFacto,
            })
            .collect();

        // Possible only runs where DeFacto had no baseline to work with at all — otherwise
        // its looser check would just duplicate DeFacto's own findings at lower confidence.
        let high_thresh = motor_min + motor_range * MOTOR_DESYNC_HIGH_CMD_PERCENTILE / 100.0;
        let baseline_n = samples.motor.iter().filter(|m| **m >= high_thresh).count();
        if events.is_empty() && baseline_n < MOTOR_DESYNC_MIN_BASELINE_SAMPLES {
            let possible = detect_possible(&samples, motor_min, motor_range);
            events.extend(possible.into_iter().map(|t| MotorDesyncEvent {
                time_s: t,
                confidence: DesyncConfidence::Possible,
            }));
        }
        events.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));

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

    const SAMPLE_INTERVAL_S: f64 = 0.0005; // 2kHz, typical Betaflight loop rate

    #[test]
    fn healthy_motor_with_rich_history_flags_nothing() {
        // Motor cycles through its full range repeatedly (plenty of high-command baseline),
        // eRPM tracks command proportionally throughout — no DeFacto or Possible candidate.
        let mut data = Vec::new();
        for i in 0..4000u32 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 1000.0 + (i as f64 % 1000.0);
            let erpm = motor * 1.4;
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn sustained_high_command_with_no_rpm_response_is_de_facto() {
        // Plenty of prior high-command history establishes a normal baseline (~1800 eRPM at
        // high command). A later sustained window holds command high while eRPM collapses to
        // near-zero for the whole window — the confirmed-crash signature (motor commanded to
        // max, eRPM never follows), caught with a rich baseline available.
        let mut data = Vec::new();
        let mut i = 0u32;
        for _ in 0..3000 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 1000.0 + (i as f64 % 1000.0);
            let erpm = motor * 1.4;
            data.push(row(t, 1, 0, motor, erpm));
            i += 1;
        }
        for _ in 0..200 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            data.push(row(t, 1, 0, 2000.0, 50.0));
            i += 1;
        }

        let results = detect_motor_desync(&data);
        assert_eq!(results[0].events.len(), 1);
        assert_eq!(results[0].events[0].confidence, DesyncConfidence::DeFacto);
    }

    #[test]
    fn brief_full_send_with_sparse_history_is_possible() {
        // A mostly-idle short flight whose only near-ceiling command is one brief burst with
        // no eRPM response — too little high-command history for DeFacto's baseline (mirrors
        // the real short-flight crash case this tier exists for), but the looser Possible
        // check still catches it.
        let mut data = Vec::new();
        let mut i = 0u32;
        for _ in 0..2000 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 300.0 + (i as f64 % 100.0);
            let erpm = motor * 1.2;
            data.push(row(t, 1, 0, motor, erpm));
            i += 1;
        }
        for _ in 0..19 {
            // Long enough that some stride-aligned window start lands fully inside the burst
            // (window=12 samples, stride=6 — a shorter burst can fall between stride steps
            // and never get a fully-contained window at all), but still below
            // MOTOR_DESYNC_MIN_BASELINE_SAMPLES (20): the burst must not become its own
            // degenerate self-referential "baseline", or DeFacto's guard against too-little
            // history never triggers Possible at all.
            let t = i as f64 * SAMPLE_INTERVAL_S;
            data.push(row(t, 1, 0, 2000.0, 40.0));
            i += 1;
        }
        for _ in 0..500 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 300.0 + (i as f64 % 100.0);
            let erpm = motor * 1.2;
            data.push(row(t, 1, 0, motor, erpm));
            i += 1;
        }

        let results = detect_motor_desync(&data);
        assert_eq!(results[0].events.len(), 1);
        assert_eq!(results[0].events[0].confidence, DesyncConfidence::Possible);
    }

    #[test]
    fn low_signal_motor_is_not_analyzed() {
        let mut data = Vec::new();
        for i in 0..200u32 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 200.0 + (i as f64 % 100.0) * 10.0;
            data.push(row(t, 1, 0, motor, 12.0));
        }

        let results = detect_motor_desync(&data);
        assert!(!results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn narrow_motor_range_idle_only_log_is_not_analyzed() {
        let mut data = Vec::new();
        for i in 0..200u32 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 1000.0 + (i as f64 % 50.0);
            let erpm = if i == 100 { 100.0 } else { 1000.0 };
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert!(!results[0].erpm_signal_available);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn noisy_motor_stays_within_own_baseline_variance() {
        // A motor that's naturally erratic at high command every time it gets there (its own
        // baseline stdev is already large) must not be flagged just for being noisy — DeFacto
        // compares against this motor's OWN established variance, not a fixed noise floor.
        // Regression case: a chronically-noisy-but-otherwise-healthy motor in real crash-log
        // calibration data produced exactly this pattern outside its actual failure window.
        let mut data = Vec::new();
        let noisy_pattern = [1900.0, 2050.0, 1980.0, 2100.0, 1950.0, 2030.0];
        for i in 0u32..3000 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 1000.0 + (i as f64 % 1000.0);
            let erpm = if motor >= 1750.0 {
                noisy_pattern[(i as usize) % noisy_pattern.len()]
            } else {
                motor * 1.4
            };
            data.push(row(t, 1, 0, motor, erpm));
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].events.is_empty());
    }
}

// src/plot_functions/motor_desync.rs
