// src/plot_functions/motor_desync.rs

use crate::axis_names::AXIS_COUNT;
use crate::constants::{
    MOTOR_DESYNC_ERPM_REF_PERCENTILE, MOTOR_DESYNC_EVENT_REFRACTORY_S,
    MOTOR_DESYNC_FALLBACK_ERROR_PERCENTILE, MOTOR_DESYNC_FALLBACK_HIGH_CMD_PERCENTILE,
    MOTOR_DESYNC_FALLBACK_OSCILLATION_OVERLAP_S, MOTOR_DESYNC_FALLBACK_SUSTAIN_S,
    MOTOR_DESYNC_HIGH_CMD_PERCENTILE, MOTOR_DESYNC_MIN_BASELINE_SAMPLES,
    MOTOR_DESYNC_MIN_ERPM_RANGE, MOTOR_DESYNC_MIN_MOTOR_RANGE, MOTOR_DESYNC_NOISE_MULTIPLIER,
    MOTOR_DESYNC_POSSIBLE_CEILING_FRACTION, MOTOR_DESYNC_POSSIBLE_HIGH_CMD_PERCENTILE,
    MOTOR_DESYNC_POSSIBLE_SUSTAIN_S, MOTOR_DESYNC_RESPONSE_FLOOR_FRACTION, MOTOR_DESYNC_SUSTAIN_S,
};
use crate::data_input::log_data::LogRowData;
use crate::plot_functions::plot_motor_spectrums::MotorOscillationResult;

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
    /// No eRPM telemetry exists in this log at all (e.g. EmuFlight, or Betaflight without
    /// bidirectional DShot enabled) — this motor's own RPM response can't be checked directly.
    /// Flags a motor commanded near its own ceiling while the aircraft's rotation diverges
    /// sharply from what was actually commanded (gyro vs. setpoint), the same shape a human
    /// reviewing the traces would call "obviously fighting something" — but with no way to
    /// confirm the cause is a motor/ESC failure rather than a hard intentional maneuver or a
    /// different failure mode entirely (radio glitch, prop strike, mechanical damage). The
    /// least confident tier; see `detect_motor_desync` for calibration notes.
    Fallback,
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
    /// missing telemetry, near-constant command, or the motor never spun meaningfully. Also
    /// false when `fallback_used` is true, since no real eRPM analysis ran for any motor.
    pub erpm_signal_available: bool,
    /// True when this log has no eRPM telemetry at all and the motor+gyro/setpoint `Fallback`
    /// heuristic ran instead of the eRPM-based `DeFacto`/`Possible` checks.
    pub fallback_used: bool,
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

/// Floor-based (not linear-interpolation) percentile — `p=100.0` returns the max, `p=0.0` the
/// min. Every caller in this file uses it the same way, so the non-standard rounding is
/// internally consistent, but don't assume it matches a statistics library's `percentile`.
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

/// `DeFacto` check: this motor has enough of its own high-command history *elsewhere* in the
/// flight to know what its eRPM normally does there. "Elsewhere" is load-bearing: a contiguous
/// stretch of high command is grouped into one run, and a run is never allowed to contribute to
/// its own comparison baseline — only other runs can. Without this, a motor that never reaches
/// its own top quartile except during the anomaly itself builds a baseline entirely out of the
/// anomaly's own already-collapsed eRPM, which delays detection until the window looks even more
/// extreme than that self-contaminated reference (observed directly on a confirmed real crash:
/// the baseline's median dropped to near zero, pushed there by the crash itself, and detection
/// didn't fire until ~0.9s after the visual divergence). Returns the flagged times plus whether
/// any run had an independent (other-runs-only) baseline at all — the caller uses that to decide
/// whether `Possible` should run instead, replacing a raw sample-count check that couldn't tell
/// "genuine baseline elsewhere" apart from "this motor's only high-command run is the anomaly."
fn detect_de_facto(samples: &MotorSamples, motor_min: f64, motor_range: f64) -> (Vec<f64>, bool) {
    let high_thresh = motor_min + motor_range * MOTOR_DESYNC_HIGH_CMD_PERCENTILE / 100.0;
    let n = samples.motor.len();

    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if samples.motor[i] >= high_thresh {
            let start = i;
            while i < n && samples.motor[i] >= high_thresh {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }

    let win = window_len(&samples.times, MOTOR_DESYNC_SUSTAIN_S);
    let mut events = Vec::new();
    let mut baseline_available = false;
    let mut last_event_time: Option<f64> = None;

    for &(start, end) in &runs {
        // A run shorter than one evaluation window can never be tested itself — counting its
        // baseline availability would be meaningless and could wrongly mask a real gap in
        // coverage for the run that actually matters (e.g. a couple of sample-level flickers
        // right before a sustained crash, each too short to evaluate, otherwise "borrowing" the
        // crash's own run as their baseline and reporting a baseline as available overall).
        if end - start < win {
            continue;
        }
        let baseline: Vec<f64> = runs
            .iter()
            .filter(|&&(s, _)| s != start)
            .flat_map(|&(s, e)| samples.erpm[s..e].iter().copied())
            .collect();
        if baseline.len() < MOTOR_DESYNC_MIN_BASELINE_SAMPLES {
            continue;
        }
        baseline_available = true;
        let base_median = median(&baseline);
        let base_stdev = population_stdev(&baseline);

        let mut i = start;
        while i + win <= end {
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
            i += (win / 2).max(1);
        }
    }
    (events, baseline_available)
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

/// `Fallback` check: used only when this log has no eRPM telemetry anywhere at all. Compares
/// each row's tracking error — how far actual rotation (gyro) diverges from commanded rotation
/// (setpoint), on whichever of roll/pitch/yaw diverges most — against this same flight's own
/// error distribution, self-relative like every other tier here. A window is flagged only when
/// that error is a rare, sustained outlier for this specific flight (not just a single-sample
/// overshoot, which ordinary punchy flying produces routinely) AND at least one motor is
/// commanded near its own ceiling in that same window — the flight controller visibly fighting
/// something. Returns `(motor_idx, time_s)` pairs, one per motor that was high during a flagged
/// window (a window can implicate more than one motor).
///
/// Calibrated against the same three confirmed crash logs used for `DeFacto`/`Possible` (using
/// only their motor/gyro/setpoint columns, as if eRPM didn't exist) plus twelve further real
/// flights, including one containing a violent uncommanded rotation the pilot reported as a
/// confirmed desync. Caught all of these. Also produced a handful of residual false positives
/// this session couldn't fully eliminate — notably on a log already established as chronic
/// tune/mechanical oscillation, not desync (see `plot_motor_spectrums.rs`'s windowed check) —
/// which is why this is its own least-confident tier, not folded into `Possible`.
fn detect_control_loss_fallback(log_data: &[LogRowData]) -> Vec<(usize, f64)> {
    let motor_count = log_data
        .iter()
        .map(|row| row.motors.len())
        .max()
        .unwrap_or(0);

    let mut times = Vec::new();
    // Per-motor Option, not a collapsed Vec<f64> — a single motor's momentary telemetry gap
    // must not discard this row's tracking-error sample (and every OTHER motor's data in it)
    // wholesale; each motor's own high-command check below only looks at its own valid samples.
    let mut motor_rows: Vec<Vec<Option<f64>>> = Vec::new();
    let mut err_mag = Vec::new();

    for row in log_data {
        let Some(t) = row.time_sec else { continue };
        let mut err: f64 = 0.0;
        let mut have_axis = false;
        for axis in 0..AXIS_COUNT {
            if let (Some(g), Some(s)) = (row.gyro[axis], row.setpoint[axis]) {
                err = err.max((g - s).abs());
                have_axis = true;
            }
        }
        if !have_axis {
            continue;
        }
        times.push(t);
        motor_rows.push(row.motors.clone());
        err_mag.push(err);
    }

    if times.len() < 2 {
        return Vec::new();
    }

    let mut err_sorted = err_mag.clone();
    err_sorted.sort_by(|a, b| a.total_cmp(b));
    // A constant (or all-zero) tracking error has no outlier to find — percentile() would
    // return that same constant as err_thresh, and `err_avg >= err_thresh` would then hold for
    // every window, false-flagging perfectly normal flying (or a ground test with no motion).
    if err_sorted
        .first()
        .zip(err_sorted.last())
        .is_some_and(|(min, max)| (max - min).abs() <= f64::EPSILON)
    {
        return Vec::new();
    }
    let err_thresh = percentile(&err_sorted, MOTOR_DESYNC_FALLBACK_ERROR_PERCENTILE);

    let mut high_thresh = vec![f64::INFINITY; motor_count];
    for (k, thresh) in high_thresh.iter_mut().enumerate().take(motor_count) {
        let vals: Vec<f64> = motor_rows
            .iter()
            .filter_map(|m| m.get(k).copied().flatten())
            .collect();
        if vals.is_empty() {
            continue;
        }
        let mmin = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let mmax = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if mmax - mmin >= MOTOR_DESYNC_MIN_MOTOR_RANGE {
            *thresh = mmin + (mmax - mmin) * MOTOR_DESYNC_FALLBACK_HIGH_CMD_PERCENTILE / 100.0;
        }
    }

    let win = window_len(&times, MOTOR_DESYNC_FALLBACK_SUSTAIN_S);
    let mut last_event_time: Vec<Option<f64>> = vec![None; motor_count];
    let mut events = Vec::new();
    let mut i = 0;
    while i + win <= times.len() {
        let eseg = &err_mag[i..i + win];
        let err_avg = mean(eseg);
        if err_avg >= err_thresh {
            for k in 0..motor_count {
                if high_thresh[k].is_infinite() {
                    continue;
                }
                let motor_high = (i..i + win).any(|j| {
                    motor_rows[j]
                        .get(k)
                        .copied()
                        .flatten()
                        .is_some_and(|v| v >= high_thresh[k])
                });
                if motor_high {
                    let t = times[i];
                    if last_event_time[k]
                        .map_or(true, |last| t - last >= MOTOR_DESYNC_EVENT_REFRACTORY_S)
                    {
                        events.push((k, t));
                        last_event_time[k] = Some(t);
                    }
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
/// Two eRPM-based confidence tiers plus a motor+gyro/setpoint `Fallback` used only when this
/// log has no eRPM telemetry at all — see `DesyncConfidence`. The eRPM tiers are calibrated
/// and validated against three confirmed real crash logs (each ending mid-tumble, log
/// terminating at the crash) and checked for false positives across roughly 500 seconds of
/// otherwise-normal flight in those same logs plus eight further real flights: `DeFacto`
/// caught two of the three crashes cleanly with zero false positives; the third crash's motor
/// had too little other high-command history in that short flight for `DeFacto`'s baseline,
/// but was caught by `Possible`, which also produced zero false positives across the same
/// data. Still a heuristic, not a certainty — cross-check any flagged time against
/// gyro/accelerometer disturbance at the same timestamp.
pub fn detect_motor_desync(log_data: &[LogRowData]) -> Vec<MotorDesyncResult> {
    let motor_count = log_data
        .iter()
        .map(|row| row.motors.len())
        .max()
        .unwrap_or(0);

    let has_any_erpm = log_data
        .iter()
        .any(|row| row.erpms.iter().any(|e| e.is_some()));

    if !has_any_erpm && motor_count > 0 {
        println!(
            "  ⚠️  No eRPM telemetry in this log — using motor+gyro/setpoint fallback for desync detection (lower confidence, cannot confirm RPM response)"
        );
        let fallback_events = detect_control_loss_fallback(log_data);
        let mut per_motor: Vec<Vec<MotorDesyncEvent>> =
            (0..motor_count).map(|_| Vec::new()).collect();
        for (motor_idx, t) in fallback_events {
            per_motor[motor_idx].push(MotorDesyncEvent {
                time_s: t,
                confidence: DesyncConfidence::Fallback,
            });
        }
        return per_motor
            .into_iter()
            .enumerate()
            .map(|(motor_idx, events)| MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                fallback_used: true,
                events,
            })
            .collect();
    }

    let mut results = Vec::with_capacity(motor_count);

    for motor_idx in 0..motor_count {
        let samples = collect_motor_samples(log_data, motor_idx);

        if samples.motor.len() < 2 {
            results.push(MotorDesyncResult {
                motor_idx,
                erpm_signal_available: false,
                fallback_used: false,
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
                fallback_used: false,
                events: Vec::new(),
            });
            continue;
        }

        let (de_facto, de_facto_baseline_available) =
            detect_de_facto(&samples, motor_min, motor_range);
        let mut events: Vec<MotorDesyncEvent> = de_facto
            .into_iter()
            .map(|t| MotorDesyncEvent {
                time_s: t,
                confidence: DesyncConfidence::DeFacto,
            })
            .collect();

        // Possible only runs where DeFacto had no independent baseline to work with at all —
        // otherwise its looser check would just duplicate DeFacto's own findings at lower
        // confidence.
        if events.is_empty() && !de_facto_baseline_available {
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
            fallback_used: false,
            events,
        });
    }

    results
}

/// Finds `Fallback`-tier events that land within `MOTOR_DESYNC_FALLBACK_OSCILLATION_OVERLAP_S`
/// of a same-motor Motor Oscillation detection (`plot_motor_spectrums.rs`) — a known, confirmed
/// cause of `Fallback` false positives (chronic tune/mechanical resonance can also produce a
/// large gyro/setpoint tracking error, with no way for `Fallback` to tell that apart from a
/// real desync). Returns `(motor_idx, time_s)` pairs to caveat in the console and report,
/// without suppressing or altering the underlying event — `Fallback` stays as-is by design;
/// this only flags when its known blind spot may apply.
pub fn fallback_oscillation_overlaps(
    desync_results: &[MotorDesyncResult],
    motor_results: &[MotorOscillationResult],
) -> Vec<(usize, f64)> {
    let mut overlaps = Vec::new();
    for desync in desync_results {
        let Some(osc) = motor_results
            .iter()
            .find(|o| o.motor_idx == desync.motor_idx)
        else {
            continue;
        };
        if !osc.oscillation_detected {
            continue;
        }
        let Some(osc_time) = osc.event_time_s else {
            continue;
        };
        for event in &desync.events {
            if event.confidence == DesyncConfidence::Fallback
                && (event.time_s - osc_time).abs() <= MOTOR_DESYNC_FALLBACK_OSCILLATION_OVERLAP_S
            {
                overlaps.push((desync.motor_idx, event.time_s));
            }
        }
    }
    overlaps
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

    fn fallback_row(time_s: f64, motor: f64, gyro: [f64; 3], setpoint: [f64; 3]) -> LogRowData {
        let mut r = LogRowData {
            time_sec: Some(time_s),
            ..Default::default()
        };
        r.motors = vec![Some(motor)]; // single motor is enough to exercise the fallback path
        r.erpms = Vec::new(); // no eRPM columns at all -- this is what selects Fallback
        for axis in 0..3 {
            r.gyro[axis] = Some(gyro[axis]);
            r.setpoint[axis] = Some(setpoint[axis]);
        }
        r
    }

    #[test]
    fn uncommanded_rotation_with_maxed_motor_is_fallback() {
        // Long, calm baseline (motor cycling normally, gyro tightly tracking setpoint)
        // establishes a tight self-relative tracking-error distribution. A later sustained
        // window holds the motor near its own ceiling while gyro diverges sharply from an
        // unchanged (near-zero) setpoint — rotation the pilot never commanded, the actual
        // no-eRPM crash signature this tier targets.
        let mut data = Vec::new();
        let mut i = 0u32;
        for _ in 0..20_000 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 500.0 + (i as f64 % 1000.0);
            let sp = (i as f64 % 20.0) - 10.0; // small, bounded commanded rate
            data.push(fallback_row(t, motor, [sp + 2.0, sp, sp], [sp, sp, sp]));
            i += 1;
        }
        for _ in 0..320 {
            // > FALLBACK window length (0.15s / 0.0005s = 300 samples)
            let t = i as f64 * SAMPLE_INTERVAL_S;
            data.push(fallback_row(
                t,
                2000.0,
                [500.0, 500.0, 500.0],
                [0.0, 0.0, 0.0],
            ));
            i += 1;
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].fallback_used);
        assert!(!results[0].erpm_signal_available);
        assert_eq!(results[0].events.len(), 1);
        assert_eq!(results[0].events[0].confidence, DesyncConfidence::Fallback);
    }

    #[test]
    fn commanded_flip_with_maxed_motor_is_not_flagged() {
        // Same maxed-motor event as above, but the aircraft's actual rotation matches what was
        // commanded (setpoint tracks gyro closely) — a real, intentional hard flip, not a
        // divergence. Tracking error never becomes a self-relative outlier, so this must not
        // be flagged: Fallback distinguishes "fighting something uncommanded" from "doing
        // exactly what was asked, aggressively."
        let mut data = Vec::new();
        let mut i = 0u32;
        for _ in 0..20_000 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 500.0 + (i as f64 % 1000.0);
            let sp = (i as f64 % 20.0) - 10.0;
            data.push(fallback_row(t, motor, [sp + 2.0, sp, sp], [sp, sp, sp]));
            i += 1;
        }
        for _ in 0..320 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            data.push(fallback_row(
                t,
                2000.0,
                [500.0, 500.0, 500.0],
                [500.0, 500.0, 500.0],
            ));
            i += 1;
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].fallback_used);
        assert!(results[0].events.is_empty());
    }

    #[test]
    fn constant_tracking_error_is_never_flagged_as_fallback() {
        // Gyro exactly equals setpoint on every single sample (tracking error is a constant
        // zero throughout), including while the motor is repeatedly commanded to its own
        // ceiling. A zero-range error distribution has no outlier to find: percentile() would
        // return that same constant, and err_avg >= err_thresh would then hold for every
        // window, false-flagging perfectly normal (or perfectly idle) flying.
        let mut data = Vec::new();
        for i in 0u32..20_320 {
            let t = i as f64 * SAMPLE_INTERVAL_S;
            let motor = 500.0 + (i as f64 % 1600.0); // cycles up to its own ceiling repeatedly
            let sp = (i as f64 % 20.0) - 10.0;
            data.push(fallback_row(t, motor, [sp, sp, sp], [sp, sp, sp]));
        }

        let results = detect_motor_desync(&data);
        assert!(results[0].fallback_used);
        assert!(results[0].events.is_empty());
    }

    fn osc_result(
        motor_idx: usize,
        detected: bool,
        event_time_s: Option<f64>,
    ) -> MotorOscillationResult {
        MotorOscillationResult {
            motor_idx,
            max_amplitude: None,
            oscillation_detected: detected,
            peak_in_range: None,
            avg_in_range: None,
            event_time_s,
        }
    }

    fn desync_result(motor_idx: usize, events: Vec<MotorDesyncEvent>) -> MotorDesyncResult {
        MotorDesyncResult {
            motor_idx,
            erpm_signal_available: false,
            fallback_used: true,
            events,
        }
    }

    #[test]
    fn fallback_near_oscillation_is_flagged_as_overlap() {
        let desync = vec![desync_result(
            0,
            vec![MotorDesyncEvent {
                time_s: 44.94,
                confidence: DesyncConfidence::Fallback,
            }],
        )];
        let osc = vec![osc_result(0, true, Some(45.5))]; // within the 2.0s overlap tolerance

        let overlaps = fallback_oscillation_overlaps(&desync, &osc);
        assert_eq!(overlaps, vec![(0, 44.94)]);
    }

    #[test]
    fn fallback_far_from_oscillation_is_not_flagged() {
        let desync = vec![desync_result(
            0,
            vec![MotorDesyncEvent {
                time_s: 44.94,
                confidence: DesyncConfidence::Fallback,
            }],
        )];
        let osc = vec![osc_result(0, true, Some(120.0))]; // far outside tolerance

        assert!(fallback_oscillation_overlaps(&desync, &osc).is_empty());
    }

    #[test]
    fn fallback_with_no_oscillation_detected_is_not_flagged() {
        let desync = vec![desync_result(
            0,
            vec![MotorDesyncEvent {
                time_s: 44.94,
                confidence: DesyncConfidence::Fallback,
            }],
        )];
        let osc = vec![osc_result(0, false, None)]; // this motor's own spectrum never crossed the bar

        assert!(fallback_oscillation_overlaps(&desync, &osc).is_empty());
    }
}

// src/plot_functions/motor_desync.rs
