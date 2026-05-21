// src/data_analysis/torque_inertia_profiler.rs
//
// Torque-Inertia Profiler
//
// Derives aircraft-specific Td targets from throttle-punch events in flight logs.
// Replaces the empirical frame-class lookup table with values measured from actual
// aircraft dynamics.
//
// Algorithm:
//   1. Detect throttle-punch events: setpoint[3] increases >= THROTTLE_PUNCH_MIN_DELTA
//      within THROTTLE_PUNCH_WINDOW_MS.
//   2. For each punch: measure peak |d(gyro[axis])/dt| in the response window.
//   3. Normalize by throttle delta: torque_inertia_ratio = peak_alpha / cmd_delta_normalized
//   4. Aggregate across all logs in the aircraft group; compute median + spread.
//   5. Td_target_ms = TORQUE_PROFILER_TD_CALC_K / sqrt((P / P_SCALE) * torque_inertia_ratio)
//
// The TORQUE_PROFILER_P_SCALE constant accounts for the unit mismatch between raw
// Betaflight/EmuFlight P gain values and the physics formula. It may require empirical
// calibration against known-good aircraft.

use crate::axis_names::AXIS_COUNT;
use crate::constants::{
    OPTIMAL_P_SECONDS_TO_MS_MULTIPLIER, THROTTLE_COMMAND_SCALE, THROTTLE_PUNCH_MIN_DELTA,
    THROTTLE_PUNCH_WINDOW_MS, THROTTLE_RESPONSE_WINDOW_MS,
    TORQUE_PROFILER_MIN_CMD_DELTA_NORMALIZED, TORQUE_PROFILER_MIN_DT_S, TORQUE_PROFILER_MIN_EVENTS,
    TORQUE_PROFILER_P_SCALE, TORQUE_PROFILER_SETTLE_SAMPLES, TORQUE_PROFILER_TD_CALC_K,
};
use crate::data_input::log_data::LogRowData;

/// Per-axis profiling result derived from aggregated throttle-punch events.
#[derive(Debug, Clone, Default)]
pub struct AxisProfile {
    /// Median torque-to-inertia ratio [deg/s² per normalized throttle unit (0–1)]
    pub torque_inertia_ratio: Option<f64>,
    /// Spread as a fraction of the median (half inter-quartile range / median).
    /// Used for tolerance estimation. None if fewer than 4 events.
    pub ratio_spread_fraction: Option<f64>,
    /// Number of valid throttle-punch events used to compute this profile.
    pub event_count: usize,
}

impl AxisProfile {
    fn from_ratios(mut ratios: Vec<f64>) -> Self {
        let n = ratios.len();
        if n == 0 {
            return Self::default();
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if n % 2 == 0 {
            (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0
        } else {
            ratios[n / 2]
        };

        if median <= 0.0 {
            return Self::default();
        }

        // Half-IQR normalised by median as a spread estimate.
        let ratio_spread_fraction = if n >= 4 {
            let q1 = ratios[n / 4];
            let q3 = ratios[(3 * n) / 4];
            let half_iqr = (q3 - q1) / 2.0;
            Some(half_iqr / median)
        } else {
            None
        };

        AxisProfile {
            torque_inertia_ratio: Some(median),
            ratio_spread_fraction,
            event_count: n,
        }
    }

    /// Compute the physics-derived Td target and tolerance for this axis given the
    /// current P gain.
    ///
    /// Returns None when fewer than TORQUE_PROFILER_MIN_EVENTS events were found or
    /// the profiling data is otherwise insufficient.
    pub fn td_target_ms(&self, current_p: u32) -> Option<(f64, f64)> {
        if self.event_count < TORQUE_PROFILER_MIN_EVENTS {
            return None;
        }
        let ratio = self.torque_inertia_ratio?;
        if ratio <= 0.0 || current_p == 0 {
            return None;
        }
        let p_effective = (current_p as f64) / TORQUE_PROFILER_P_SCALE;
        let discriminant = p_effective * ratio;
        if discriminant <= 0.0 {
            return None;
        }
        let omega_n = discriminant.sqrt();
        let td_ms = (TORQUE_PROFILER_TD_CALC_K / omega_n)
            * crate::constants::TORQUE_PROFILER_ACHIEVABILITY_FACTOR;
        if !td_ms.is_finite() || td_ms <= 0.0 {
            return None;
        }
        // Propagate ratio spread to tolerance; default 25 % when spread is unknown.
        let tolerance_ms = self
            .ratio_spread_fraction
            .map(|f| td_ms * f * 0.5)
            .unwrap_or(td_ms * 0.25)
            .max(1.0); // minimum 1 ms
        Some((td_ms, tolerance_ms))
    }
}

/// Per-aircraft profile aggregated across all logs in a group.
#[derive(Debug, Clone, Default)]
pub struct AircraftProfile {
    /// Per-axis profiles (Roll=0, Pitch=1, Yaw=2).
    pub axes: [AxisProfile; AXIS_COUNT],
}

impl AircraftProfile {
    /// Build a profile from pre-collected per-axis ratio vectors.
    pub fn from_axis_ratios(axis_ratios: [Vec<f64>; AXIS_COUNT]) -> Self {
        let [r0, r1, r2] = axis_ratios;
        AircraftProfile {
            axes: [
                AxisProfile::from_ratios(r0),
                AxisProfile::from_ratios(r1),
                AxisProfile::from_ratios(r2),
            ],
        }
    }

    /// Returns a human-readable summary for console display.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let axis_names = crate::axis_names::AXIS_NAMES;
        for (i, (profile, name)) in self
            .axes
            .iter()
            .zip(axis_names.iter())
            .enumerate()
            .take(crate::axis_names::ROLL_PITCH_AXIS_COUNT)
        {
            if let Some(ratio) = profile.torque_inertia_ratio {
                s.push_str(&format!(
                    "  {}: torque_ratio={:.0} deg/s²/cmd ({} events)\n",
                    name, ratio, profile.event_count
                ));
            } else {
                s.push_str(&format!(
                    "  {}: insufficient events ({}/{} required)\n",
                    name, profile.event_count, TORQUE_PROFILER_MIN_EVENTS
                ));
            }
            let _ = i; // suppress unused warning
        }
        s
    }
}

/// Detect throttle-punch events in a single log and return the per-axis
/// torque-inertia ratio estimates.
///
/// A throttle-punch is defined as `setpoint[3]` increasing by at least
/// `THROTTLE_PUNCH_MIN_DELTA` (in 0–1000 units) within `THROTTLE_PUNCH_WINDOW_MS`.
/// For each punch the peak angular acceleration in the following
/// `THROTTLE_RESPONSE_WINDOW_MS` window is divided by the normalised command delta
/// to yield the torque-inertia ratio for each axis.
pub fn extract_punch_ratios(log_data: &[LogRowData], sample_rate: f64) -> [Vec<f64>; AXIS_COUNT] {
    let mut axis_ratios: [Vec<f64>; AXIS_COUNT] = std::array::from_fn(|_| Vec::new());

    if log_data.len() < 10 || sample_rate <= 0.0 {
        return axis_ratios;
    }

    let punch_window = ((THROTTLE_PUNCH_WINDOW_MS / OPTIMAL_P_SECONDS_TO_MS_MULTIPLIER)
        * sample_rate)
        .ceil() as usize;
    let response_window = ((THROTTLE_RESPONSE_WINDOW_MS / OPTIMAL_P_SECONDS_TO_MS_MULTIPLIER)
        * sample_rate)
        .ceil() as usize;
    let punch_window = punch_window.max(2);
    let response_window = response_window.max(5);

    let n = log_data.len();
    let mut i = 0usize;

    while i + punch_window + response_window < n {
        let t_start = log_data[i].setpoint[3];
        let t_end = log_data[i + punch_window].setpoint[3];

        if let (Some(throttle_start), Some(throttle_end)) = (t_start, t_end) {
            let delta = throttle_end - throttle_start;
            let cmd_delta_normalized = delta / THROTTLE_COMMAND_SCALE;

            if delta >= THROTTLE_PUNCH_MIN_DELTA
                && cmd_delta_normalized >= TORQUE_PROFILER_MIN_CMD_DELTA_NORMALIZED
            {
                // Punch detected at sample i.
                let resp_start = (i + punch_window + TORQUE_PROFILER_SETTLE_SAMPLES).min(n - 1);
                let resp_end = (resp_start + response_window).min(n - 1);

                if resp_end > resp_start + 2 {
                    // Axis 2 (Yaw) is collected here for diagnostic completeness but
                    // is not used in optimal-P analysis (Roll/Pitch only).
                    for (axis, axis_ratio_vec) in axis_ratios.iter_mut().enumerate() {
                        let mut peak_alpha: f64 = 0.0;
                        for j in resp_start..resp_end {
                            if let (Some(g0), Some(g1), Some(t0), Some(t1)) = (
                                log_data[j].gyro[axis],
                                log_data[j + 1].gyro[axis],
                                log_data[j].time_sec,
                                log_data[j + 1].time_sec,
                            ) {
                                let dt = t1 - t0;
                                if dt > TORQUE_PROFILER_MIN_DT_S {
                                    let alpha = (g1 - g0).abs() / dt;
                                    if alpha > peak_alpha {
                                        peak_alpha = alpha;
                                    }
                                }
                            }
                        }
                        if peak_alpha > 0.0 {
                            let ratio = peak_alpha / cmd_delta_normalized;
                            if ratio.is_finite() && ratio > 0.0 {
                                axis_ratio_vec.push(ratio);
                            }
                        }
                    }
                }

                // Advance past the response window.
                i += punch_window + response_window;
                continue;
            }
        }
        i += 1;
    }

    axis_ratios
}

// src/data_analysis/torque_inertia_profiler.rs
