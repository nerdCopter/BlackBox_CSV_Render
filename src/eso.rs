// src/eso.rs
// 2nd-order Linear Extended State Observer (LESO) for flight controller blackbox data.
// Implements discrete Euler-forward LESO simulation with argmin GoldenSectionSearch for the
// optimal observer bandwidth (omega_0) using PID sum as control input and filtered
// gyro as measured output.
//
// Cost function: N-step-ahead open-loop prediction MSE (unimodal objective).
// After a correction-phase warm-up, the observer state at each sample is propagated
// ESO_N_AHEAD_STEPS forward WITHOUT correction; the prediction is compared to the actual
// measurement. This objective is U-shaped: too-low omega_0 leaves f_hat stale (poor
// prediction), too-high omega_0 amplifies noise into f_hat (also poor prediction).

use std::error::Error;

use argmin::core::{CostFunction, Executor};
use argmin::solver::goldensectionsearch::GoldenSectionSearch;

use crate::constants::{
    ESO_B0_MIN_CONTROL_THRESHOLD, ESO_DEFAULT_B0, ESO_GSS_MAX_ITER, ESO_GSS_TOLERANCE,
    ESO_N_AHEAD_STEPS, ESO_OMEGA0_MAX, ESO_OMEGA0_MIN, ESO_WARMUP_FRACTION,
};
use crate::data_input::log_data::LogRowData;

/// Configuration for a single-axis ESO optimization run.
#[derive(Debug, Clone)]
pub struct EsoConfig {
    /// Control effectiveness (scales PID sum to angular acceleration). Default: 1.0.
    pub b0: f64,
    /// Observer bandwidth search lower bound (rad/s).
    pub omega0_min: f64,
    /// Observer bandwidth search upper bound (rad/s).
    pub omega0_max: f64,
}

impl Default for EsoConfig {
    fn default() -> Self {
        Self {
            b0: ESO_DEFAULT_B0,
            omega0_min: ESO_OMEGA0_MIN,
            omega0_max: ESO_OMEGA0_MAX,
        }
    }
}

/// Result of a single-axis ESO bandwidth optimization.
#[derive(Debug, Clone)]
pub struct EsoResult {
    /// Axis index (0=Roll, 1=Pitch, 2=Yaw).
    #[allow(dead_code)]
    pub axis: usize,
    /// Optimal observer bandwidth in rad/s.
    pub omega0_opt: f64,
    /// Observer gain beta1 = 2 * omega0_opt.
    pub beta1: f64,
    /// Observer gain beta2 = omega0_opt^2.
    pub beta2: f64,
    /// Control effectiveness used.
    pub b0: f64,
    /// True when b0 was estimated from data; false when provided by the user.
    pub b0_auto: bool,
    /// N-step-ahead prediction MSE at the optimal omega0.
    pub mse: f64,
    /// Number of samples used in the optimization.
    pub sample_count: usize,
    /// Timestamps (seconds) aligned to the trace data.
    pub timestamps: Vec<f64>,
    /// Measured angular rate (filtered gyro) used for optimization.
    pub omega_meas_trace: Vec<f64>,
    /// omega_hat trace from final simulation with optimal gains.
    pub omega_hat_trace: Vec<f64>,
    /// f_hat (disturbance estimate) trace from final simulation.
    pub f_hat_trace: Vec<f64>,
}

/// Compute 2nd-order bandwidth-parameterized LESO gains from omega_0.
/// Returns (beta1, beta2): beta1 = 2*omega0, beta2 = omega0^2.
fn leso2_gains(omega0: f64) -> (f64, f64) {
    (2.0 * omega0, omega0 * omega0)
}

/// Simulate 2nd-order discrete LESO (Euler forward) and return (omega_hat, f_hat) traces.
///
/// Discrete update at each step k:
///   e          = omega_meas[k] - omega_hat
///   omega_hat += Ts * (f_hat + b0 * u[k] + beta1 * e)
///   f_hat     += Ts * (beta2 * e)
///
/// # Arguments
/// * `omega_meas` - Measured angular rate (deg/s, filtered gyro).
/// * `u` - Control input (PID sum: P + I + D + F per axis).
/// * `ts` - Sample period in seconds (1 / sample_rate).
/// * `omega0` - Observer bandwidth (rad/s).
/// * `b0` - Control effectiveness.
fn simulate_leso2(
    omega_meas: &[f64],
    u: &[f64],
    ts: f64,
    omega0: f64,
    b0: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (beta1, beta2) = leso2_gains(omega0);
    let n = omega_meas.len().min(u.len());

    let mut omega_hat = omega_meas.first().copied().unwrap_or(0.0);
    let mut f_hat = 0.0_f64;

    let mut omega_hats = Vec::with_capacity(n);
    let mut f_hats = Vec::with_capacity(n);

    for k in 0..n {
        omega_hats.push(omega_hat);
        f_hats.push(f_hat);
        let e = omega_meas[k] - omega_hat;
        omega_hat += ts * (f_hat + b0 * u[k] + beta1 * e);
        f_hat += ts * (beta2 * e);
    }
    (omega_hats, f_hats)
}

/// Compute the N-step-ahead open-loop prediction MSE for a given omega_0.
///
/// The observer runs with full correction on all data (first pass) to obtain per-sample
/// states. Then for each sample after the warm-up fraction, the state is propagated
/// ESO_N_AHEAD_STEPS forward open-loop (no correction — f_hat frozen) and compared to
/// the actual measurement at k + N. This creates a unimodal cost:
///   - Low omega_0: f_hat is stale → poor N-step prediction.
///   - High omega_0: noise amplified into f_hat → poor N-step prediction.
///   - Optimal omega_0: balanced disturbance estimation → best N-step prediction.
fn nstep_prediction_mse(omega_meas: &[f64], u: &[f64], ts: f64, omega0: f64, b0: f64) -> f64 {
    let n = omega_meas.len().min(u.len());
    if n <= ESO_N_AHEAD_STEPS + 1 {
        return f64::INFINITY;
    }
    let (beta1, beta2) = leso2_gains(omega0);

    // First pass: run observer with correction to capture states at each sample.
    let mut omega_hat_states = vec![0.0_f64; n];
    let mut f_hat_states = vec![0.0_f64; n];
    let mut omega_hat = omega_meas[0];
    let mut f_hat = 0.0_f64;
    for k in 0..n {
        omega_hat_states[k] = omega_hat;
        f_hat_states[k] = f_hat;
        let e = omega_meas[k] - omega_hat;
        omega_hat += ts * (f_hat + b0 * u[k] + beta1 * e);
        f_hat += ts * (beta2 * e);
    }

    // Warm-up: skip initial fraction to let the observer states converge.
    let warmup = ((n as f64 * ESO_WARMUP_FRACTION) as usize).max(1);
    let end = n.saturating_sub(ESO_N_AHEAD_STEPS);
    if warmup >= end {
        return f64::INFINITY;
    }

    // Second pass: N-step open-loop prediction from each warm sample.
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for k in warmup..end {
        let mut omega_pred = omega_hat_states[k];
        let f_pred = f_hat_states[k]; // frozen — no correction in open-loop propagation
        for j in 0..ESO_N_AHEAD_STEPS {
            omega_pred += ts * (f_pred + b0 * u[k + j]);
        }
        sum_sq += (omega_pred - omega_meas[k + ESO_N_AHEAD_STEPS]).powi(2);
        count += 1;
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum_sq / count as f64
    }
}

/// argmin CostFunction wrapper for single-axis ESO bandwidth search.
struct EsoCostFn<'a> {
    omega_meas: &'a [f64],
    u: &'a [f64],
    ts: f64,
    b0: f64,
}

impl CostFunction for EsoCostFn<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, omega0: &f64) -> Result<f64, argmin::core::Error> {
        Ok(nstep_prediction_mse(
            self.omega_meas,
            self.u,
            self.ts,
            *omega0,
            self.b0,
        ))
    }
}

/// Estimate control effectiveness b0 via ordinary least squares on rate derivative increments.
///
/// QuickFlash's guidance: set beta1/beta2 → 0 (no observer correction), find b0 that
/// minimises prediction error so "it's pretty much just b0 doing the job".
///
/// With correction gains = 0 the LESO update reduces to:
///   ω[k+1] − ω[k] ≈ Ts · b0 · u[k]
///
/// OLS closed form: b0 = Σ(u[k] · Δω[k]) / (Ts · Σ(u[k]²))
///
/// Only samples where |u[k]| ≥ ESO_B0_MIN_CONTROL_THRESHOLD are included to avoid
/// numerical issues from near-zero control inputs.
/// Returns None when fewer than 10 valid samples are available or when the estimate
/// is non-positive (which would indicate a mis-matched control law).
fn estimate_b0(omega_meas: &[f64], u: &[f64], ts: f64) -> Option<f64> {
    let n = omega_meas.len().min(u.len()).saturating_sub(1);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut count = 0usize;

    for k in 0..n {
        if u[k].abs() < ESO_B0_MIN_CONTROL_THRESHOLD {
            continue;
        }
        let delta_omega = omega_meas[k + 1] - omega_meas[k];
        num += u[k] * delta_omega;
        den += u[k] * u[k] * ts;
        count += 1;
    }

    if count < 10 || den.abs() < 1e-12 {
        return None;
    }
    let b0 = num / den;
    if b0.is_finite() && b0.abs() > 1e-9 {
        Some(b0)
    } else {
        None
    }
}

/// Extract gyro measurements, PID sum, and timestamps for an axis from log data.
/// Rows with missing gyro are skipped; PID terms default to 0.0 if absent.
/// Returns (omega_meas, pid_sum, timestamps_sec) or None when fewer than 2 samples are available.
fn extract_axis_data(
    log_data: &[LogRowData],
    axis: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut omega_meas = Vec::with_capacity(log_data.len());
    let mut pid_sum = Vec::with_capacity(log_data.len());
    let mut timestamps = Vec::with_capacity(log_data.len());

    for row in log_data {
        if let Some(gyro) = row.gyro[axis] {
            let p = row.p_term[axis].unwrap_or(0.0);
            let i_val = row.i_term[axis].unwrap_or(0.0);
            let d = row.d_term[axis].unwrap_or(0.0);
            let f_val = row.f_term[axis].unwrap_or(0.0);
            omega_meas.push(gyro);
            pid_sum.push(p + i_val + d + f_val);
            timestamps.push(row.time_sec.unwrap_or(0.0));
        }
    }

    if omega_meas.len() < 2 {
        return None;
    }
    Some((omega_meas, pid_sum, timestamps))
}

/// Run ESO gain optimization for a single axis using argmin GoldenSectionSearch on omega_0.
///
/// The search is constrained to omega_0 < sample_rate / 3 for discrete-time stability.
/// The cost function is N-step-ahead open-loop prediction MSE, which is unimodal.
///
/// # Arguments
/// * `log_data` - Parsed blackbox log rows.
/// * `sample_rate` - Loop rate in Hz.
/// * `axis` - Axis index (0=Roll, 1=Pitch, 2=Yaw).
/// * `config` - ESO configuration (b0 and omega_0 search bounds).
pub fn run_eso_optimization(
    log_data: &[LogRowData],
    sample_rate: f64,
    axis: usize,
    config: &EsoConfig,
) -> Result<EsoResult, Box<dyn Error>> {
    let (omega_meas, pid_sum, timestamps) = extract_axis_data(log_data, axis)
        .ok_or("Insufficient data for ESO optimization (fewer than 2 usable samples)")?;

    // Enforce discrete-time stability: omega_0 < sample_rate / 3
    let omega0_max_stable = (sample_rate / 3.0).min(config.omega0_max);
    if omega0_max_stable <= config.omega0_min {
        return Err(format!(
            "Sample rate {:.1} Hz too low for ESO search (need > {:.1} Hz)",
            sample_rate,
            config.omega0_min * 3.0
        )
        .into());
    }

    let ts = 1.0 / sample_rate;

    // Stage 1: estimate b0 from data via OLS on rate derivatives (QuickFlash guidance).
    // If the user explicitly provided a non-default b0 via --eso-b0, respect it.
    let (b0, b0_auto) = if (config.b0 - ESO_DEFAULT_B0).abs() < 1e-12 {
        match estimate_b0(&omega_meas, &pid_sum, ts) {
            Some(estimated) => (estimated, true),
            None => (config.b0, false),
        }
    } else {
        (config.b0, false)
    };

    let problem = EsoCostFn {
        omega_meas: &omega_meas,
        u: &pid_sum,
        ts,
        b0,
    };

    let solver = GoldenSectionSearch::new(config.omega0_min, omega0_max_stable)
        .map_err(|e| -> Box<dyn Error> { format!("argmin GSS init: {e}").into() })?
        .with_tolerance(ESO_GSS_TOLERANCE)
        .map_err(|e| -> Box<dyn Error> { format!("argmin GSS tolerance: {e}").into() })?;

    let initial = (config.omega0_min + omega0_max_stable) / 2.0;

    let run_result = Executor::new(problem, solver)
        .configure(|state| state.param(initial).max_iters(ESO_GSS_MAX_ITER))
        .run()
        .map_err(|e| -> Box<dyn Error> { format!("argmin optimization: {e}").into() })?;

    let omega0_opt = run_result
        .state()
        .best_param
        .ok_or("ESO optimization returned no solution")?;
    let mse = run_result.state().best_cost;

    let (beta1, beta2) = leso2_gains(omega0_opt);

    // Final simulation with optimal gains to produce trace data for plotting.
    let (omega_hat_trace, f_hat_trace) = simulate_leso2(&omega_meas, &pid_sum, ts, omega0_opt, b0);

    Ok(EsoResult {
        axis,
        omega0_opt,
        beta1,
        beta2,
        b0,
        b0_auto,
        mse,
        sample_count: omega_meas.len(),
        timestamps,
        omega_meas_trace: omega_meas,
        omega_hat_trace,
        f_hat_trace,
    })
}
