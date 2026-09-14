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

use crate::axis_names::AXIS_COUNT;
use crate::constants::{
    ESO_B0_ESTIMATE_MIN_POSITIVE, ESO_B0_MIN_CONTROL_THRESHOLD, ESO_B0_MIN_OLS_SAMPLES,
    ESO_DEFAULT_B0, ESO_GSS_MAX_ITER, ESO_GSS_TOLERANCE, ESO_N_AHEAD_STEPS, ESO_OMEGA0_MAX,
    ESO_OMEGA0_MIN, ESO_OMEGA0_STABILITY_RATIO, ESO_WARMUP_FRACTION, VALUE_EPSILON,
};
use crate::data_input::log_data::LogRowData;

/// Configuration for a single-axis ESO optimization run.
#[derive(Debug, Clone)]
pub struct EsoConfig {
    /// Control effectiveness (scales PID sum to angular acceleration). Default: 1.0.
    pub b0: f64,
    /// True when `b0` was explicitly supplied by the user via `--eso-b0`.
    /// When false, `run_eso_optimization` will attempt OLS auto-estimation.
    pub b0_user_override: bool,
    /// Observer bandwidth search lower bound (rad/s).
    pub omega0_min: f64,
    /// Observer bandwidth search upper bound (rad/s).
    pub omega0_max: f64,
}

impl Default for EsoConfig {
    fn default() -> Self {
        Self {
            b0: ESO_DEFAULT_B0,
            b0_user_override: false,
            omega0_min: ESO_OMEGA0_MIN,
            omega0_max: ESO_OMEGA0_MAX,
        }
    }
}

/// Where a `EsoResult`'s `b0` value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum B0Source {
    /// Explicitly supplied via `--eso-b0`.
    UserSupplied,
    /// Estimated from data via OLS.
    AutoEstimated,
    /// OLS estimation failed or was rejected; fell back to `ESO_DEFAULT_B0`.
    DefaultFallback,
}

impl B0Source {
    pub fn label(self) -> &'static str {
        match self {
            B0Source::UserSupplied => "user-supplied",
            B0Source::AutoEstimated => "auto-estimated",
            B0Source::DefaultFallback => "default fallback",
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
    /// Where `b0` came from: user override, OLS estimate, or default fallback.
    pub b0_source: B0Source,
    /// N-step-ahead prediction MSE at the optimal omega0.
    pub mse: f64,
    /// True when omega0_opt is at the search ceiling (result may not be the true optimum).
    pub at_ceiling: bool,
    /// Number of samples used in the optimization.
    #[allow(dead_code)]
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
        let e = omega_meas[k] - omega_hat;
        omega_hat += ts * (f_hat + b0 * u[k] + beta1 * e);
        f_hat += ts * (beta2 * e);
        // Store the state *after* incorporating omega_meas[k], so the open-loop forecast
        // below starts from the freshest correction rather than one sample stale.
        omega_hat_states[k] = omega_hat;
        f_hat_states[k] = f_hat;
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
                                      // omega_hat_states[k] already incorporates sample k, so only k+1..k+N-1 remain.
        for j in 1..ESO_N_AHEAD_STEPS {
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
/// Returns None when fewer than ESO_B0_MIN_OLS_SAMPLES valid samples are available,
/// when the denominator is near-zero, or when the estimate is not strictly positive
/// (a negative estimate indicates an inverted sign convention between u and the gyro axis).
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

    if count < ESO_B0_MIN_OLS_SAMPLES || den.abs() < VALUE_EPSILON {
        return None;
    }
    let b0 = num / den;
    if b0.is_finite() && b0 > ESO_B0_ESTIMATE_MIN_POSITIVE {
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
    if axis >= AXIS_COUNT {
        return Err(format!("Invalid axis index {axis}; expected 0..{}", AXIS_COUNT - 1).into());
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(format!("Invalid sample rate: {sample_rate}").into());
    }
    if !config.b0.is_finite()
        || !config.omega0_min.is_finite()
        || !config.omega0_max.is_finite()
        || config.omega0_min <= 0.0
        || config.omega0_min >= config.omega0_max
    {
        return Err("Invalid ESO configuration".into());
    }

    let (omega_meas, pid_sum, timestamps) = extract_axis_data(log_data, axis)
        .ok_or("Insufficient data for ESO optimization (fewer than 2 usable samples)")?;

    if !pid_sum
        .iter()
        .any(|u| u.is_finite() && u.abs() > VALUE_EPSILON)
    {
        return Err("Insufficient control-input excitation for ESO optimization".into());
    }

    // Enforce discrete-time stability: omega_0 < sample_rate / ESO_OMEGA0_STABILITY_RATIO
    let omega0_max_stable = (sample_rate / ESO_OMEGA0_STABILITY_RATIO).min(config.omega0_max);
    if omega0_max_stable <= config.omega0_min {
        return Err(format!(
            "Sample rate {:.1} Hz too low for ESO search (need > {:.1} Hz)",
            sample_rate,
            config.omega0_min * ESO_OMEGA0_STABILITY_RATIO
        )
        .into());
    }

    let ts = 1.0 / sample_rate;

    // Stage 1: estimate b0 from data via OLS on rate derivatives (QuickFlash guidance).
    // If the user explicitly provided b0 via --eso-b0 (b0_user_override = true), respect it.
    let (b0, b0_source) = if config.b0_user_override {
        (config.b0, B0Source::UserSupplied)
    } else {
        match estimate_b0(&omega_meas, &pid_sum, ts) {
            Some(estimated) => (estimated, B0Source::AutoEstimated),
            None => (config.b0, B0Source::DefaultFallback),
        }
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
    let at_ceiling = omega0_opt >= omega0_max_stable - ESO_GSS_TOLERANCE;

    let (beta1, beta2) = leso2_gains(omega0_opt);

    // Final simulation with optimal gains to produce trace data for plotting.
    let (omega_hat_trace, f_hat_trace) = simulate_leso2(&omega_meas, &pid_sum, ts, omega0_opt, b0);

    Ok(EsoResult {
        axis,
        omega0_opt,
        beta1,
        beta2,
        b0,
        b0_source,
        mse,
        at_ceiling,
        sample_count: omega_meas.len(),
        timestamps,
        omega_meas_trace: omega_meas,
        omega_hat_trace,
        f_hat_trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SAMPLE_RATE: f64 = 2000.0;

    fn row_with_gyro_and_pid(gyro: f64, pid_sum: f64, t: f64) -> LogRowData {
        LogRowData {
            time_sec: Some(t),
            gyro: [Some(gyro), None, None],
            p_term: [Some(pid_sum), None, None],
            ..Default::default()
        }
    }

    // --- leso2_gains: pure formula ---

    #[test]
    fn leso2_gains_matches_bandwidth_parameterisation() {
        for omega0 in [1.0, 50.0, 123.4, 500.0] {
            let (beta1, beta2) = leso2_gains(omega0);
            assert_eq!(beta1, 2.0 * omega0);
            assert_eq!(beta2, omega0 * omega0);
        }
    }

    // --- nstep_prediction_mse: regression test for the posterior-state fix ---
    //
    // Hand-derived with ts=1, omega0=1 (beta1=2, beta2=1), b0=1, n=7 (the minimum that
    // yields exactly one evaluated k, since warmup=1 and end=n-ESO_N_AHEAD_STEPS=2 when
    // n=7 — this makes the expected value tractable to compute by hand instead of trusting
    // the implementation under test.
    //
    // First pass (only k=0,1 affect the single evaluated state at k=1):
    //   X(0)=omega_meas[0]=0, F(0)=0
    //   k=0: e=0-0=0;            X(1)=0+(0+1*u0+2*0)=u0=1;         F(1)=0+1*0=0
    //   k=1: e=omega_meas[1]-X(1)=2-1=1;
    //        X(2)=1+(0+1*u1+2*1)=1+(-1)+2=2;   F(2)=0+1*1*1=1
    //   Posterior storage means omega_hat_states[1]=X(2)=2, f_hat_states[1]=F(2)=1.
    //
    // Second pass at k=1 (j=1..4, using u[2..5]=2,0.5,-0.5,1.5, f_pred frozen at 1):
    //   pred=2; +1*(1+2)=3 -> 5; +1*(1+0.5)=1.5 -> 6.5; +1*(1-0.5)=0.5 -> 7.0; +1*(1+1.5)=2.5 -> 9.5
    //   diff = 9.5 - omega_meas[6](=6) = 3.5 -> MSE = 12.25
    //
    // A pre-fix implementation (state stored *before* the correction, propagated with
    // j=0..5 instead of 1..5) gives MSE=6.25 on this same input — this test fails under
    // that implementation, so it guards the fix rather than merely restating the code.
    #[test]
    fn nstep_prediction_mse_uses_posterior_state_for_forecast() {
        let omega_meas = vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0];
        let u = vec![1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0];
        let mse = nstep_prediction_mse(&omega_meas, &u, 1.0, 1.0, 1.0);
        assert!(
            (mse - 12.25).abs() < 1e-9,
            "expected MSE=12.25 (posterior-state forecast), got {mse}"
        );
    }

    #[test]
    fn nstep_prediction_mse_infinite_below_minimum_length() {
        // n <= ESO_N_AHEAD_STEPS + 1 must short-circuit rather than index out of bounds.
        let short = vec![0.0; ESO_N_AHEAD_STEPS + 1];
        let mse = nstep_prediction_mse(&short, &short, 1.0, 100.0, 1.0);
        assert_eq!(mse, f64::INFINITY);
    }

    // --- estimate_b0: OLS ground-truth recovery + rejection paths ---

    #[test]
    fn estimate_b0_recovers_true_value_from_noiseless_data() {
        let ts = 0.001;
        let b0_true = 2.5;
        let u: Vec<f64> = (0..20).map(|k| 50.0 + 2.0 * k as f64).collect();
        let mut omega_meas = vec![100.0];
        for k in 0..19 {
            let next = omega_meas[k] + ts * b0_true * u[k];
            omega_meas.push(next);
        }
        let result = estimate_b0(&omega_meas, &u, ts);
        let b0 = result.expect("noiseless high-excitation data must yield an estimate");
        assert!(
            (b0 - b0_true).abs() < 1e-6,
            "expected b0≈{b0_true}, got {b0}"
        );
    }

    #[test]
    fn estimate_b0_none_below_min_ols_sample_count() {
        // Only 5 samples clear ESO_B0_MIN_CONTROL_THRESHOLD (< ESO_B0_MIN_OLS_SAMPLES=10).
        let mut u = vec![20.0; 5];
        u.extend(vec![1.0; 25]);
        let omega_meas = vec![0.0; u.len() + 1];
        assert_eq!(estimate_b0(&omega_meas, &u, 0.001), None);
    }

    #[test]
    fn estimate_b0_none_for_inverted_sign_convention() {
        // Same construction as the recovery test but with the sign flipped: OLS finds
        // b0=-2.5, which must be rejected as non-positive rather than returned.
        let ts = 0.001;
        let b0_true = -2.5;
        let u: Vec<f64> = (0..20).map(|k| 50.0 + 2.0 * k as f64).collect();
        let mut omega_meas = vec![100.0];
        for k in 0..19 {
            let next = omega_meas[k] + ts * b0_true * u[k];
            omega_meas.push(next);
        }
        assert_eq!(estimate_b0(&omega_meas, &u, ts), None);
    }

    // --- B0Source: label text ---

    #[test]
    fn b0_source_labels() {
        assert_eq!(B0Source::UserSupplied.label(), "user-supplied");
        assert_eq!(B0Source::AutoEstimated.label(), "auto-estimated");
        assert_eq!(B0Source::DefaultFallback.label(), "default fallback");
    }

    // --- run_eso_optimization: input validation ---

    #[test]
    fn rejects_invalid_axis() {
        let config = EsoConfig::default();
        assert!(run_eso_optimization(&[], TEST_SAMPLE_RATE, AXIS_COUNT, &config).is_err());
    }

    #[test]
    fn rejects_nonpositive_or_nonfinite_sample_rate() {
        let config = EsoConfig::default();
        for bad_rate in [0.0, -100.0, f64::NAN, f64::INFINITY] {
            assert!(
                run_eso_optimization(&[], bad_rate, 0, &config).is_err(),
                "sample_rate={bad_rate} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_inverted_omega0_bounds() {
        let config = EsoConfig {
            omega0_min: 500.0,
            omega0_max: 50.0,
            ..Default::default()
        };
        assert!(run_eso_optimization(&[], TEST_SAMPLE_RATE, 0, &config).is_err());
    }

    #[test]
    fn rejects_insufficient_data() {
        let config = EsoConfig::default();
        let log_data = vec![row_with_gyro_and_pid(1.0, 10.0, 0.0)];
        assert!(run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config).is_err());
    }

    #[test]
    fn rejects_no_control_input_excitation() {
        let config = EsoConfig::default();
        let log_data: Vec<LogRowData> = (0..20)
            .map(|k| row_with_gyro_and_pid(1.0, 0.0, k as f64 / TEST_SAMPLE_RATE))
            .collect();
        let err = run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config).unwrap_err();
        assert!(err.to_string().contains("excitation"));
    }

    // --- run_eso_optimization: end-to-end behavior ---

    #[test]
    fn recovers_b0_from_noiseless_synthetic_log() {
        let ts = 1.0 / TEST_SAMPLE_RATE;
        let b0_true = 3.0;
        let u: Vec<f64> = (0..200)
            .map(|k| 30.0 + 5.0 * (k as f64 * 0.3).sin())
            .collect();
        let mut omega_meas = vec![0.0];
        for k in 0..199 {
            let next = omega_meas[k] + ts * b0_true * u[k];
            omega_meas.push(next);
        }
        let log_data: Vec<LogRowData> = omega_meas
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(k, (&g, &p))| row_with_gyro_and_pid(g, p, k as f64 * ts))
            .collect();

        let config = EsoConfig::default();
        let result = run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config)
            .expect("well-excited noiseless log must succeed");

        assert_eq!(result.b0_source, B0Source::AutoEstimated);
        assert!(
            (result.b0 - b0_true).abs() < 1e-3,
            "expected b0≈{b0_true}, got {}",
            result.b0
        );
    }

    #[test]
    fn respects_user_supplied_b0() {
        let ts = 1.0 / TEST_SAMPLE_RATE;
        let u: Vec<f64> = (0..200)
            .map(|k| 30.0 + 5.0 * (k as f64 * 0.3).sin())
            .collect();
        let mut omega_meas = vec![0.0];
        for k in 0..199 {
            let next = omega_meas[k] + ts * 3.0 * u[k];
            omega_meas.push(next);
        }
        let log_data: Vec<LogRowData> = omega_meas
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(k, (&g, &p))| row_with_gyro_and_pid(g, p, k as f64 * ts))
            .collect();

        let config = EsoConfig {
            b0: 7.0,
            b0_user_override: true,
            ..Default::default()
        };
        let result = run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config).unwrap();

        assert_eq!(result.b0, 7.0);
        assert_eq!(result.b0_source, B0Source::UserSupplied);
    }

    #[test]
    fn falls_back_to_default_b0_when_ols_rejects() {
        // Every sample clears VALUE_EPSILON (so the coarse excitation gate passes) but
        // stays below ESO_B0_MIN_CONTROL_THRESHOLD (so OLS's own count never reaches
        // ESO_B0_MIN_OLS_SAMPLES) — must fall back, not silently label itself user-supplied.
        let config = EsoConfig::default();
        let log_data: Vec<LogRowData> = (0..20)
            .map(|k| row_with_gyro_and_pid(1.0, 0.5, k as f64 / TEST_SAMPLE_RATE))
            .collect();
        let result = run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config).unwrap();

        assert_eq!(result.b0_source, B0Source::DefaultFallback);
        assert_eq!(result.b0, ESO_DEFAULT_B0);
    }

    #[test]
    fn search_finds_a_point_no_worse_than_either_boundary() {
        // A minimizer must never do worse than the bracket's own endpoints. Data mixes a
        // slow "disturbance" component with a faster one so the cost genuinely varies with
        // omega_0, rather than being flat (which would make this check vacuous).
        let ts = 1.0 / TEST_SAMPLE_RATE;
        let omega_meas: Vec<f64> = (0..200)
            .map(|k| {
                let t = k as f64 * ts;
                10.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
                    + 2.0 * (2.0 * std::f64::consts::PI * 80.0 * t).sin()
            })
            .collect();
        let u: Vec<f64> = (0..200)
            .map(|k| {
                let t = k as f64 * ts;
                30.0 * (2.0 * std::f64::consts::PI * 5.0 * t).cos()
            })
            .collect();
        let log_data: Vec<LogRowData> = omega_meas
            .iter()
            .zip(u.iter())
            .enumerate()
            .map(|(k, (&g, &p))| row_with_gyro_and_pid(g, p, k as f64 * ts))
            .collect();

        let config = EsoConfig::default();
        let result = run_eso_optimization(&log_data, TEST_SAMPLE_RATE, 0, &config).unwrap();

        let omega0_max_stable =
            (TEST_SAMPLE_RATE / ESO_OMEGA0_STABILITY_RATIO).min(config.omega0_max);
        let mse_at_min = nstep_prediction_mse(&omega_meas, &u, ts, config.omega0_min, result.b0);
        let mse_at_max = nstep_prediction_mse(&omega_meas, &u, ts, omega0_max_stable, result.b0);

        assert!(result.mse <= mse_at_min + 1e-9);
        assert!(result.mse <= mse_at_max + 1e-9);
    }
}
