// src/eso.rs
// 2nd-order Linear Extended State Observer (LESO) for flight controller blackbox data.
// Implements discrete Euler-forward LESO simulation with golden-section search for the
// optimal observer bandwidth (omega_0) using PID sum as control input and filtered
// gyro as measured output.

use std::error::Error;

use crate::constants::{
    ESO_DEFAULT_B0, ESO_GSS_MAX_ITER, ESO_GSS_TOLERANCE, ESO_OMEGA0_MAX, ESO_OMEGA0_MIN,
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
    /// MSE tracking cost at the optimal omega0.
    pub mse: f64,
    /// Number of samples used in the optimization.
    pub sample_count: usize,
}

/// Compute 2nd-order bandwidth-parameterized LESO gains from omega_0.
/// Returns (beta1, beta2): beta1 = 2*omega0, beta2 = omega0^2.
fn leso2_gains(omega0: f64) -> (f64, f64) {
    (2.0 * omega0, omega0 * omega0)
}

/// Simulate 2nd-order discrete LESO (Euler forward) and return estimated rate sequence.
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
fn simulate_leso2(omega_meas: &[f64], u: &[f64], ts: f64, omega0: f64, b0: f64) -> Vec<f64> {
    let (beta1, beta2) = leso2_gains(omega0);
    let n = omega_meas.len().min(u.len());

    let mut omega_hat = omega_meas.first().copied().unwrap_or(0.0);
    let mut f_hat = 0.0_f64;

    let mut estimated = Vec::with_capacity(n);
    for k in 0..n {
        estimated.push(omega_hat);
        let e = omega_meas[k] - omega_hat;
        omega_hat += ts * (f_hat + b0 * u[k] + beta1 * e);
        f_hat += ts * (beta2 * e);
    }
    estimated
}

/// Compute MSE between LESO-estimated and measured rate for a given omega_0.
fn mse_cost(omega_meas: &[f64], u: &[f64], ts: f64, omega0: f64, b0: f64) -> f64 {
    let estimated = simulate_leso2(omega_meas, u, ts, omega0, b0);
    let n = estimated.len().min(omega_meas.len());
    if n == 0 {
        return f64::INFINITY;
    }
    let sum_sq: f64 = estimated
        .iter()
        .zip(omega_meas.iter())
        .map(|(e, m)| (e - m).powi(2))
        .sum();
    sum_sq / n as f64
}

/// Golden-section search for the minimum of a unimodal function on [lo, hi].
/// Returns (best_x, f(best_x)).
fn golden_section_search<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64) {
    // Inverse golden ratio: (sqrt(5) - 1) / 2
    const PHI_INV: f64 = 0.618_033_988_749_895;
    let mut x1 = hi - PHI_INV * (hi - lo);
    let mut x2 = lo + PHI_INV * (hi - lo);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for _ in 0..max_iter {
        if (hi - lo).abs() < tol {
            break;
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - PHI_INV * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + PHI_INV * (hi - lo);
            f2 = f(x2);
        }
    }
    let best_x = (lo + hi) / 2.0;
    (best_x, f(best_x))
}

/// Extract gyro measurements and PID sum for an axis from log data.
/// Rows with missing gyro are skipped; PID terms default to 0.0 if absent.
/// Returns (omega_meas, pid_sum) or None when fewer than 2 samples are available.
fn extract_axis_data(log_data: &[LogRowData], axis: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut omega_meas = Vec::with_capacity(log_data.len());
    let mut pid_sum = Vec::with_capacity(log_data.len());

    for row in log_data {
        if let Some(gyro) = row.gyro[axis] {
            let p = row.p_term[axis].unwrap_or(0.0);
            let i_val = row.i_term[axis].unwrap_or(0.0);
            let d = row.d_term[axis].unwrap_or(0.0);
            let f_val = row.f_term[axis].unwrap_or(0.0);
            omega_meas.push(gyro);
            pid_sum.push(p + i_val + d + f_val);
        }
    }

    if omega_meas.len() < 2 {
        return None;
    }
    Some((omega_meas, pid_sum))
}

/// Run ESO gain optimization for a single axis using golden-section search on omega_0.
///
/// The search is constrained to omega_0 < sample_rate / 3 for discrete-time stability.
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
    let (omega_meas, pid_sum) = extract_axis_data(log_data, axis)
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
    let b0 = config.b0;
    let cost_fn = |omega0: f64| mse_cost(&omega_meas, &pid_sum, ts, omega0, b0);

    let (omega0_opt, mse) = golden_section_search(
        cost_fn,
        config.omega0_min,
        omega0_max_stable,
        ESO_GSS_TOLERANCE,
        ESO_GSS_MAX_ITER,
    );

    let (beta1, beta2) = leso2_gains(omega0_opt);

    Ok(EsoResult {
        axis,
        omega0_opt,
        beta1,
        beta2,
        b0,
        mse,
        sample_count: omega_meas.len(),
    })
}
