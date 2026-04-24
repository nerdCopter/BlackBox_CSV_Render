// src/report.rs
// Markdown statistical report generation for flight controller blackbox data.
// Produces a per-axis summary: signal statistics and optional ESO optimization results.

use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::Path;

use crate::axis_names::{AXIS_COUNT, AXIS_NAMES};
use crate::data_input::log_data::LogRowData;
use crate::eso::EsoResult;

/// Descriptive statistics for a time-series signal.
#[derive(Debug, Clone)]
pub struct SignalStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub rms: f64,
    pub count: usize,
}

/// Compute descriptive statistics for a slice of f64 values.
/// Returns None if the slice is empty.
///
/// **Variance convention:** population variance (divide by N), not sample variance (N-1).
/// This is appropriate for a complete recorded time-series rather than a statistical sample.
/// To switch to sample std-dev, change the divisor to `(count - 1)` and guard for `count < 2`.
pub fn compute_signal_stats(data: &[f64]) -> Option<SignalStats> {
    let count = data.len();
    if count == 0 {
        return None;
    }
    let mean = data.iter().sum::<f64>() / count as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = var.sqrt();
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rms = (data.iter().map(|x| x * x).sum::<f64>() / count as f64).sqrt();
    Some(SignalStats {
        mean,
        std_dev,
        min,
        max,
        rms,
        count,
    })
}

/// Per-axis signals extracted from log data for report generation.
struct AxisSignals {
    gyro: Vec<f64>,
    setpoint: Vec<f64>,
    pid_sum: Vec<f64>,
    pid_p: Vec<f64>,
    pid_i: Vec<f64>,
    pid_d: Vec<f64>,
}

fn extract_axis_signals(log_data: &[LogRowData], axis: usize) -> AxisSignals {
    let mut gyro = Vec::new();
    let mut setpoint = Vec::new();
    let mut pid_sum = Vec::new();
    let mut pid_p = Vec::new();
    let mut pid_i = Vec::new();
    let mut pid_d = Vec::new();

    for row in log_data {
        if let Some(g) = row.gyro[axis] {
            gyro.push(g);
        }
        if let Some(p) = row.p_term[axis] {
            pid_p.push(p);
        }
        if let Some(i_val) = row.i_term[axis] {
            pid_i.push(i_val);
        }
        if let Some(d) = row.d_term[axis] {
            pid_d.push(d);
        }
        let pid_terms = [
            row.p_term[axis],
            row.i_term[axis],
            row.d_term[axis],
            row.f_term[axis],
        ];
        if pid_terms.iter().any(|term| term.is_some()) {
            pid_sum.push(pid_terms.iter().map(|term| term.unwrap_or(0.0)).sum());
        }
        if let Some(sp) = row.setpoint.get(axis).copied().flatten() {
            setpoint.push(sp);
        }
    }

    AxisSignals {
        gyro,
        setpoint,
        pid_sum,
        pid_p,
        pid_i,
        pid_d,
    }
}

fn write_stats_row(md: &mut String, name: &str, stats: &SignalStats) -> std::fmt::Result {
    writeln!(
        md,
        "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |",
        name, stats.mean, stats.std_dev, stats.min, stats.max, stats.rms, stats.count
    )
}

/// Generate a markdown statistical report and write it to `output_path`.
///
/// # Arguments
/// * `log_data` - Parsed blackbox log rows.
/// * `sample_rate` - Detected sample rate in Hz (None if unknown).
/// * `header_metadata` - Raw header key-value pairs from the log file.
/// * `output_path` - Destination file path for the `.md` report.
/// * `eso_results` - Per-axis ESO results (indexed 0=Roll, 1=Pitch, 2=Yaw).
pub fn generate_markdown_report(
    log_data: &[LogRowData],
    sample_rate: Option<f64>,
    header_metadata: &[(String, String)],
    output_path: &Path,
    eso_results: &[Option<EsoResult>; AXIS_COUNT],
) -> Result<(), Box<dyn Error>> {
    let mut md = String::new();

    writeln!(md, "# BlackBox Statistical Report")?;
    writeln!(md)?;
    writeln!(md, "## Metadata")?;
    writeln!(md)?;

    match sample_rate {
        Some(sr) => writeln!(md, "- **Sample Rate:** {:.1} Hz", sr)?,
        None => writeln!(md, "- **Sample Rate:** Unknown")?,
    }
    writeln!(md, "- **Total Rows:** {}", log_data.len())?;

    if let (Some(first), Some(last)) = (
        log_data.first().and_then(|r| r.time_sec),
        log_data.last().and_then(|r| r.time_sec),
    ) {
        writeln!(
            md,
            "- **Duration:** {:.2} s  ({:.2} s to {:.2} s)",
            last - first,
            first,
            last
        )?;
    }
    writeln!(md)?;

    // Selected firmware / configuration keys
    let interesting_keys = [
        "Firmware revision",
        "Craft name",
        "looptime",
        "gyro_lpf_hz",
        "dterm_lpf_hz",
        "pid_process_denom",
        "rollPID",
        "pitchPID",
        "yawPID",
    ];
    let mut wrote_fw_header = false;
    for (k, v) in header_metadata {
        if interesting_keys.iter().any(|ik| k.eq_ignore_ascii_case(ik)) {
            if !wrote_fw_header {
                writeln!(md, "### Firmware / Configuration")?;
                writeln!(md)?;
                wrote_fw_header = true;
            }
            writeln!(md, "- **{}:** {}", k, v)?;
        }
    }
    if wrote_fw_header {
        writeln!(md)?;
    }

    // Per-axis signal statistics
    writeln!(md, "## Per-Axis Signal Statistics")?;
    writeln!(md)?;

    for axis_idx in 0..AXIS_COUNT {
        let axis_name = AXIS_NAMES[axis_idx];
        let sigs = extract_axis_signals(log_data, axis_idx);

        writeln!(md, "### {}", axis_name)?;
        writeln!(md)?;
        writeln!(md, "| Signal | Mean | Std Dev | Min | Max | RMS | Count |")?;
        writeln!(md, "|--------|------|---------|-----|-----|-----|-------|")?;

        if let Some(s) = compute_signal_stats(&sigs.gyro) {
            write_stats_row(&mut md, "Gyro (filt)", &s)?;
        }
        if let Some(s) = compute_signal_stats(&sigs.setpoint) {
            write_stats_row(&mut md, "Setpoint", &s)?;
        }
        if let Some(s) = compute_signal_stats(&sigs.pid_sum) {
            write_stats_row(&mut md, "PID Sum", &s)?;
        }
        if let Some(s) = compute_signal_stats(&sigs.pid_p) {
            write_stats_row(&mut md, "P-term", &s)?;
        }
        if let Some(s) = compute_signal_stats(&sigs.pid_i) {
            write_stats_row(&mut md, "I-term", &s)?;
        }
        if let Some(s) = compute_signal_stats(&sigs.pid_d) {
            write_stats_row(&mut md, "D-term", &s)?;
        }
        writeln!(md)?;

        // ESO optimization results for this axis
        if let Some(eso) = &eso_results[axis_idx] {
            writeln!(md, "**ESO Optimization Result (2nd-order LESO):**")?;
            writeln!(md)?;
            writeln!(md, "| Parameter | Value |")?;
            writeln!(md, "|-----------|-------|")?;
            writeln!(md, "| omega_0 (optimal) | {:.2} rad/s |", eso.omega0_opt)?;
            writeln!(md, "| beta1 = 2*omega_0 | {:.2} |", eso.beta1)?;
            writeln!(md, "| beta2 = omega_0^2 | {:.4} |", eso.beta2)?;
            let b0_label = if eso.b0_auto {
                "(auto-estimated from data)"
            } else {
                "(user-supplied)"
            };
            writeln!(
                md,
                "| b0 (control effectiveness) | {:.4} {b0_label} |",
                eso.b0
            )?;
            writeln!(md, "| MSE (N-step-ahead prediction) | {:.6} |", eso.mse)?;
            writeln!(md, "| Samples | {} |", eso.sample_count)?;
            writeln!(md)?;
            writeln!(
                md,
                "> Stability requirement: omega_0 < loop_rate / 3. \
                Use in an ADRC implementation. b0 is estimated from data via OLS on rate \
                derivatives (QuickFlash method); override with --eso-b0 if needed."
            )?;
            writeln!(md)?;
        }
    }

    fs::write(output_path, md)?;
    Ok(())
}
