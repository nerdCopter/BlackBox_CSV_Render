// src/report.rs
// Markdown statistical report generation for flight controller blackbox data.
// Produces per-axis signal statistics and links to generated PNG plots.

use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use crate::axis_names::{AXIS_COUNT, AXIS_NAMES};
use crate::data_input::log_data::LogRowData;

/// Descriptive statistics for a time-series signal.
pub struct SignalStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub rms: f64,
    pub count: usize,
}

/// Compute descriptive statistics for a slice of f64 values.
/// Uses population variance (divide by N) appropriate for a complete time-series.
/// Returns None if the slice is empty.
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

fn extract_axis_gyro(log_data: &[LogRowData], axis: usize) -> Vec<f64> {
    log_data.iter().filter_map(|r| r.gyro[axis]).collect()
}

fn extract_axis_setpoint(log_data: &[LogRowData], axis: usize) -> Vec<f64> {
    log_data
        .iter()
        .filter_map(|r| r.setpoint.get(axis).copied().flatten())
        .collect()
}

fn extract_axis_pid_term(log_data: &[LogRowData], axis: usize, term: u8) -> Vec<f64> {
    log_data
        .iter()
        .filter_map(|r| match term {
            0 => r.p_term[axis],
            1 => r.i_term[axis],
            2 => r.d_term[axis],
            3 => r.f_term[axis],
            _ => None,
        })
        .collect()
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
/// * `sample_rate` - Detected sample rate in Hz, or None if unknown.
/// * `header_metadata` - Raw header key-value pairs from the log file.
/// * `output_path` - Destination `.md` file path.
/// * `png_links` - List of PNG filenames to link (relative, same directory).
pub fn generate_markdown_report(
    log_data: &[LogRowData],
    sample_rate: Option<f64>,
    header_metadata: &[(String, String)],
    output_path: &Path,
    png_links: &[String],
) -> Result<(), Box<dyn Error>> {
    let mut md = String::new();

    writeln!(md, "# BlackBox Statistical Report")?;
    writeln!(md)?;

    // --- Metadata ---
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
            "- **Duration:** {:.2} s  ({:.2} s – {:.2} s)",
            last - first,
            first,
            last
        )?;
    }
    writeln!(md)?;

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

    // --- Per-axis signal statistics ---
    writeln!(md, "## Per-Axis Signal Statistics")?;
    writeln!(md)?;

    for (axis_idx, axis_name) in AXIS_NAMES.iter().enumerate().take(AXIS_COUNT) {
        writeln!(md, "### {}", axis_name)?;
        writeln!(md)?;
        writeln!(md, "| Signal | Mean | Std Dev | Min | Max | RMS | Count |")?;
        writeln!(md, "|--------|------|---------|-----|-----|-----|-------|")?;

        if let Some(s) = compute_signal_stats(&extract_axis_gyro(log_data, axis_idx)) {
            write_stats_row(&mut md, "Gyro (filt)", &s)?;
        }
        if let Some(s) = compute_signal_stats(&extract_axis_setpoint(log_data, axis_idx)) {
            write_stats_row(&mut md, "Setpoint", &s)?;
        }
        if let Some(s) = compute_signal_stats(&extract_axis_pid_term(log_data, axis_idx, 0)) {
            write_stats_row(&mut md, "P-term", &s)?;
        }
        if let Some(s) = compute_signal_stats(&extract_axis_pid_term(log_data, axis_idx, 1)) {
            write_stats_row(&mut md, "I-term", &s)?;
        }
        if let Some(s) = compute_signal_stats(&extract_axis_pid_term(log_data, axis_idx, 2)) {
            write_stats_row(&mut md, "D-term", &s)?;
        }
        if let Some(s) = compute_signal_stats(&extract_axis_pid_term(log_data, axis_idx, 3)) {
            write_stats_row(&mut md, "F-term", &s)?;
        }
        writeln!(md)?;
    }

    // --- Generated plots ---
    if !png_links.is_empty() {
        writeln!(md, "## Generated Plots")?;
        writeln!(md)?;
        for name in png_links {
            writeln!(md, "- [{}]({})", name, name)?;
        }
        writeln!(md)?;
    }

    fs::write(output_path, md)?;
    Ok(())
}
