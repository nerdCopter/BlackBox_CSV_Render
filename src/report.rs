// src/report.rs
// Structured markdown report capturing computed flight analysis outputs.
// Derives all content from typed structs returned by analysis functions —
// no re-reading of CSV data, no duplication of println logic.

use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use crate::axis_names::{AXIS_COUNT, AXIS_NAMES};
use crate::constants::{MOTOR_OSCILLATION_FREQ_MAX_HZ, MOTOR_OSCILLATION_FREQ_MIN_HZ};
use crate::data_analysis::filter_response::{
    AllFilterConfigs, DynamicNotchConfig, RpmFilterConfig,
};
use crate::data_analysis::optimal_p_estimation::{OptimalPAnalysis, PRecommendation};
use crate::data_analysis::transfer_function_estimation::Confidence;
use crate::plot_functions::plot_bode::BodeAxisResult;
use crate::plot_functions::plot_d_term_spectrums::DTermAxisResult;
use crate::plot_functions::plot_gyro_spectrums::GyroAnalysisResult;
use crate::plot_functions::plot_motor_spectrums::MotorOscillationResult;

/// D-term recommendation for one tier (conservative, moderate, or aggressive)
pub struct DTermRec {
    pub pd_ratio: f64,
    pub d: Option<u32>,
    pub d_min: Option<u32>,
    pub d_max: Option<u32>,
}

/// Step response analysis result for one axis
pub struct StepAxisReport {
    pub axis_name: &'static str,
    pub peak_value: f64,
    pub assessment: &'static str,
    pub current_pd_ratio: f64,
    pub conservative: Option<DTermRec>,
    pub moderate: Option<DTermRec>,
    pub aggressive: Option<DTermRec>,
    pub setpoint_authority_name: Option<&'static str>,
    pub setpoint_authority_mean: Option<f32>,
    pub warnings: Vec<String>,
}

/// Aggregated analysis results collected from one flight log
pub struct FlightReport {
    pub root_name: String,
    pub sample_rate: Option<f64>,
    pub header_metadata: Vec<(String, String)>,
    pub pd_ratios: [Option<f64>; 3],
    pub step_reports: Vec<StepAxisReport>,
    pub optimal_p: [Option<OptimalPAnalysis>; AXIS_COUNT],
    pub gyro_analysis: Option<GyroAnalysisResult>,
    pub dterm_results: Vec<DTermAxisResult>,
    pub bode_results: Vec<BodeAxisResult>,
    pub motor_results: Vec<MotorOscillationResult>,
    pub png_links: Vec<String>,
    pub filter_config: Option<AllFilterConfigs>,
    pub dynamic_notch: Option<DynamicNotchConfig>,
    pub rpm_filter: Option<RpmFilterConfig>,
    /// True when gyroUnfilt came from debug channels instead of dedicated gyroUnfilt columns.
    pub debug_fallback: bool,
    pub debug_mode_name: Option<&'static str>,
}

/// Generate a structured markdown report and write it to `output_path`.
pub fn generate_markdown_report(
    report: &FlightReport,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut md = String::new();

    writeln!(md, "# BlackBox Flight Report: {}", report.root_name)?;
    writeln!(md)?;

    // --- Metadata ---
    writeln!(md, "## Metadata")?;
    writeln!(md)?;
    match report.sample_rate {
        Some(sr) => writeln!(md, "- **Sample Rate:** {:.1} Hz", sr)?,
        None => writeln!(md, "- **Sample Rate:** Unknown")?,
    }
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
    for (k, v) in &report.header_metadata {
        if interesting_keys.iter().any(|ik| k.eq_ignore_ascii_case(ik)) {
            writeln!(md, "- **{}:** {}", k, v)?;
        }
    }
    if report.debug_fallback {
        let mode_str = report
            .debug_mode_name
            .map_or(String::new(), |m| format!(" ({})", m));
        writeln!(
            md,
            "- **⚠ gyroUnfilt source:** debug[0-2] fallback{} — unfiltered gyro spectrum and filtering delay derived from debug channels, not dedicated gyroUnfilt columns",
            mode_str
        )?;
    }
    writeln!(md)?;

    // --- Filter Configuration ---
    if let Some(ref fc) = report.filter_config {
        let axis_labels = ["Roll", "Pitch", "Yaw"];
        // Only emit section if at least one axis has any filter configured
        let has_any = (0..3).any(|i| {
            fc.gyro[i].lpf1.is_some()
                || fc.gyro[i].lpf2.is_some()
                || fc.gyro[i].dynamic_lpf1.is_some()
                || fc.gyro[i].imuf.is_some()
        });
        if has_any {
            writeln!(md, "## Filter Configuration")?;
            writeln!(md)?;
            writeln!(md, "| Axis | LPF1 | LPF2 | IMUF / Pseudo-Kalman |")?;
            writeln!(md, "|------|------|------|----------------------|")?;
            for (i, label) in axis_labels.iter().enumerate() {
                let lpf1 =
                    fmt_filter_stage(fc.gyro[i].lpf1.as_ref(), fc.gyro[i].dynamic_lpf1.as_ref());
                let lpf2 = fc.gyro[i]
                    .lpf2
                    .as_ref()
                    .map_or("—".into(), fmt_static_filter);
                let imuf = fc.gyro[i].imuf.as_ref().map_or("—".into(), fmt_imuf);
                writeln!(md, "| {} | {} | {} | {} |", label, lpf1, lpf2, imuf)?;
            }
            writeln!(md)?;

            if let Some(ref dn) = report.dynamic_notch {
                writeln!(
                    md,
                    "- **Dynamic Notch:** {:.0}–{:.0} Hz, Q={:.0}, {} notch{}",
                    dn.min_hz,
                    dn.max_hz,
                    dn.q_factor,
                    dn.notch_count,
                    if dn.notch_count == 1 { "" } else { "es" }
                )?;
            }
            if let Some(ref rpm) = report.rpm_filter {
                writeln!(
                    md,
                    "- **RPM Filter:** {} harmonic{}, Q={:.0}, min {:.0} Hz",
                    rpm.harmonics,
                    if rpm.harmonics == 1 { "" } else { "s" },
                    rpm.q_factor,
                    rpm.min_hz
                )?;
            }
            if report.dynamic_notch.is_some() || report.rpm_filter.is_some() {
                writeln!(md)?;
            }
        }
    }

    // --- PID Tuning ---
    writeln!(md, "## PID Tuning")?;
    writeln!(md)?;
    writeln!(md, "| Axis | P:D Ratio |")?;
    writeln!(md, "|------|-----------|")?;
    let axis_labels = ["Roll", "Pitch", "Yaw (informational)"];
    for (i, label) in axis_labels.iter().enumerate() {
        match report.pd_ratios[i] {
            Some(r) => writeln!(md, "| {} | {:.2} |", label, r)?,
            None => writeln!(md, "| {} | N/A |", label)?,
        }
    }
    writeln!(md)?;

    // --- Step Response Analysis ---
    if !report.step_reports.is_empty() {
        writeln!(md, "## Step Response Analysis")?;
        writeln!(md)?;
        for axis_report in &report.step_reports {
            writeln!(md, "### {}", axis_report.axis_name)?;
            writeln!(md)?;
            writeln!(md, "- **Peak Value:** {:.3}", axis_report.peak_value)?;
            writeln!(md, "- **Assessment:** {}", axis_report.assessment)?;
            writeln!(md, "- **Current P:D:** {:.2}", axis_report.current_pd_ratio)?;
            if let (Some(name), Some(mean)) = (
                axis_report.setpoint_authority_name,
                axis_report.setpoint_authority_mean,
            ) {
                writeln!(
                    md,
                    "- **Setpoint Authority:** {} (mean {:.0} dps)",
                    name, mean
                )?;
            }
            if let Some(rec) = &axis_report.conservative {
                writeln!(
                    md,
                    "- **Recommendation (conservative):** {}",
                    fmt_dterm_rec(rec)
                )?;
            }
            if let Some(rec) = &axis_report.moderate {
                writeln!(
                    md,
                    "- **Recommendation (moderate):** {}",
                    fmt_dterm_rec(rec)
                )?;
            }
            if let Some(rec) = &axis_report.aggressive {
                writeln!(
                    md,
                    "- **Recommendation (aggressive):** {}",
                    fmt_dterm_rec(rec)
                )?;
            }
            if axis_report.conservative.is_none()
                && axis_report.moderate.is_none()
                && axis_report.aggressive.is_none()
            {
                writeln!(
                    md,
                    "- **Recommendation:** No obvious tuning adjustments needed"
                )?;
            }
            for w in &axis_report.warnings {
                writeln!(md, "- **⚠ Warning:** {}", w)?;
            }
            writeln!(md)?;
        }
    }

    // --- Gyro Analysis (per-axis filtering delay + spectrum peaks) ---
    if let Some(gyro) = &report.gyro_analysis {
        let has_gyro_data = gyro
            .axes
            .iter()
            .any(|a| !a.peaks.is_empty() || a.delay_ms.is_some());
        if has_gyro_data {
            writeln!(md, "## Gyro Analysis")?;
            writeln!(md)?;
            writeln!(
                md,
                "| Axis | Delay (ms) | Confidence | Peak | Freq (Hz) | Amplitude |"
            )?;
            writeln!(
                md,
                "|------|-----------|------------|------|-----------|-----------|"
            )?;
            for axis in &gyro.axes {
                let delay = axis.delay_ms.map_or("N/A".into(), |v| format!("{:.2}", v));
                let conf = axis
                    .delay_confidence
                    .map_or("N/A".into(), |v| format!("{:.0}%", v * 100.0));
                if axis.peaks.is_empty() {
                    writeln!(
                        md,
                        "| {} | {} | {} | N/A | N/A | N/A |",
                        axis.axis_name, delay, conf
                    )?;
                } else {
                    for (idx, (freq, amp)) in axis.peaks.iter().enumerate() {
                        let label = if idx == 0 {
                            "Primary".to_string()
                        } else {
                            format!("Sub {}", idx)
                        };
                        let (d_col, c_col) = if idx == 0 {
                            (delay.clone(), conf.clone())
                        } else {
                            ("".into(), "".into())
                        };
                        writeln!(
                            md,
                            "| {} | {} | {} | {} | {:.1} | {:.2} |",
                            axis.axis_name, d_col, c_col, label, freq, amp
                        )?;
                    }
                }
            }
            writeln!(md)?;
        }
    }

    // --- D-Term Analysis (per-axis delay + spectrum peaks) ---
    let has_dterm_data = report
        .dterm_results
        .iter()
        .any(|r| !r.peaks.is_empty() || r.delay_ms.is_some());
    if has_dterm_data {
        writeln!(md, "## D-Term Analysis")?;
        writeln!(md)?;
        writeln!(
            md,
            "| Axis | Delay (ms) | Confidence | Peak | Freq (Hz) | Amplitude |"
        )?;
        writeln!(
            md,
            "|------|-----------|------------|------|-----------|-----------|"
        )?;
        for r in &report.dterm_results {
            let delay = r.delay_ms.map_or_else(
                || {
                    r.na_reason
                        .map_or("N/A".into(), |reason| format!("N/A ({})", reason))
                },
                |v| format!("{:.1}", v),
            );
            let conf = r
                .delay_confidence
                .map_or("N/A".into(), |v| format!("{:.0}%", v * 100.0));
            if r.peaks.is_empty() {
                writeln!(
                    md,
                    "| {} | {} | {} | N/A | N/A | N/A |",
                    r.axis_name, delay, conf
                )?;
            } else {
                for (idx, (freq, amp)) in r.peaks.iter().enumerate() {
                    let label = if idx == 0 {
                        "Primary".to_string()
                    } else {
                        format!("Sub {}", idx)
                    };
                    let (d_col, c_col) = if idx == 0 {
                        (delay.clone(), conf.clone())
                    } else {
                        ("".into(), "".into())
                    };
                    writeln!(
                        md,
                        "| {} | {} | {} | {} | {:.1} | {:.2} |",
                        r.axis_name, d_col, c_col, label, freq, amp
                    )?;
                }
            }
        }
        writeln!(md)?;
    }

    // --- Optimal P Estimation ---
    let has_optimal_p = report.optimal_p.iter().any(|o| o.is_some());
    if has_optimal_p {
        writeln!(md, "## Optimal P Estimation")?;
        writeln!(md)?;
        for (axis_index, opt) in report.optimal_p.iter().enumerate() {
            if let Some(analysis) = opt {
                writeln!(md, "### {}", AXIS_NAMES[axis_index])?;
                writeln!(md)?;
                writeln!(md, "- **Current P:** {}", analysis.current_p)?;
                if let Some(d) = analysis.current_d {
                    writeln!(md, "- **Current D:** {}", d)?;
                }
                writeln!(
                    md,
                    "- **Td measured:** {:.1} ms ({} samples)",
                    analysis.td_stats.mean_ms, analysis.td_stats.num_samples
                )?;
                writeln!(
                    md,
                    "- **Td target:** {:.1} ms ± {:.1} ms",
                    analysis.td_target_ms, analysis.td_tolerance_ms
                )?;
                writeln!(
                    md,
                    "- **Td deviation:** {:.1}% ({})",
                    analysis.td_deviation_percent,
                    analysis.td_deviation.name()
                )?;
                writeln!(md, "- **Noise level:** {}", analysis.noise_level.name())?;
                let rec_text = match &analysis.recommendation {
                    PRecommendation::Optimal { reasoning } => {
                        format!("No change — {}", reasoning)
                    }
                    PRecommendation::Increase {
                        conservative_p,
                        reasoning,
                    } => format!("Increase P to {} — {}", conservative_p, reasoning),
                    PRecommendation::Decrease {
                        recommended_p,
                        reasoning,
                    } => format!("Decrease P to {} — {}", recommended_p, reasoning),
                    PRecommendation::Investigate { issue } => format!("Investigate — {}", issue),
                };
                writeln!(md, "- **Recommendation:** {}", rec_text)?;
                if analysis.source_files > 1 {
                    writeln!(
                        md,
                        "- **Source:** {} files, {} throttle-punch events",
                        analysis.source_files, analysis.source_events
                    )?;
                } else {
                    writeln!(
                        md,
                        "- **Source:** {} throttle-punch events",
                        analysis.source_events
                    )?;
                }
                writeln!(md)?;
            }
        }
    }

    // --- Bode Analysis ---
    if !report.bode_results.is_empty() {
        writeln!(md, "## Bode Analysis")?;
        writeln!(md)?;
        writeln!(
            md,
            "> ⚠ Bode analysis is designed for controlled system-identification test flights."
        )?;
        writeln!(md)?;
        writeln!(
            md,
            "| Axis | Phase Margin | Gain Margin | Gain Crossover | Bandwidth | Confidence |"
        )?;
        writeln!(
            md,
            "|------|-------------|-------------|----------------|-----------|------------|"
        )?;
        for r in &report.bode_results {
            let m = &r.margins;
            let pm = m
                .phase_margin_deg
                .map_or("N/A".into(), |v| format!("{:.1}°", v));
            let gm = m
                .gain_margin_db
                .map_or("N/A".into(), |v| format!("{:.1} dB", v));
            let gc = m
                .gain_crossover_hz
                .map_or("N/A".into(), |v| format!("{:.2} Hz", v));
            let bw = m
                .bandwidth_hz
                .map_or("N/A".into(), |v| format!("{:.2} Hz", v));
            let conf = match m.confidence {
                Confidence::High => "High",
                Confidence::Medium => "Medium",
                Confidence::Low => "Low",
            };
            writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} |",
                r.axis_name, pm, gm, gc, bw, conf
            )?;
        }
        writeln!(md)?;
    }

    // --- Motor Oscillation ---
    if !report.motor_results.is_empty() {
        writeln!(md, "## Motor Oscillation")?;
        writeln!(md)?;
        writeln!(
            md,
            "Analysis range: {:.0}–{:.0} Hz",
            MOTOR_OSCILLATION_FREQ_MIN_HZ, MOTOR_OSCILLATION_FREQ_MAX_HZ
        )?;
        writeln!(md)?;
        writeln!(
            md,
            "| Motor | Max Amplitude | Oscillation | Peak in Range | Avg in Range |"
        )?;
        writeln!(
            md,
            "|-------|--------------|-------------|---------------|-------------|"
        )?;
        for r in &report.motor_results {
            let max_amp = r
                .max_amplitude
                .map_or("N/A".into(), |v| format!("{:.2}", v));
            let osc = if r.oscillation_detected {
                "⚠ Detected"
            } else {
                "None"
            };
            let peak = r
                .peak_in_range
                .map_or("N/A".into(), |v| format!("{:.2}", v));
            let avg = r.avg_in_range.map_or("N/A".into(), |v| format!("{:.2}", v));
            writeln!(
                md,
                "| {} | {} | {} | {} | {} |",
                r.motor_idx, max_amp, osc, peak, avg
            )?;
        }
        writeln!(md)?;
    }

    // --- Generated Plots ---
    if !report.png_links.is_empty() {
        writeln!(md, "## Generated Plots")?;
        writeln!(md)?;
        for name in &report.png_links {
            writeln!(md, "- [{}]({})", name, name)?;
        }
        writeln!(md)?;
    }

    fs::write(output_path, md)?;
    Ok(())
}

fn fmt_static_filter(f: &crate::data_analysis::filter_response::FilterConfig) -> String {
    match f.q_factor {
        Some(q) => format!(
            "{} @ {:.0} Hz (Q={:.2})",
            f.filter_type.name(),
            f.cutoff_hz,
            q
        ),
        None => format!("{} @ {:.0} Hz", f.filter_type.name(), f.cutoff_hz),
    }
}

fn fmt_filter_stage(
    lpf1: Option<&crate::data_analysis::filter_response::FilterConfig>,
    dyn_lpf: Option<&crate::data_analysis::filter_response::DynamicFilterConfig>,
) -> String {
    if let Some(f) = lpf1 {
        fmt_static_filter(f)
    } else if let Some(d) = dyn_lpf {
        format!(
            "Dyn {} {:.0}–{:.0} Hz",
            d.filter_type.name(),
            d.min_cutoff_hz,
            d.max_cutoff_hz
        )
    } else {
        "—".into()
    }
}

fn fmt_imuf(f: &crate::data_analysis::filter_response::ImufFilterConfig) -> String {
    if f.lowpass_cutoff_hz > 0.0 {
        let stages = match f.ptn_order {
            1 => "PT1",
            2 => "2×PT1",
            3 => "3×PT1",
            4 => "4×PT1",
            _ => "2×PT1",
        };
        let rev = f.revision.map_or("IMUF".into(), |r| format!("IMUF v{}", r));
        let w_part = f
            .pseudo_kalman_w
            .map_or(String::new(), |w| format!(", w={:.0}", w));
        format!(
            "{}: {} Combined={:.0} Hz per-stage={:.0} Hz (Q={:.1}{})",
            rev, stages, f.lowpass_cutoff_hz, f.effective_cutoff_hz, f.q_factor, w_part
        )
    } else {
        let w_part = f
            .pseudo_kalman_w
            .map_or(String::new(), |w| format!(", w={:.0}", w));
        format!("Pseudo-Kalman Q={:.1}{}", f.q_factor, w_part)
    }
}

fn fmt_dterm_rec(rec: &DTermRec) -> String {
    if rec.d_min.is_some() || rec.d_max.is_some() {
        let d_min_s = rec.d_min.map_or("N/A".to_string(), |v| v.to_string());
        let d_max_s = rec.d_max.map_or("N/A".to_string(), |v| v.to_string());
        format!(
            "P:D={:.2} (D-Min≈{}, D-Max≈{})",
            rec.pd_ratio, d_min_s, d_max_s
        )
    } else if let Some(d) = rec.d {
        format!("P:D={:.2} (D≈{})", rec.pd_ratio, d)
    } else {
        format!("P:D={:.2}", rec.pd_ratio)
    }
}
