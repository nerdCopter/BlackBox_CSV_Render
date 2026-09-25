// tests/report_filter_config_test.rs
// Verifies that the "IMUF / Pseudo-Kalman" column is only rendered in the
// generated `*_report.md` Gyro Filter Configuration table when at least one
// axis actually has an IMUF filter configured (i.e. EmuFlight logs), and
// that the section is titled "Gyro Filter Configuration".

use std::fs;

use BlackBox_CSV_Render::data_analysis::filter_response::{
    AllFilterConfigs, AxisFilterConfig, FilterConfig, FilterType, ImufFilterConfig,
};
use BlackBox_CSV_Render::report::{generate_markdown_report, FlightReport};

fn base_report(filter_config: Option<AllFilterConfigs>) -> FlightReport {
    FlightReport {
        root_name: "test_log".to_string(),
        sample_rate: Some(1000.0),
        trim_window: None,
        header_metadata: vec![],
        pd_ratios: [None, None, None],
        step_reports: vec![],
        optimal_p: [None, None, None],
        gyro_analysis: None,
        dterm_results: vec![],
        bode_results: vec![],
        motor_results: vec![],
        motor_desync_results: vec![],
        rc_command_steps: vec![],
        stick_distribution_results: vec![],
        png_links: vec![],
        skipped_plots: vec![],
        filter_config,
        dynamic_notch: None,
        rpm_filter: None,
        debug_fallback: false,
        debug_mode_name: None,
    }
}

fn lpf1_only_axis_config() -> AxisFilterConfig {
    AxisFilterConfig {
        lpf1: Some(FilterConfig {
            filter_type: FilterType::PT1,
            cutoff_hz: 250.0,
            q_factor: None,
            enabled: true,
        }),
        lpf2: None,
        dynamic_lpf1: None,
        imuf: None,
    }
}

#[test]
fn filter_config_table_omits_imuf_column_when_no_axis_has_imuf() {
    let mut fc = AllFilterConfigs::default();
    fc.gyro[0] = lpf1_only_axis_config();
    fc.gyro[1] = lpf1_only_axis_config();
    fc.gyro[2] = lpf1_only_axis_config();

    let report = base_report(Some(fc));
    let dir = tempfile::tempdir().expect("create temp dir");
    let out_path = dir.path().join("no_imuf_report.md");
    generate_markdown_report(&report, &out_path).expect("generate report");
    let contents = fs::read_to_string(&out_path).expect("read report");

    assert!(contents.contains("## Gyro Filter Configuration"));
    assert!(!contents.contains("IMUF / Pseudo-Kalman"));
    assert!(contents.contains("| Axis | LPF1 | LPF2 |"));
}

#[test]
fn filter_config_table_includes_imuf_column_when_an_axis_has_imuf() {
    let mut fc = AllFilterConfigs::default();
    fc.gyro[0] = AxisFilterConfig {
        lpf1: None,
        lpf2: None,
        dynamic_lpf1: None,
        imuf: Some(ImufFilterConfig {
            lowpass_cutoff_hz: 90.0,
            ptn_order: 1,
            q_factor: 200.0,
            revision: Some(256),
            pseudo_kalman_w: Some(4.0),
            effective_cutoff_hz: 90.0,
            enabled: true,
        }),
    };
    fc.gyro[1] = lpf1_only_axis_config();
    fc.gyro[2] = lpf1_only_axis_config();

    let report = base_report(Some(fc));
    let dir = tempfile::tempdir().expect("create temp dir");
    let out_path = dir.path().join("imuf_report.md");
    generate_markdown_report(&report, &out_path).expect("generate report");
    let contents = fs::read_to_string(&out_path).expect("read report");

    assert!(contents.contains("## Gyro Filter Configuration"));
    assert!(contents.contains("| Axis | LPF1 | LPF2 | IMUF / Pseudo-Kalman |"));
}
