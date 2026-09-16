// src/plot_functions/mod.rs

use crate::data_input::log_data::LogRowData;

pub mod motor_desync;
pub mod peak_detection;
pub mod plot_bode;
pub mod plot_d_term_heatmap;
pub mod plot_d_term_psd;
pub mod plot_d_term_spectrums;
pub mod plot_gyro_spectrums;
pub mod plot_gyro_vs_unfilt;
pub mod plot_motor_erpm;
pub mod plot_motor_spectrums;
pub mod plot_pid_activity;
pub mod plot_pidsum_error_setpoint;
pub mod plot_psd;
pub mod plot_psd_db_heatmap;
pub mod plot_rc_command_activity;
pub mod plot_setpoint_derivative;
pub mod plot_setpoint_vs_gyro;
pub mod plot_step_response;
pub mod plot_stick_distribution;
pub mod plot_throttle_freq_heatmap;

// Helper function for formatting debug suffix in plot labels
pub fn format_debug_suffix(
    base_label: &str,
    using_debug_fallback: bool,
    debug_mode_name: Option<&str>,
) -> String {
    if using_debug_fallback {
        if let Some(mode_name) = debug_mode_name {
            format!("{} [Debug={}]", base_label, mode_name)
        } else {
            format!("{} [Debug]", base_label)
        }
    } else {
        base_label.to_string()
    }
}

/// True when no row carries unfiltered gyro data for this axis — the parser has already
/// resolved a valid `debug[0-2]` fallback into `gyro_unfilt`, so this covers both the
/// no-header and the fallback-rejected cases in one check.
pub fn axis_lacks_unfiltered_data(log_data: &[LogRowData], axis_idx: usize) -> bool {
    !log_data
        .iter()
        .any(|row| row.gyro_unfilt[axis_idx].is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(gyro: Option<f64>, gyro_unfilt: Option<f64>) -> LogRowData {
        LogRowData {
            gyro: [gyro, None, None],
            gyro_unfilt: [gyro_unfilt, None, None],
            ..Default::default()
        }
    }

    #[test]
    fn no_unfiltered_data_anywhere_is_filtered_only() {
        let rows = vec![row(Some(1.0), None), row(Some(2.0), None), row(None, None)];
        assert!(axis_lacks_unfiltered_data(&rows, 0));
    }

    #[test]
    fn any_unfiltered_sample_disables_filtered_only() {
        let rows = vec![
            row(Some(1.0), None),
            row(Some(2.0), Some(2.0)),
            row(None, None),
        ];
        assert!(!axis_lacks_unfiltered_data(&rows, 0));
    }

    #[test]
    fn unfiltered_present_on_rows_with_no_matching_filtered_sample() {
        // Unfiltered data exists on this axis, but never on a row that also has filtered
        // data. `axis_lacks_unfiltered_data` still reports paired mode (matches the
        // preserved paired-data contract) — the matched-pair extraction then correctly
        // yields zero samples for this axis rather than fabricating a pairing.
        let rows = vec![row(Some(1.0), None), row(None, Some(9.0))];
        assert!(!axis_lacks_unfiltered_data(&rows, 0));
    }

    #[test]
    fn other_axes_are_independent() {
        let rows = vec![LogRowData {
            gyro: [Some(1.0), Some(1.0), Some(1.0)],
            gyro_unfilt: [Some(1.0), None, None],
            ..Default::default()
        }];
        // Only the first axis in the fixture carries unfiltered data.
        for (axis_idx, axis_name) in crate::axis_names::AXIS_NAMES.iter().enumerate() {
            let expected = axis_idx != 0;
            assert_eq!(
                axis_lacks_unfiltered_data(&rows, axis_idx),
                expected,
                "{axis_name} axis mismatch"
            );
        }
    }
}
