// src/plot_functions/mod.rs

pub mod peak_detection;
pub mod plot_d_term_heatmap;
pub mod plot_d_term_psd;
pub mod plot_d_term_spectrums;
pub mod plot_gyro_spectrums;
pub mod plot_gyro_vs_unfilt;
pub mod plot_motor_spectrums;
pub mod plot_pidsum_error_setpoint;
pub mod plot_psd;
pub mod plot_psd_db_heatmap;
pub mod plot_setpoint_vs_gyro;
pub mod plot_step_response;
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
