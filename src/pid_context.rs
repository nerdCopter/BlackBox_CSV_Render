// src/pid_context.rs

use crate::data_input::pid_metadata::PidMetadata;

/// Context struct containing PID metadata and related parameters for plotting functions
/// This centralizes PID-related data and makes it easier to extend functionality
/// without breaking existing function signatures
#[derive(Debug, Clone)]
pub struct PidContext {
    /// Sample rate for time-based calculations and axis labeling
    #[allow(dead_code)] // Future use in plot functions
    pub sample_rate: Option<f64>,
    
    /// PID metadata extracted from blackbox headers (firmware type, PID values)
    pub pid_metadata: PidMetadata,
    
    /// Root filename (without path/extension) for output file naming
    #[allow(dead_code)] // Future use in plot functions
    pub root_name: String,
}

impl PidContext {
    /// Create a new PidContext with the provided parameters
    pub fn new(
        sample_rate: Option<f64>,
        pid_metadata: PidMetadata,
        root_name: String,
    ) -> Self {
        Self {
            sample_rate,
            pid_metadata,
            root_name,
        }
    }
    
    /// Get axis name for display purposes
    #[allow(dead_code)] // Future use in plot functions
    pub fn get_axis_name(&self, axis_index: usize) -> &'static str {
        match axis_index {
            0 => "Roll",
            1 => "Pitch", 
            2 => "Yaw",
            _ => panic!("Invalid axis index: {}. Expected 0 (roll), 1 (pitch), or 2 (yaw)", axis_index),
        }
    }
    
    /// Get firmware-specific axis title with PID information
    #[allow(dead_code)] // Future use in plot functions
    pub fn get_axis_title_with_pids(&self, axis_index: usize) -> String {
        let axis_name = self.get_axis_name(axis_index);
        let axis_pid = self.pid_metadata.get_axis(axis_index);
        let firmware_type = self.pid_metadata.get_firmware_type();
        let pid_info = axis_pid.format_for_title(firmware_type);
        
        if pid_info.is_empty() {
            axis_name.to_string()
        } else {
            format!("{} ({})", axis_name, pid_info)
        }
    }
}
