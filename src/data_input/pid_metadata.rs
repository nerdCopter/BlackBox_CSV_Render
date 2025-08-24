// src/data_input/pid_metadata.rs

/// Firmware type detection for appropriate terminology
#[derive(Debug, Clone, PartialEq)]
pub enum FirmwareType {
    Betaflight,
    EmuFlight,
    Inav,
    Unknown,
}

impl Default for FirmwareType {
    fn default() -> Self {
        FirmwareType::Unknown
    }
}

/// PID values for a single axis
#[derive(Debug, Clone, Default)]
pub struct AxisPid {
    pub p: Option<u32>,
    pub i: Option<u32>,
    pub d: Option<u32>,
    pub ff: Option<u32>,
}

impl AxisPid {
    /// Format PID values for display with firmware-specific terminology
    pub fn format_for_title(&self, firmware_type: &FirmwareType) -> String {
        let mut parts = Vec::new();
        
        if let Some(p) = self.p {
            parts.push(format!("P:{}", p));
        }
        if let Some(i) = self.i {
            parts.push(format!("I:{}", i));
        }
        if let Some(d) = self.d {
            parts.push(format!("D:{}", d));
        }
        if let Some(ff) = self.ff {
            if ff > 0 {
                let ff_label = match firmware_type {
                    FirmwareType::EmuFlight => "DF",
                    _ => "FF",
                };
                parts.push(format!("{}:{}", ff_label, ff));
            }
        }
        
        if parts.is_empty() {
            String::new()
        } else {
            format!(" - {}", parts.join(" "))
        }
    }
}

/// PID metadata for all three axes (Roll, Pitch, Yaw) with firmware type
#[derive(Debug, Clone, Default)]
pub struct PidMetadata {
    pub roll: AxisPid,   // Axis 0
    pub pitch: AxisPid,  // Axis 1
    pub yaw: AxisPid,    // Axis 2
    pub firmware_type: FirmwareType,
}

impl PidMetadata {
    /// Get PID data for a specific axis (0=roll, 1=pitch, 2=yaw)
    pub fn get_axis(&self, axis_index: usize) -> &AxisPid {
        match axis_index {
            0 => &self.roll,
            1 => &self.pitch,
            2 => &self.yaw,
            _ => &self.roll, // Default to roll for invalid indices
        }
    }
    
    /// Get firmware type
    pub fn get_firmware_type(&self) -> &FirmwareType {
        &self.firmware_type
    }
}

/// Detect firmware type from header metadata
fn detect_firmware_type(header_metadata: &[(String, String)]) -> FirmwareType {
    let header_map: std::collections::HashMap<String, String> = header_metadata
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.to_lowercase()))
        .collect();
    
    // Check for firmware revision field
    if let Some(firmware_rev) = header_map.get("firmware revision") {
        if firmware_rev.contains("emuflight") {
            return FirmwareType::EmuFlight;
        }
        if firmware_rev.contains("betaflight") {
            return FirmwareType::Betaflight;
        }
        if firmware_rev.contains("inav") {
            return FirmwareType::Inav;
        }
    }
    
    // Check for firmware type field
    if let Some(firmware_type) = header_map.get("firmware type") {
        if firmware_type.contains("emuflight") {
            return FirmwareType::EmuFlight;
        }
        if firmware_type.contains("betaflight") {
            return FirmwareType::Betaflight;
        }
        if firmware_type.contains("inav") {
            return FirmwareType::Inav;
        }
    }
    
    // Check for EmuFlight-specific fields
    if header_map.contains_key("df_yaw") {
        return FirmwareType::EmuFlight;
    }
    
    // Check for Betaflight-specific fields  
    if header_map.contains_key("ff_weight") {
        return FirmwareType::Betaflight;
    }
    
    FirmwareType::Unknown
}

/// Parse PID metadata from header key-value pairs
/// Supports Betaflight, Emuflight, and INAV formats
/// Returns default/empty values if no metadata is available
pub fn parse_pid_metadata(header_metadata: &[(String, String)]) -> PidMetadata {
    let mut pid_data = PidMetadata::default();
    
    // Detect firmware type first
    pid_data.firmware_type = detect_firmware_type(header_metadata);
    
    // If no metadata available, return default (empty) values
    if header_metadata.is_empty() {
        return pid_data;
    }
    
    // Convert header metadata to a lookup map for easier access
    let header_map: std::collections::HashMap<String, String> = header_metadata
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.clone()))
        .collect();
    
    // Parse roll PID
    if let Some(roll_pid_str) = header_map.get("rollpid") {
        pid_data.roll = parse_axis_pid(roll_pid_str);
    }
    
    // Parse pitch PID
    if let Some(pitch_pid_str) = header_map.get("pitchpid") {
        pid_data.pitch = parse_axis_pid(pitch_pid_str);
    }
    
    // Parse yaw PID
    if let Some(yaw_pid_str) = header_map.get("yawpid") {
        pid_data.yaw = parse_axis_pid(yaw_pid_str);
    }
    
    // Handle FF values based on flight controller type
    
    // Betaflight style: ff_weight with roll,pitch,yaw values
    if let Some(ff_weight_str) = header_map.get("ff_weight") {
        let ff_values = parse_comma_separated_values(ff_weight_str);
        if ff_values.len() >= 3 {
            if ff_values[0] > 0 {
                pid_data.roll.ff = Some(ff_values[0]);
            }
            if ff_values[1] > 0 {
                pid_data.pitch.ff = Some(ff_values[1]);
            }
            if ff_values[2] > 0 {
                pid_data.yaw.ff = Some(ff_values[2]);
            }
        }
    }
    
    // Emuflight style: df_yaw for yaw feedforward only
    if let Some(df_yaw_str) = header_map.get("df_yaw") {
        if let Ok(df_yaw_val) = df_yaw_str.parse::<u32>() {
            if df_yaw_val > 0 {
                pid_data.yaw.ff = Some(df_yaw_val);
            }
        }
    }
    
    pid_data
}

/// Parse PID values from a string like "31,56,21" or "45,80,40,120" (INAV with FF)
fn parse_axis_pid(pid_str: &str) -> AxisPid {
    let values = parse_comma_separated_values(pid_str);
    
    let mut axis_pid = AxisPid::default();
    
    if !values.is_empty() {
        axis_pid.p = Some(values[0]);
    }
    if values.len() > 1 {
        axis_pid.i = Some(values[1]);
    }
    if values.len() > 2 {
        axis_pid.d = Some(values[2]);
    }
    if values.len() > 3 && values[3] > 0 {
        // INAV style with FF as 4th value
        axis_pid.ff = Some(values[3]);
    }
    
    axis_pid
}

/// Parse comma-separated values from a string
fn parse_comma_separated_values(value_str: &str) -> Vec<u32> {
    value_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_betaflight_parsing() {
        let metadata = vec![
            ("rollPID".to_string(), "31,56,21".to_string()),
            ("pitchPID".to_string(), "32,58,23".to_string()),
            ("yawPID".to_string(), "31,56,0".to_string()),
            ("ff_weight".to_string(), "84,87,84".to_string()),
        ];
        
        let pid_data = parse_pid_metadata(&metadata);
        
        assert_eq!(pid_data.roll.p, Some(31));
        assert_eq!(pid_data.roll.i, Some(56));
        assert_eq!(pid_data.roll.d, Some(21));
        assert_eq!(pid_data.roll.ff, Some(84));
        
        assert_eq!(pid_data.pitch.p, Some(32));
        assert_eq!(pid_data.pitch.i, Some(58));
        assert_eq!(pid_data.pitch.d, Some(23));
        assert_eq!(pid_data.pitch.ff, Some(87));
        
        assert_eq!(pid_data.yaw.p, Some(31));
        assert_eq!(pid_data.yaw.i, Some(56));
        assert_eq!(pid_data.yaw.d, Some(0));
        assert_eq!(pid_data.yaw.ff, Some(84));
    }

    #[test]
    fn test_emuflight_parsing() {
        let metadata = vec![
            ("rollPID".to_string(), "52,57,38".to_string()),
            ("pitchPID".to_string(), "62,57,44".to_string()),
            ("yawPID".to_string(), "90,90,7".to_string()),
            ("df_yaw".to_string(), "15".to_string()),
        ];
        
        let pid_data = parse_pid_metadata(&metadata);
        
        assert_eq!(pid_data.roll.p, Some(52));
        assert_eq!(pid_data.roll.i, Some(57));
        assert_eq!(pid_data.roll.d, Some(38));
        assert_eq!(pid_data.roll.ff, None);
        
        assert_eq!(pid_data.pitch.p, Some(62));
        assert_eq!(pid_data.pitch.i, Some(57));
        assert_eq!(pid_data.pitch.d, Some(44));
        assert_eq!(pid_data.pitch.ff, None);
        
        assert_eq!(pid_data.yaw.p, Some(90));
        assert_eq!(pid_data.yaw.i, Some(90));
        assert_eq!(pid_data.yaw.d, Some(7));
        assert_eq!(pid_data.yaw.ff, Some(15));
    }

    #[test]
    fn test_inav_parsing() {
        let metadata = vec![
            ("rollPID".to_string(), "45,80,40,120".to_string()),
            ("pitchPID".to_string(), "47,84,46,125".to_string()),
            ("yawPID".to_string(), "45,80,0,120".to_string()),
        ];
        
        let pid_data = parse_pid_metadata(&metadata);
        
        assert_eq!(pid_data.roll.p, Some(45));
        assert_eq!(pid_data.roll.i, Some(80));
        assert_eq!(pid_data.roll.d, Some(40));
        assert_eq!(pid_data.roll.ff, Some(120));
        
        assert_eq!(pid_data.pitch.p, Some(47));
        assert_eq!(pid_data.pitch.i, Some(84));
        assert_eq!(pid_data.pitch.d, Some(46));
        assert_eq!(pid_data.pitch.ff, Some(125));
        
        assert_eq!(pid_data.yaw.p, Some(45));
        assert_eq!(pid_data.yaw.i, Some(80));
        assert_eq!(pid_data.yaw.d, Some(0));
        assert_eq!(pid_data.yaw.ff, Some(120));
    }

    #[test]
    fn test_format_for_title() {
        let mut axis_pid = AxisPid::default();
        axis_pid.p = Some(31);
        axis_pid.i = Some(56);
        axis_pid.d = Some(21);
        axis_pid.ff = Some(84);
        
        // Test Betaflight formatting
        assert_eq!(axis_pid.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21 FF:84");
        
        // Test EmuFlight formatting
        assert_eq!(axis_pid.format_for_title(&FirmwareType::EmuFlight), " - P:31 I:56 D:21 DF:84");
        
        // Test with zero FF (should be omitted)
        axis_pid.ff = Some(0);
        assert_eq!(axis_pid.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
        
        // Test with no FF
        axis_pid.ff = None;
        assert_eq!(axis_pid.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
    }

    #[test]
    fn test_empty_metadata() {
        let empty_metadata = vec![];
        let pid_data = parse_pid_metadata(&empty_metadata);
        
        // Should return default values
        assert_eq!(pid_data.roll.p, None);
        assert_eq!(pid_data.roll.i, None);
        assert_eq!(pid_data.roll.d, None);
        assert_eq!(pid_data.roll.ff, None);
        
        // Title should be empty
        assert_eq!(pid_data.roll.format_for_title(&FirmwareType::Unknown), "");
    }
}
