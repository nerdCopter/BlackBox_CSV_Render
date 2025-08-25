// src/data_input/pid_metadata.rs

/// Firmware type detection for appropriate terminology
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FirmwareType {
    Betaflight,
    EmuFlight,
    Inav,
    #[default]
    Unknown,
}

/// PID values for a single axis
#[derive(Debug, Clone, Default)]
pub struct AxisPid {
    pub p: Option<u32>,
    pub i: Option<u32>,
    pub d: Option<u32>,
    pub d_min: Option<u32>,  // D-Min for Betaflight
    pub d_max: Option<u32>,  // D-Max for newer Betaflight
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
        
        // Handle D, D-Min, and D-Max formatting
        match (self.d, self.d_min, self.d_max) {
            (_, Some(d_min), Some(d_max)) if d_min != d_max && d_min > 0 && d_max > 0 => {
                // Show D:min/max format when D-Min and D-Max are different and both non-zero
                parts.push(format!("D:{}/{}", d_min, d_max));
            }
            (_, Some(d_min), Some(d_max)) if d_min != d_max && d_min > 0 && d_max == 0 => {
                // Show D:min format when D-Max is zero (don't show /0)
                parts.push(format!("D:{}", d_min));
            }
            (_, Some(d_min), Some(d_max)) if d_min != d_max && d_min == 0 && d_max > 0 => {
                // Show D:max format when D-Min is zero (don't show 0/)
                parts.push(format!("D:{}", d_max));
            }
            (Some(d), Some(_d_min), Some(_d_max)) => {
                // Show D:XX format when D-Min and D-Max are the same (use actual D value)
                parts.push(format!("D:{}", d));
            }
            (Some(d), None, Some(d_max)) if d != d_max && d_max > 0 => {
                // Show D:XX/XX format when only D-Max is available, different from D, and non-zero
                parts.push(format!("D:{}/{}", d, d_max));
            }
            (Some(d), _, _) => {
                // Show simple D:XX when no D-Min/D-Max, they're the same as D, or involve zeros
                parts.push(format!("D:{}", d));
            }
            _ => {}
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
    /// Returns None if axis_index is invalid (not 0, 1, or 2)
    pub fn get_axis(&self, axis_index: usize) -> Option<&AxisPid> {
        match axis_index {
            0 => Some(&self.roll),
            1 => Some(&self.pitch),
            2 => Some(&self.yaw),
            _ => None,
        }
    }
    
    /// Get firmware type
    pub fn get_firmware_type(&self) -> &FirmwareType {
        &self.firmware_type
    }
}

/// Detect firmware type from header metadata map
fn detect_firmware_type(header_map: &std::collections::HashMap<String, String>) -> FirmwareType {
    // Check for firmware revision field (primary detection method)
    if let Some(firmware_rev) = header_map.get("firmware revision") {
        let normalized_rev = firmware_rev.trim().to_lowercase();
        if normalized_rev.contains("emuflight") {
            return FirmwareType::EmuFlight;
        }
        if normalized_rev.contains("betaflight") {
            return FirmwareType::Betaflight;
        }
        if normalized_rev.contains("inav") {
            return FirmwareType::Inav;
        }
    }
    
    // Fallback: Check for firmware type field (secondary, if revision missing)
    if let Some(firmware_type) = header_map.get("firmware type") {
        let normalized_type = firmware_type.trim().to_lowercase();
        if normalized_type.contains("emuflight") {
            return FirmwareType::EmuFlight;
        }
        if normalized_type.contains("betaflight") {
            return FirmwareType::Betaflight;
        }
        if normalized_type.contains("inav") {
            return FirmwareType::Inav;
        }
    }
    
    // Fallback: Check for firmware-specific fields
    if header_map.contains_key("df_yaw") {
        return FirmwareType::EmuFlight;
    }
    
    if header_map.contains_key("ff_weight") {
        return FirmwareType::Betaflight;
    }
    
    FirmwareType::Unknown
}

/// Helper function to parse D-Min or D-Max values for all axes
fn parse_d_values(
    header_map: &std::collections::HashMap<String, String>,
    comma_separated_key: &str,
    individual_keys: [&str; 6], // [roll_key1, roll_key2, pitch_key1, pitch_key2, yaw_key1, yaw_key2]
) -> [Option<u32>; 3] {
    let mut values = [None; 3];
    
    // Try comma-separated format first
    if let Some(value_str) = header_map.get(comma_separated_key) {
        let parsed_values = parse_comma_separated_values(value_str);
        if parsed_values.len() >= 3 {
            values[0] = Some(parsed_values[0]);
            values[1] = Some(parsed_values[1]);
            values[2] = Some(parsed_values[2]);
            return values;
        }
    }
    
    // Fallback to individual fields
    for axis in 0..3 {
        let key1 = individual_keys[axis * 2];
        let key2 = individual_keys[axis * 2 + 1];
        
        if let Some(value_str) = header_map.get(key1).or_else(|| header_map.get(key2)) {
            if let Ok(value) = value_str.parse::<u32>() {
                values[axis] = Some(value);
            }
        }
    }
    
    values
}

/// Helper function to parse FF values for all axes
fn parse_ff_values(
    header_map: &std::collections::HashMap<String, String>,
    existing_pids: &(AxisPid, AxisPid, AxisPid),
) -> [Option<u32>; 3] {
    let mut ff_values = [
        existing_pids.0.ff, // Roll FF from PID string
        existing_pids.1.ff, // Pitch FF from PID string
        existing_pids.2.ff, // Yaw FF from PID string
    ];
    
    // Betaflight style: ff_weight with roll,pitch,yaw values (overrides PID string FF)
    if let Some(ff_weight_str) = header_map.get("ff_weight") {
        let values = parse_comma_separated_values(ff_weight_str);
        if values.len() >= 3 {
            for (i, &value) in values.iter().enumerate().take(3) {
                if value > 0 {
                    ff_values[i] = Some(value);
                }
            }
        }
    }
    
    // Emuflight style: df_yaw for yaw feedforward only (overrides PID string FF for yaw)
    if let Some(df_yaw_str) = header_map.get("df_yaw") {
        if let Ok(df_yaw_val) = df_yaw_str.parse::<u32>() {
            if df_yaw_val > 0 {
                ff_values[2] = Some(df_yaw_val); // Yaw is index 2
            }
        }
    }
    
    ff_values
}

/// Helper function to parse axis PID values from header map
fn parse_axis_pids(header_map: &std::collections::HashMap<String, String>) -> (AxisPid, AxisPid, AxisPid) {
    let roll = if let Some(roll_pid_str) = header_map.get("rollpid") {
        parse_axis_pid(roll_pid_str)
    } else {
        AxisPid::default()
    };
    
    let pitch = if let Some(pitch_pid_str) = header_map.get("pitchpid") {
        parse_axis_pid(pitch_pid_str)
    } else {
        AxisPid::default()
    };
    
    let yaw = if let Some(yaw_pid_str) = header_map.get("yawpid") {
        parse_axis_pid(yaw_pid_str)
    } else {
        AxisPid::default()
    };
    
    (roll, pitch, yaw)
}

/// Parse PID metadata from header key-value pairs
/// Supports Betaflight, Emuflight, and INAV formats
/// Returns default/empty values if no metadata is available
pub fn parse_pid_metadata(header_metadata: &[(String, String)]) -> PidMetadata {
    let mut pid_data = PidMetadata::default();
    
    // If no metadata available, return default (empty) values
    if header_metadata.is_empty() {
        return pid_data;
    }
    
    // Convert header metadata to a lookup map for easier access (build once, use everywhere)
    let header_map: std::collections::HashMap<String, String> = header_metadata
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.clone()))
        .collect();
    
    // Detect firmware type using the same map
    pid_data.firmware_type = detect_firmware_type(&header_map);
    
    // Parse axis PID values
    let (roll, pitch, yaw) = parse_axis_pids(&header_map);
    pid_data.roll = roll;
    pid_data.pitch = pitch;
    pid_data.yaw = yaw;
    
    // Handle FF values based on flight controller type
    let ff_values = parse_ff_values(&header_map, &(pid_data.roll.clone(), pid_data.pitch.clone(), pid_data.yaw.clone()));
    pid_data.roll.ff = ff_values[0];
    pid_data.pitch.ff = ff_values[1];
    pid_data.yaw.ff = ff_values[2];
    
    // Look for separate D-Min fields in header metadata (Betaflight separate headers)
    // Only override PID string D-Min values if separate fields exist
    let d_min_values = parse_d_values(
        &header_map,
        "d_min",
        ["rolldmin", "roll_d_min", "pitchdmin", "pitch_d_min", "yawdmin", "yaw_d_min"],
    );
    if d_min_values[0].is_some() {
        pid_data.roll.d_min = d_min_values[0];
    }
    if d_min_values[1].is_some() {
        pid_data.pitch.d_min = d_min_values[1];
    }
    if d_min_values[2].is_some() {
        pid_data.yaw.d_min = d_min_values[2];
    }
    
    // Look for separate D-Max fields in header metadata (newer Betaflight separate headers)
    // Only override PID string D-Max values if separate fields exist
    let d_max_values = parse_d_values(
        &header_map,
        "d_max",
        ["rolldmax", "roll_d_max", "pitchdmax", "pitch_d_max", "yawdmax", "yaw_d_max"],
    );
    if d_max_values[0].is_some() {
        pid_data.roll.d_max = d_max_values[0];
    }
    if d_max_values[1].is_some() {
        pid_data.pitch.d_max = d_max_values[1];
    }
    if d_max_values[2].is_some() {
        pid_data.yaw.d_max = d_max_values[2];
    }
    
    pid_data
}

/// Parse PID values from a string like "31,56,21" (basic) or "45,80,40,120" (INAV with FF) 
/// or "57,66,58,58,206" (Betaflight 4.6+ with P,I,D,D-Max,FF)
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
    
    // Handle different formats based on value count
    match values.len() {
        4 => {
            // INAV style: P,I,D,FF
            if values[3] > 0 {
                axis_pid.ff = Some(values[3]);
            }
        }
        5 => {
            // Betaflight 4.6+ style: P,I,D,D-Max,FF
            axis_pid.d_max = Some(values[3]);
            if values[4] > 0 {
                axis_pid.ff = Some(values[4]);
            }
        }
        _ => {
            // 3 or fewer values: basic P,I,D format (no FF or D-Max)
        }
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
    fn test_betaflight_5_value_parsing() {
        // Test Betaflight 4.6+ format: P,I,D,D-Max,FF
        let metadata = vec![
            ("rollPID".to_string(), "57,66,58,58,206".to_string()),
            ("pitchPID".to_string(), "59,69,72,72,215".to_string()),
            ("yawPID".to_string(), "57,66,0,0,206".to_string()),
        ];
        
        let pid_data = parse_pid_metadata(&metadata);
        
        assert_eq!(pid_data.roll.p, Some(57));
        assert_eq!(pid_data.roll.i, Some(66));
        assert_eq!(pid_data.roll.d, Some(58));
        assert_eq!(pid_data.roll.d_max, Some(58));
        assert_eq!(pid_data.roll.ff, Some(206));
        
        assert_eq!(pid_data.pitch.p, Some(59));
        assert_eq!(pid_data.pitch.i, Some(69));
        assert_eq!(pid_data.pitch.d, Some(72));
        assert_eq!(pid_data.pitch.d_max, Some(72));
        assert_eq!(pid_data.pitch.ff, Some(215));
        
        assert_eq!(pid_data.yaw.p, Some(57));
        assert_eq!(pid_data.yaw.i, Some(66));
        assert_eq!(pid_data.yaw.d, Some(0));
        assert_eq!(pid_data.yaw.d_max, Some(0));
        assert_eq!(pid_data.yaw.ff, Some(206));
    }

    #[test]
    fn test_betaflight_d_min_d_max_parsing() {
        // Test Betaflight 4.5 format with separate d_min and d_max fields
        let metadata = vec![
            ("rollPID".to_string(), "57,66,58".to_string()),
            ("pitchPID".to_string(), "59,69,72".to_string()),
            ("yawPID".to_string(), "57,66,0".to_string()),
            ("d_min".to_string(), "39,44,0".to_string()),
            ("d_max".to_string(), "80,90,0".to_string()),
            ("ff_weight".to_string(), "206,215,206".to_string()),
        ];
        
        let pid_data = parse_pid_metadata(&metadata);
        
        assert_eq!(pid_data.roll.p, Some(57));
        assert_eq!(pid_data.roll.i, Some(66));
        assert_eq!(pid_data.roll.d, Some(58));
        assert_eq!(pid_data.roll.d_min, Some(39));
        assert_eq!(pid_data.roll.d_max, Some(80));
        assert_eq!(pid_data.roll.ff, Some(206));
        
        assert_eq!(pid_data.pitch.d_min, Some(44));
        assert_eq!(pid_data.pitch.d_max, Some(90));
        
        assert_eq!(pid_data.yaw.d_min, Some(0));
        assert_eq!(pid_data.yaw.d_max, Some(0));
        
        // VERIFY D:XX/XX FORMATTING IS WORKING
        let roll_formatted = pid_data.roll.format_for_title(&FirmwareType::Betaflight);
        assert_eq!(roll_formatted, " - P:57 I:66 D:39/80 FF:206");
        
        let pitch_formatted = pid_data.pitch.format_for_title(&FirmwareType::Betaflight);
        assert_eq!(pitch_formatted, " - P:59 I:69 D:44/90 FF:215");
    }

    #[test]
    fn test_format_for_title() {
        let axis_pid = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            ff: Some(84),
            ..Default::default()
        };
        
        // Test Betaflight formatting
        assert_eq!(axis_pid.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21 FF:84");
        
        // Test EmuFlight formatting
        assert_eq!(axis_pid.format_for_title(&FirmwareType::EmuFlight), " - P:31 I:56 D:21 DF:84");
        
        // Test with zero FF (should be omitted)
        let axis_pid_zero_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            ff: Some(0),
            ..Default::default()
        };
        assert_eq!(axis_pid_zero_ff.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
        
        // Test with no FF
        let axis_pid_no_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            ff: None,
            ..Default::default()
        };
        assert_eq!(axis_pid_no_ff.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
        
        // Test D:XX/XX format when D and D-Max are different
        let axis_pid_diff_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_max: Some(35),
            ff: None,
            ..Default::default()
        };
        assert_eq!(axis_pid_diff_dmax.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21/35");
        
        // Test D:XX format when D and D-Max are the same
        let axis_pid_same_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_max: Some(21),
            ff: None,
            ..Default::default()
        };
        assert_eq!(axis_pid_same_dmax.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
        
        // Test D:min/max format when D-Min and D-Max are different
        let axis_pid_dmin_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(axis_pid_dmin_dmax.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:15/35");
        
        // Test D:min/max format with FF
        let axis_pid_dmin_dmax_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: Some(84),
        };
        assert_eq!(axis_pid_dmin_dmax_ff.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:15/35 FF:84");
    }

    #[test]
    fn test_format_for_title_zero_handling() {
        // Test D:XX/0 should become D:XX (don't show /0)
        let axis_pid_dmax_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(0), // D-Max is zero
            ff: None,
        };
        assert_eq!(axis_pid_dmax_zero.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:15");
        
        // Test 0/XX should become D:XX (don't show 0/)
        let axis_pid_dmin_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(0), // D-Min is zero
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(axis_pid_dmin_zero.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:35");
        
        // Test D:XX/0 with only D-Max (no D-Min) should become D:XX
        let axis_pid_only_dmax_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: None,
            d_max: Some(0), // D-Max is zero
            ff: None,
        };
        assert_eq!(axis_pid_only_dmax_zero.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:21");
        
        // Test normal case still works (both non-zero and different)
        let axis_pid_normal = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(axis_pid_normal.format_for_title(&FirmwareType::Betaflight), " - P:31 I:56 D:15/35");
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

    #[test]
    fn test_firmware_detection_case_insensitive() {
        // Test case-insensitive firmware detection with various whitespace and case combinations
        
        // Betaflight variations
        let metadata_bf1 = vec![("firmware revision".to_string(), "BETAFLIGHT 4.6.0".to_string())];
        let pid_data_bf1 = parse_pid_metadata(&metadata_bf1);
        assert_eq!(pid_data_bf1.firmware_type, FirmwareType::Betaflight);
        
        let metadata_bf2 = vec![("firmware revision".to_string(), "  betaflight 4.5.1  ".to_string())];
        let pid_data_bf2 = parse_pid_metadata(&metadata_bf2);
        assert_eq!(pid_data_bf2.firmware_type, FirmwareType::Betaflight);
        
        let metadata_bf3 = vec![("firmware revision".to_string(), "BetaFlight Custom Build".to_string())];
        let pid_data_bf3 = parse_pid_metadata(&metadata_bf3);
        assert_eq!(pid_data_bf3.firmware_type, FirmwareType::Betaflight);
        
        // EmuFlight variations
        let metadata_ef1 = vec![("firmware revision".to_string(), "EMUFLIGHT 0.4.2".to_string())];
        let pid_data_ef1 = parse_pid_metadata(&metadata_ef1);
        assert_eq!(pid_data_ef1.firmware_type, FirmwareType::EmuFlight);
        
        let metadata_ef2 = vec![("firmware revision".to_string(), "  EmuFlight Beta  ".to_string())];
        let pid_data_ef2 = parse_pid_metadata(&metadata_ef2);
        assert_eq!(pid_data_ef2.firmware_type, FirmwareType::EmuFlight);
        
        // INAV variations
        let metadata_inav1 = vec![("firmware revision".to_string(), "INAV 8.0.0".to_string())];
        let pid_data_inav1 = parse_pid_metadata(&metadata_inav1);
        assert_eq!(pid_data_inav1.firmware_type, FirmwareType::Inav);
        
        let metadata_inav2 = vec![("firmware revision".to_string(), "  iNav Latest  ".to_string())];
        let pid_data_inav2 = parse_pid_metadata(&metadata_inav2);
        assert_eq!(pid_data_inav2.firmware_type, FirmwareType::Inav);
        
        // Fallback to firmware type field
        let metadata_type = vec![("firmware type".to_string(), "  BETAFLIGHT  ".to_string())];
        let pid_data_type = parse_pid_metadata(&metadata_type);
        assert_eq!(pid_data_type.firmware_type, FirmwareType::Betaflight);
        
        // Field-based detection (when text-based fails)
        let metadata_field = vec![("df_yaw".to_string(), "15".to_string())];
        let pid_data_field = parse_pid_metadata(&metadata_field);
        assert_eq!(pid_data_field.firmware_type, FirmwareType::EmuFlight);
    }
}
