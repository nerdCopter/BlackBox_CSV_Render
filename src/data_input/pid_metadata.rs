// src/data_input/pid_metadata.rs

// Import AXIS_COUNT for non-test builds
#[cfg(not(test))]
use crate::axis_names::AXIS_COUNT;

// For test builds (when included via include!()), define locally to match axis_names
#[cfg(test)]
const AXIS_COUNT: usize = 3; // Keep in sync with crate::axis_names::AXIS_COUNT

// Axis indices for array access consistency
const ROLL_AXIS: usize = 0;
const PITCH_AXIS: usize = 1;
const YAW_AXIS: usize = 2;

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
    pub d_min: Option<u32>, // D-Min for Betaflight
    pub d_max: Option<u32>, // D-Max for newer Betaflight
    pub ff: Option<u32>,
}

impl AxisPid {
    /// Calculate P:D ratio (P divided by D)
    /// Returns None if P or D is missing, or if D is 0
    /// Uses base D value (not D-Min) when D-Min/D-Max is configured
    /// This represents the "setpoint" D value that D-Min/D-Max modulates around
    pub fn calculate_pd_ratio(&self) -> Option<f64> {
        let p_val = self.p?;

        // Use base D value - this is what the user sets and what D-Min/D-Max modulate around
        // When D-Min/D-Max are disabled (both 0 or one is 0), base D is the actual D
        // When D-Min/D-Max are enabled, base D is still the reference value for tuning
        let d_val = self.d?;

        if d_val == 0 {
            return None; // Avoid division by zero
        }

        Some((p_val as f64) / (d_val as f64))
    }

    /// Calculate the recommended D value to achieve a target P:D ratio
    /// Returns None if P is missing
    /// Default target ratio is 1.4 (recommended for good damping)
    pub fn calculate_goal_d_for_ratio(&self, target_ratio: f64) -> Option<u32> {
        let p_val = self.p?;

        if target_ratio <= 0.0 {
            return None; // Invalid target ratio
        }

        // Calculate D = P / target_ratio
        let recommended_d = (p_val as f64) / target_ratio;
        Some(recommended_d.round() as u32)
    }

    /// Calculate recommended D, D-Min, and D-Max values to achieve a target P:D ratio
    /// Maintains the same proportional relationship between base D, D-Min, and D-Max
    /// Returns (base_d, d_min, d_max) tuple - any value can be None if not calculable
    ///
    /// Note: For Betaflight < 4.6, base D = D-Max (maximum). For BF >= 4.6, base D = D-Min (minimum).
    /// This function calculates proportionally regardless of firmware version.
    ///
    /// When dmax_enabled is false, returns (base_d, None, None) regardless of d_min/d_max field presence.
    pub fn calculate_goal_d_with_range(
        &self,
        target_ratio: f64,
        dmax_enabled: bool,
    ) -> (Option<u32>, Option<u32>, Option<u32>) {
        let recommended_base_d = self.calculate_goal_d_for_ratio(target_ratio);

        // If we can't calculate base D or don't have current D, return early
        let Some(rec_base) = recommended_base_d else {
            return (None, None, None);
        };
        let Some(current_base) = self.d else {
            return (recommended_base_d, None, None);
        };

        if current_base == 0 {
            return (recommended_base_d, None, None);
        }

        // If D-Min/D-Max system is disabled, only return base D
        if !dmax_enabled {
            return (recommended_base_d, None, None);
        }

        // Calculate the scaling factor
        let scale_factor = (rec_base as f64) / (current_base as f64);

        // Calculate recommended D-Min if present
        let rec_d_min = self.d_min.map(|current_min| {
            if current_min > 0 {
                ((current_min as f64) * scale_factor).round() as u32
            } else {
                0
            }
        });

        // Calculate recommended D-Max
        let rec_d_max = if let Some(current_max) = self.d_max {
            // D-Max field exists (BF 4.6+)
            Some(if current_max > 0 {
                ((current_max as f64) * scale_factor).round() as u32
            } else {
                0
            })
        } else if self.d_min.is_some() {
            // D-Max field missing but D-Min exists (older firmware < 4.6)
            // In old firmware, base D IS the D-Max, so use recommended base D as D-Max
            recommended_base_d // Base D = D-Max in old firmware
        } else {
            // No D-Min/D-Max system at all
            None
        };

        (recommended_base_d, rec_d_min, rec_d_max)
    }

    /// Format D term with dynamic range support (D-Min/D-Max)
    ///
    /// Decision tree when D-Max system is enabled:
    /// 1. If D-Min != D-Max and both non-zero → show "D:min/max" range
    /// 2. If only D-Min or D-Max is non-zero → show that single value
    /// 3. If D-Min == D-Max and both non-zero → show base D value (or D-Min if D is missing)
    /// 4. Fallback to base D value if available
    fn format_d_term(d: Option<u32>, d_min: Option<u32>, d_max: Option<u32>) -> Option<String> {
        match (d_min, d_max) {
            // Both D-Min and D-Max available and different - show range
            (Some(min), Some(max)) if min != max && min > 0 && max > 0 => {
                Some(format!("D:{min}/{max}"))
            }
            // D-Min non-zero but D-Max is zero - show only D-Min
            (Some(min), Some(0)) if min > 0 => Some(format!("D:{min}")),
            // D-Max non-zero but D-Min is zero - show only D-Max
            (Some(0), Some(max)) if max > 0 => Some(format!("D:{max}")),
            // D-Min and D-Max are equal and non-zero - prefer base D, fallback to D-Min
            (Some(min), Some(max)) if min == max && min > 0 => {
                Some(format!("D:{}", d.unwrap_or(min)))
            }
            // Only D-Min present (no D-Max)
            // If base D exists and differs, show min/base (min first to maintain min/max ordering)
            (Some(min), None) if min > 0 => match d {
                Some(base) if base != min => Some(format!("D:{min}/{base}")),
                _ => Some(format!("D:{min}")),
            },
            // Only D-Max present and base D is missing - show D-Max directly
            (None, Some(max)) if d.is_none() && max > 0 => Some(format!("D:{max}")),
            // Only D-Max available (no D-Min) and different from base D
            (None, Some(max)) if d.is_some() && d != Some(max) && max > 0 => {
                Some(format!("D:{}/{max}", d.unwrap()))
            }
            // Fallback to base D value
            _ => d.map(|v| format!("D:{v}")),
        }
    }

    /// Format PID values for display with firmware-specific terminology
    /// dmax_enabled: Set to true only if D-Min/D-Max system is actively enabled
    pub fn format_for_title(&self, firmware_type: &FirmwareType, dmax_enabled: bool) -> String {
        self.format_for_title_with_ratio(firmware_type, dmax_enabled, false)
    }

    /// Format PID values for display with firmware-specific terminology and optional P:D ratio
    /// dmax_enabled: Set to true only if D-Min/D-Max system is actively enabled
    /// If show_pd_ratio is true and ratio can be calculated, appends P:D ratio
    pub fn format_for_title_with_ratio(
        &self,
        firmware_type: &FirmwareType,
        dmax_enabled: bool,
        show_pd_ratio: bool,
    ) -> String {
        let mut parts = Vec::new();

        if let Some(p) = self.p {
            parts.push(format!("P:{p}"));
        }
        if let Some(i) = self.i {
            parts.push(format!("I:{i}"));
        }

        // Handle D, D-Min, and D-Max formatting
        if dmax_enabled {
            if let Some(d_str) = Self::format_d_term(self.d, self.d_min, self.d_max) {
                parts.push(d_str);
            }
        } else {
            // D-Max system disabled - show only base D value
            if let Some(d) = self.d {
                parts.push(format!("D:{d}"));
            }
        }

        // Add FF before P:D ratio
        if let Some(ff) = self.ff {
            if ff > 0 {
                let ff_label = match firmware_type {
                    FirmwareType::EmuFlight => "DF",
                    _ => "FF",
                };
                parts.push(format!("{ff_label}:{ff}"));
            }
        }

        // Add P:D ratio if requested and calculable (separated by hyphen for readability)
        if show_pd_ratio {
            if let Some(pd_ratio) = self.calculate_pd_ratio() {
                if parts.is_empty() {
                    return format!(" - P:D={pd_ratio:.2}");
                } else {
                    return format!(" - {} - P:D={pd_ratio:.2}", parts.join(" "));
                }
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
    pub roll: AxisPid,  // Axis 0
    pub pitch: AxisPid, // Axis 1
    pub yaw: AxisPid,   // Axis 2
    pub firmware_type: FirmwareType,
    // D-Min/D-Max control parameters - used to determine if dynamic D is enabled
    pub d_max_gain: Option<u32>, // BF 4.5+: Controls gyro-based D boost sensitivity
    pub d_max_advance: Option<u32>, // BF 4.5+: Controls setpoint-based D boost sensitivity
    pub simplified_d_max_gain: Option<u32>, // BF 4.6+: Simplified tuning D-Max gain
}

impl PidMetadata {
    /// Get PID data for a specific axis (ROLL_AXIS=0, PITCH_AXIS=1, YAW_AXIS=2)
    /// Returns None if axis_index is invalid
    #[allow(dead_code)]
    pub fn get_axis(&self, axis_index: usize) -> Option<&AxisPid> {
        match axis_index {
            ROLL_AXIS => Some(&self.roll),
            PITCH_AXIS => Some(&self.pitch),
            YAW_AXIS => Some(&self.yaw),
            _ => None,
        }
    }

    /// Get firmware type
    #[allow(dead_code)]
    pub fn get_firmware_type(&self) -> &FirmwareType {
        &self.firmware_type
    }

    /// Check if D-Min/D-Max dynamic D is enabled
    /// Returns true if any control parameter (gain, advance, or simplified gain) is non-zero
    /// AND there's actual dynamic range on roll/pitch axes.
    ///
    /// Dynamic range checks:
    /// - BF 4.6+ (d_max field exists): D-Min ≠ D-Max
    /// - BF <4.6 (no d_max field): D ≠ D-Min (where D is the max)
    ///
    /// When values are equal, there's no dynamic range and system is effectively disabled.
    pub fn is_dmax_enabled(&self) -> bool {
        // Check if control parameters are non-zero
        let has_active_params = self.d_max_gain.is_some_and(|g| g > 0)
            || self.d_max_advance.is_some_and(|a| a > 0)
            || self.simplified_d_max_gain.is_some_and(|sg| sg > 0);

        if !has_active_params {
            return false;
        }

        // Check roll axis for dynamic range
        let roll_has_range = if self.roll.d_max.is_some() {
            // BF 4.6+: Check if D-Min ≠ D-Max
            self.roll
                .d_min
                .and_then(|dmin| self.roll.d_max.map(|dmax| dmin != dmax))
                .unwrap_or(false)
        } else {
            // BF <4.6: Check if D ≠ D-Min (D is the max in old firmware)
            self.roll
                .d
                .and_then(|d| self.roll.d_min.map(|dmin| d != dmin))
                .unwrap_or(false)
        };

        // Check pitch axis for dynamic range
        let pitch_has_range = if self.pitch.d_max.is_some() {
            // BF 4.6+: Check if D-Min ≠ D-Max
            self.pitch
                .d_min
                .and_then(|dmin| self.pitch.d_max.map(|dmax| dmin != dmax))
                .unwrap_or(false)
        } else {
            // BF <4.6: Check if D ≠ D-Min (D is the max in old firmware)
            self.pitch
                .d
                .and_then(|d| self.pitch.d_min.map(|dmin| d != dmin))
                .unwrap_or(false)
        };

        // System is enabled if either roll or pitch has dynamic range
        roll_has_range || pitch_has_range
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
) -> [Option<u32>; AXIS_COUNT] {
    let mut values = [None; AXIS_COUNT];

    // Try comma-separated format first
    if let Some(value_str) = header_map.get(comma_separated_key) {
        let parsed_values = parse_comma_separated_values(value_str);
        if parsed_values.len() >= AXIS_COUNT {
            values[ROLL_AXIS] = Some(parsed_values[ROLL_AXIS]);
            values[PITCH_AXIS] = Some(parsed_values[PITCH_AXIS]);
            values[YAW_AXIS] = Some(parsed_values[YAW_AXIS]);
            return values;
        }
    }

    // Fallback to individual fields
    for axis in 0..AXIS_COUNT {
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
    existing_pids: &(&AxisPid, &AxisPid, &AxisPid),
) -> [Option<u32>; AXIS_COUNT] {
    let mut ff_values = [
        existing_pids.0.ff, // Roll FF from PID string
        existing_pids.1.ff, // Pitch FF from PID string
        existing_pids.2.ff, // Yaw FF from PID string
    ];

    // Betaflight style: ff_weight with roll,pitch,yaw values (overrides PID string FF)
    if let Some(ff_weight_str) = header_map.get("ff_weight") {
        let values = parse_comma_separated_values(ff_weight_str);
        if values.len() >= AXIS_COUNT {
            for (i, &value) in values.iter().enumerate().take(AXIS_COUNT) {
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
                ff_values[YAW_AXIS] = Some(df_yaw_val); // Yaw DF value
            }
        }
    }

    ff_values
}

/// Helper function to parse axis PID values from header map
fn parse_axis_pids(
    header_map: &std::collections::HashMap<String, String>,
) -> (AxisPid, AxisPid, AxisPid) {
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
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    // Detect firmware type using the same map
    pid_data.firmware_type = detect_firmware_type(&header_map);

    // Parse axis PID values
    let (roll, pitch, yaw) = parse_axis_pids(&header_map);
    pid_data.roll = roll;
    pid_data.pitch = pitch;
    pid_data.yaw = yaw;

    // Handle FF values based on flight controller type
    let ff_values = parse_ff_values(
        &header_map,
        &(&pid_data.roll, &pid_data.pitch, &pid_data.yaw),
    );
    pid_data.roll.ff = ff_values[ROLL_AXIS];
    pid_data.pitch.ff = ff_values[PITCH_AXIS];
    pid_data.yaw.ff = ff_values[YAW_AXIS];

    // Look for separate D-Min fields in header metadata (Betaflight separate headers)
    // Only override PID string D-Min values if separate fields exist
    let d_min_values = parse_d_values(
        &header_map,
        "d_min",
        [
            "rolldmin",
            "roll_d_min",
            "pitchdmin",
            "pitch_d_min",
            "yawdmin",
            "yaw_d_min",
        ],
    );
    // Apply D-Min values if available
    if let Some(value) = d_min_values[ROLL_AXIS] {
        pid_data.roll.d_min = Some(value);
    }
    if let Some(value) = d_min_values[PITCH_AXIS] {
        pid_data.pitch.d_min = Some(value);
    }
    if let Some(value) = d_min_values[YAW_AXIS] {
        pid_data.yaw.d_min = Some(value);
    }

    // Look for separate D-Max fields in header metadata (newer Betaflight separate headers)
    // Only override PID string D-Max values if separate fields exist
    let d_max_values = parse_d_values(
        &header_map,
        "d_max",
        [
            "rolldmax",
            "roll_d_max",
            "pitchdmax",
            "pitch_d_max",
            "yawdmax",
            "yaw_d_max",
        ],
    );
    // Apply D-Max values if available
    if let Some(value) = d_max_values[ROLL_AXIS] {
        pid_data.roll.d_max = Some(value);
    }
    if let Some(value) = d_max_values[PITCH_AXIS] {
        pid_data.pitch.d_max = Some(value);
    }
    if let Some(value) = d_max_values[YAW_AXIS] {
        pid_data.yaw.d_max = Some(value);
    }

    // Parse D-Max control parameters (BF 4.5+)
    // These determine if the D-Min/D-Max dynamic D system is actively enabled
    if let Some(gain_str) = header_map.get("d_max_gain") {
        pid_data.d_max_gain = gain_str.parse::<u32>().ok();
    }
    if let Some(advance_str) = header_map.get("d_max_advance") {
        pid_data.d_max_advance = advance_str.parse::<u32>().ok();
    }
    // Simplified tuning D-Max gain (BF 4.6+)
    if let Some(simplified_gain_str) = header_map.get("simplified_dmax_gain") {
        pid_data.simplified_d_max_gain = simplified_gain_str.parse::<u32>().ok();
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
            // Betaflight 4.6+ style: P,I,D(base=D-Min),D-Max,FF
            // In BF 4.6+, base D IS the minimum (floor), not the maximum
            axis_pid.d_min = Some(values[2]); // Base D is D-Min in BF 4.6+
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
        // Test Betaflight 4.6+ format: P,I,D(base=D-Min),D-Max,FF
        // In BF 4.6+, base D is the minimum value, not the maximum
        let metadata = vec![
            ("rollPID".to_string(), "57,66,58,58,206".to_string()),
            ("pitchPID".to_string(), "59,69,72,72,215".to_string()),
            ("yawPID".to_string(), "57,66,0,0,206".to_string()),
        ];

        let pid_data = parse_pid_metadata(&metadata);

        assert_eq!(pid_data.roll.p, Some(57));
        assert_eq!(pid_data.roll.i, Some(66));
        assert_eq!(pid_data.roll.d, Some(58));
        assert_eq!(pid_data.roll.d_min, Some(58)); // Base D is D-Min in BF 4.6+
        assert_eq!(pid_data.roll.d_max, Some(58));
        assert_eq!(pid_data.roll.ff, Some(206));

        assert_eq!(pid_data.pitch.p, Some(59));
        assert_eq!(pid_data.pitch.i, Some(69));
        assert_eq!(pid_data.pitch.d, Some(72));
        assert_eq!(pid_data.pitch.d_min, Some(72)); // Base D is D-Min in BF 4.6+
        assert_eq!(pid_data.pitch.d_max, Some(72));
        assert_eq!(pid_data.pitch.ff, Some(215));

        assert_eq!(pid_data.yaw.p, Some(57));
        assert_eq!(pid_data.yaw.i, Some(66));
        assert_eq!(pid_data.yaw.d, Some(0));
        assert_eq!(pid_data.yaw.d_min, Some(0)); // Base D is D-Min in BF 4.6+
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

        // VERIFY D:XX FORMATTING (D-Max disabled since no gain/advance values)
        let roll_formatted = pid_data
            .roll
            .format_for_title(&FirmwareType::Betaflight, false);
        assert_eq!(roll_formatted, " - P:57 I:66 D:58 FF:206");

        let pitch_formatted = pid_data
            .pitch
            .format_for_title(&FirmwareType::Betaflight, false);
        assert_eq!(pitch_formatted, " - P:59 I:69 D:72 FF:215");
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
        assert_eq!(
            axis_pid.format_for_title(&FirmwareType::Betaflight, false),
            " - P:31 I:56 D:21 FF:84"
        );

        // Test EmuFlight formatting
        assert_eq!(
            axis_pid.format_for_title(&FirmwareType::EmuFlight, false),
            " - P:31 I:56 D:21 DF:84"
        );

        // Test with zero FF (should be omitted)
        let axis_pid_zero_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            ff: Some(0),
            ..Default::default()
        };
        assert_eq!(
            axis_pid_zero_ff.format_for_title(&FirmwareType::Betaflight, false),
            " - P:31 I:56 D:21"
        );

        // Test with no FF
        let axis_pid_no_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            ff: None,
            ..Default::default()
        };
        assert_eq!(
            axis_pid_no_ff.format_for_title(&FirmwareType::Betaflight, false),
            " - P:31 I:56 D:21"
        );

        // Test D:XX/XX format when D and D-Max are different (with dmax_enabled=true)
        let axis_pid_diff_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_max: Some(35),
            ff: None,
            ..Default::default()
        };
        assert_eq!(
            axis_pid_diff_dmax.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:21/35"
        );

        // Test D:XX format when D and D-Max are the same
        let axis_pid_same_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_max: Some(21),
            ff: None,
            ..Default::default()
        };
        assert_eq!(
            axis_pid_same_dmax.format_for_title(&FirmwareType::Betaflight, false),
            " - P:31 I:56 D:21"
        );

        // Test D:min/max format when D-Min and D-Max are different
        let axis_pid_dmin_dmax = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(
            axis_pid_dmin_dmax.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:15/35"
        );

        // Test D:min/max format with FF (with dmax_enabled=true)
        let axis_pid_dmin_dmax_ff = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: Some(84),
        };
        assert_eq!(
            axis_pid_dmin_dmax_ff.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:15/35 FF:84"
        );
    }

    #[test]
    fn test_format_for_title_zero_handling() {
        // Test D:XX/0 should become D:XX (don't show /0) - with dmax_enabled=true
        let axis_pid_dmax_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(0), // D-Max is zero
            ff: None,
        };
        assert_eq!(
            axis_pid_dmax_zero.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:15"
        );

        // Test 0/XX should become D:XX (don't show 0/) - with dmax_enabled=true
        let axis_pid_dmin_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(0), // D-Min is zero
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(
            axis_pid_dmin_zero.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:35"
        );

        // Test D:XX/0 with only D-Max (no D-Min) should become D:XX - with dmax_enabled=true
        let axis_pid_only_dmax_zero = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: None,
            d_max: Some(0), // D-Max is zero
            ff: None,
        };
        assert_eq!(
            axis_pid_only_dmax_zero.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:21"
        );

        // Test normal case still works (both non-zero and different) - with dmax_enabled=true
        let axis_pid_normal = AxisPid {
            p: Some(31),
            i: Some(56),
            d: Some(21),
            d_min: Some(15),
            d_max: Some(35),
            ff: None,
        };
        assert_eq!(
            axis_pid_normal.format_for_title(&FirmwareType::Betaflight, true),
            " - P:31 I:56 D:15/35"
        );
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
        assert_eq!(
            pid_data
                .roll
                .format_for_title(&FirmwareType::Unknown, false),
            ""
        );
    }

    #[test]
    fn test_firmware_detection_case_insensitive() {
        // Test case-insensitive firmware detection with various whitespace and case combinations

        // Betaflight variations
        let metadata_bf1 = vec![(
            "firmware revision".to_string(),
            "BETAFLIGHT 4.6.0".to_string(),
        )];
        let pid_data_bf1 = parse_pid_metadata(&metadata_bf1);
        assert_eq!(pid_data_bf1.firmware_type, FirmwareType::Betaflight);

        let metadata_bf2 = vec![(
            "firmware revision".to_string(),
            "  betaflight 4.5.1  ".to_string(),
        )];
        let pid_data_bf2 = parse_pid_metadata(&metadata_bf2);
        assert_eq!(pid_data_bf2.firmware_type, FirmwareType::Betaflight);

        let metadata_bf3 = vec![(
            "firmware revision".to_string(),
            "BetaFlight Custom Build".to_string(),
        )];
        let pid_data_bf3 = parse_pid_metadata(&metadata_bf3);
        assert_eq!(pid_data_bf3.firmware_type, FirmwareType::Betaflight);

        // EmuFlight variations
        let metadata_ef1 = vec![(
            "firmware revision".to_string(),
            "EMUFLIGHT 0.4.2".to_string(),
        )];
        let pid_data_ef1 = parse_pid_metadata(&metadata_ef1);
        assert_eq!(pid_data_ef1.firmware_type, FirmwareType::EmuFlight);

        let metadata_ef2 = vec![(
            "firmware revision".to_string(),
            "  EmuFlight Beta  ".to_string(),
        )];
        let pid_data_ef2 = parse_pid_metadata(&metadata_ef2);
        assert_eq!(pid_data_ef2.firmware_type, FirmwareType::EmuFlight);

        // INAV variations
        let metadata_inav1 = vec![("firmware revision".to_string(), "INAV 8.0.0".to_string())];
        let pid_data_inav1 = parse_pid_metadata(&metadata_inav1);
        assert_eq!(pid_data_inav1.firmware_type, FirmwareType::Inav);

        let metadata_inav2 = vec![(
            "firmware revision".to_string(),
            "  iNav Latest  ".to_string(),
        )];
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

    #[test]
    fn test_whitespace_trimming() {
        // Test that whitespace in keys and values is properly trimmed
        let metadata = vec![
            ("  rollPID  ".to_string(), "  31,56,21  ".to_string()),
            (
                " firmware revision ".to_string(),
                " Betaflight 4.6.0 ".to_string(),
            ),
            ("ff_weight".to_string(), " 84,87,84 ".to_string()),
        ];

        let pid_data = parse_pid_metadata(&metadata);

        // Should correctly parse despite whitespace
        assert_eq!(pid_data.firmware_type, FirmwareType::Betaflight);
        assert_eq!(pid_data.roll.p, Some(31));
        assert_eq!(pid_data.roll.i, Some(56));
        assert_eq!(pid_data.roll.d, Some(21));
        assert_eq!(pid_data.roll.ff, Some(84));
    }

    #[test]
    fn test_pd_ratio_calculation() {
        // Test normal P:D ratio calculation
        let axis_pid = AxisPid {
            p: Some(54),
            i: Some(60),
            d: Some(32),
            d_min: None,
            d_max: None,
            ff: None,
        };
        let ratio = axis_pid.calculate_pd_ratio().unwrap();
        assert!((ratio - 1.6875).abs() < 0.001); // 54/32 = 1.6875

        // Test with D-Min/D-Max - should still use base D
        let axis_pid_dminmax = AxisPid {
            p: Some(60),
            i: Some(65),
            d: Some(40),
            d_min: Some(36),
            d_max: Some(45),
            ff: None,
        };
        let ratio_dminmax = axis_pid_dminmax.calculate_pd_ratio().unwrap();
        assert!((ratio_dminmax - 1.5).abs() < 0.001); // 60/40 = 1.5 (uses base d)

        // Test edge case: P is 0 (should still calculate, though unusual)
        let axis_pid_p_zero = AxisPid {
            p: Some(0),
            i: Some(50),
            d: Some(30),
            d_min: None,
            d_max: None,
            ff: None,
        };
        let ratio_p_zero = axis_pid_p_zero.calculate_pd_ratio().unwrap();
        assert_eq!(ratio_p_zero, 0.0);
    }

    #[test]
    fn test_pd_ratio_none_cases() {
        // Test when P is missing
        let axis_pid_no_p = AxisPid {
            p: None,
            i: Some(60),
            d: Some(32),
            d_min: None,
            d_max: None,
            ff: None,
        };
        assert_eq!(axis_pid_no_p.calculate_pd_ratio(), None);

        // Test when D is missing
        let axis_pid_no_d = AxisPid {
            p: Some(54),
            i: Some(60),
            d: None,
            d_min: None,
            d_max: None,
            ff: None,
        };
        assert_eq!(axis_pid_no_d.calculate_pd_ratio(), None);

        // Test when D is 0
        let axis_pid_d_zero = AxisPid {
            p: Some(54),
            i: Some(60),
            d: Some(0),
            d_min: None,
            d_max: None,
            ff: None,
        };
        assert_eq!(axis_pid_d_zero.calculate_pd_ratio(), None);

        // Test when D-Min is 0 but D is valid (should use D)
        let axis_pid_dmin_zero = AxisPid {
            p: Some(54),
            i: Some(60),
            d: Some(32),
            d_min: Some(0),
            d_max: None,
            ff: None,
        };
        let ratio = axis_pid_dmin_zero.calculate_pd_ratio().unwrap();
        assert!((ratio - 1.6875).abs() < 0.001); // Falls back to d=32
    }

    #[test]
    fn test_goal_d_calculation() {
        // Test calculation for target ratio of 1.4
        let axis_pid = AxisPid {
            p: Some(54),
            i: Some(60),
            d: Some(32),
            d_min: None,
            d_max: None,
            ff: None,
        };
        let goal_d = axis_pid.calculate_goal_d_for_ratio(1.4).unwrap();
        assert_eq!(goal_d, 39); // 54 / 1.4 = 38.57, rounds to 39

        // Test with different P value
        let axis_pid2 = AxisPid {
            p: Some(60),
            i: Some(65),
            d: Some(36),
            d_min: None,
            d_max: None,
            ff: None,
        };
        let goal_d2 = axis_pid2.calculate_goal_d_for_ratio(1.4).unwrap();
        assert_eq!(goal_d2, 43); // 60 / 1.4 = 42.86, rounds to 43

        // Test when P is None
        let axis_pid_no_p = AxisPid {
            p: None,
            i: Some(60),
            d: Some(32),
            d_min: None,
            d_max: None,
            ff: None,
        };
        assert_eq!(axis_pid_no_p.calculate_goal_d_for_ratio(1.4), None);

        // Test with invalid target ratio
        assert_eq!(axis_pid.calculate_goal_d_for_ratio(0.0), None);
        assert_eq!(axis_pid.calculate_goal_d_for_ratio(-1.0), None);
    }

    #[test]
    fn test_format_with_pd_ratio() {
        // Test formatting with P:D ratio included (separated by hyphen)
        let axis_pid = AxisPid {
            p: Some(54),
            i: Some(60),
            d: Some(32),
            d_min: None,
            d_max: None,
            ff: Some(100),
        };

        // Without ratio
        let formatted_no_ratio = axis_pid.format_for_title(&FirmwareType::Betaflight, false);
        assert_eq!(formatted_no_ratio, " - P:54 I:60 D:32 FF:100");

        // With ratio (note: hyphen separator before P:D ratio)
        let formatted_with_ratio =
            axis_pid.format_for_title_with_ratio(&FirmwareType::Betaflight, false, true);
        assert_eq!(formatted_with_ratio, " - P:54 I:60 D:32 FF:100 - P:D=1.69");

        // Test with D-Min/D-Max and ratio (hyphen separator)
        let axis_pid_dmin = AxisPid {
            p: Some(60),
            i: Some(65),
            d: Some(40),
            d_min: Some(36),
            d_max: Some(45),
            ff: None,
        };
        let formatted_dmin_ratio =
            axis_pid_dmin.format_for_title_with_ratio(&FirmwareType::Betaflight, true, true);
        assert_eq!(formatted_dmin_ratio, " - P:60 I:65 D:36/45 - P:D=1.50");

        // Test when ratio can't be calculated (no hyphen separator needed)
        let axis_pid_no_d = AxisPid {
            p: Some(54),
            i: Some(60),
            d: None,
            d_min: None,
            d_max: None,
            ff: None,
        };
        let formatted_no_d =
            axis_pid_no_d.format_for_title_with_ratio(&FirmwareType::Betaflight, false, true);
        assert_eq!(formatted_no_d, " - P:54 I:60"); // No P:D ratio shown
    }

    #[test]
    fn test_format_d_term_edge_cases() {
        // Only D-Min present, base D differs
        let axis_only_dmin = AxisPid {
            d: Some(50),
            d_min: Some(30),
            d_max: None,
            ..Default::default()
        };
        assert!(axis_only_dmin
            .format_for_title(&FirmwareType::Betaflight, true)
            .contains("D:30/50"));

        // Only D-Max present, no base D
        let axis_only_dmax_no_base = AxisPid {
            d: None,
            d_min: None,
            d_max: Some(80),
            ..Default::default()
        };
        assert!(axis_only_dmax_no_base
            .format_for_title(&FirmwareType::Betaflight, true)
            .contains("D:80"));
    }

    #[test]
    fn test_is_dmax_enabled() {
        // Test D-Max disabled when all parameters are zero
        let pid_data_disabled = PidMetadata {
            d_max_gain: Some(0),
            d_max_advance: Some(0),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(!pid_data_disabled.is_dmax_enabled());

        // Test D-Max disabled when all parameters are None
        let pid_data_none = PidMetadata {
            d_max_gain: None,
            d_max_advance: None,
            simplified_d_max_gain: None,
            ..Default::default()
        };
        assert!(!pid_data_none.is_dmax_enabled());

        // Test D-Max enabled when d_max_gain > 0 AND has dynamic range
        let pid_data_gain = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: Some(100),
            d_max_advance: Some(0),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(pid_data_gain.is_dmax_enabled());

        // Test D-Max enabled when d_max_advance > 0 AND has dynamic range
        let pid_data_advance = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: Some(0),
            d_max_advance: Some(50),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(pid_data_advance.is_dmax_enabled());

        // Test D-Max enabled when simplified_d_max_gain > 0 AND has dynamic range
        let pid_data_simplified = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: Some(0),
            d_max_advance: Some(0),
            simplified_d_max_gain: Some(75),
            ..Default::default()
        };
        assert!(pid_data_simplified.is_dmax_enabled());

        // Test D-Max disabled when d_max_gain > 0 BUT no dynamic range (D = D-Min)
        let pid_data_no_range = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(50), // D = D-Min, NO dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(50), // D = D-Min, NO dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: Some(100),
            d_max_advance: Some(50),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(!pid_data_no_range.is_dmax_enabled());

        // Test D-Max enabled with BF 4.6+ (d_max field exists, D-Min ≠ D-Max)
        let pid_data_bf46 = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(30),
                d_min: Some(30),
                d_max: Some(50), // D-Min ≠ D-Max, has dynamic range
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(30),
                d_min: Some(30),
                d_max: Some(50), // D-Min ≠ D-Max, has dynamic range
                ..Default::default()
            },
            d_max_gain: Some(100),
            d_max_advance: Some(50),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(pid_data_bf46.is_dmax_enabled());

        // Test D-Max disabled with BF 4.6+ when D-Min = D-Max (no dynamic range)
        let pid_data_bf46_no_range = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(50),
                d_max: Some(50), // D-Min = D-Max, NO dynamic range
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(50),
                d_max: Some(50), // D-Min = D-Max, NO dynamic range
                ..Default::default()
            },
            d_max_gain: Some(100),
            d_max_advance: Some(50),
            simplified_d_max_gain: Some(0),
            ..Default::default()
        };
        assert!(!pid_data_bf46_no_range.is_dmax_enabled());

        // Test D-Max enabled when multiple parameters > 0 AND has dynamic range
        let pid_data_multiple = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: Some(100),
            d_max_advance: Some(50),
            simplified_d_max_gain: Some(75),
            ..Default::default()
        };
        assert!(pid_data_multiple.is_dmax_enabled());

        // Test D-Max enabled when only one parameter > 0 and others are None AND has dynamic range
        let pid_data_mixed = PidMetadata {
            roll: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            pitch: AxisPid {
                p: Some(40),
                d: Some(50),
                d_min: Some(30), // D ≠ D-Min, has dynamic range
                d_max: None,
                ..Default::default()
            },
            d_max_gain: None,
            d_max_advance: Some(50),
            simplified_d_max_gain: None,
            ..Default::default()
        };
        assert!(pid_data_mixed.is_dmax_enabled());
    }
}
