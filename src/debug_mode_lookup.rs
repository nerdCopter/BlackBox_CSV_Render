// src/debug_mode_lookup.rs
//
// Debug mode lookup tables for various firmware types and versions.
// Used to decode the debug_mode integer from header metadata into human-readable names.

use std::collections::HashMap;

/// Common debug modes for EmuFlight 0-44 (shared across all versions)
fn emuflight_common_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = HashMap::new();
    map.insert(0, "NONE");
    map.insert(1, "CYCLETIME");
    map.insert(2, "BATTERY");
    map.insert(3, "GYRO_FILTERED");
    map.insert(4, "ACCELEROMETER");
    map.insert(5, "PIDLOOP");
    map.insert(6, "GYRO_SCALED");
    map.insert(7, "RC_INTERPOLATION");
    map.insert(8, "ANGLERATE");
    map.insert(9, "ESC_SENSOR");
    map.insert(10, "SCHEDULER");
    map.insert(11, "STACK");
    map.insert(12, "ESC_SENSOR_RPM");
    map.insert(13, "ESC_SENSOR_TMP");
    map.insert(14, "ALTITUDE");
    map.insert(15, "FFT");
    map.insert(16, "FFT_TIME");
    map.insert(17, "FFT_FREQ");
    map.insert(18, "RX_FRSKY_SPI");
    map.insert(19, "RX_SFHSS_SPI");
    map.insert(20, "GYRO_RAW");
    map.insert(21, "DUAL_GYRO");
    map.insert(22, "DUAL_GYRO_RAW");
    map.insert(23, "DUAL_GYRO_COMBINE");
    map.insert(24, "DUAL_GYRO_DIFF");
    map.insert(25, "MAX7456_SIGNAL");
    map.insert(26, "MAX7456_SPICLOCK");
    map.insert(27, "SBUS");
    map.insert(28, "FPORT");
    map.insert(29, "RANGEFINDER");
    map.insert(30, "RANGEFINDER_QUALITY");
    map.insert(31, "LIDAR_TF");
    map.insert(32, "CORE_TEMP");
    map.insert(33, "RUNAWAY_TAKEOFF");
    map.insert(34, "SDIO");
    map.insert(35, "CURRENT_SENSOR");
    map.insert(36, "USB");
    map.insert(37, "SMARTAUDIO");
    map.insert(38, "RTH");
    map.insert(39, "ITERM_RELAX");
    map.insert(40, "RC_SMOOTHING");
    map.insert(41, "RX_SIGNAL_LOSS");
    map.insert(42, "RC_SMOOTHING_RATE");
    map.insert(43, "IMU");
    map.insert(44, "KALMAN");
    map
}

/// Debug mode lookup for EmuFlight 0.3.5
/// Has SMART_SMOOTHING at mode 45, no mode 46
fn emuflight_035_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = emuflight_common_debug_modes();
    map.insert(45, "SMART_SMOOTHING");
    map
}

/// Debug mode lookup for EmuFlight 0.4.x (0.4.2, 0.4.3, etc.)
/// Has ANGLE at mode 45 and HORIZON at mode 46
fn emuflight_04x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = emuflight_common_debug_modes();
    map.insert(45, "ANGLE");
    map.insert(46, "HORIZON");
    map
}

/// Debug mode lookup for Betaflight 4.4.x
fn betaflight_44x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = HashMap::new();
    map.insert(0, "NONE");
    map.insert(1, "CYCLETIME");
    map.insert(2, "BATTERY");
    map.insert(3, "GYRO_FILTERED");
    map.insert(4, "ACCELEROMETER");
    map.insert(5, "PIDLOOP");
    map.insert(6, "GYRO_SCALED");
    map.insert(7, "RC_INTERPOLATION");
    map.insert(8, "ANGLERATE");
    map.insert(9, "ESC_SENSOR");
    map.insert(10, "SCHEDULER");
    map.insert(11, "STACK");
    map.insert(12, "ESC_SENSOR_RPM");
    map.insert(13, "ESC_SENSOR_TMP");
    map.insert(14, "ALTITUDE");
    map.insert(15, "FFT");
    map.insert(16, "FFT_TIME");
    map.insert(17, "FFT_FREQ");
    map.insert(18, "RX_FRSKY_SPI");
    map.insert(19, "RX_SFHSS_SPI");
    map.insert(20, "GYRO_RAW");
    map.insert(21, "DUAL_GYRO_RAW");
    map.insert(22, "DUAL_GYRO_DIFF");
    map.insert(23, "MAX7456_SIGNAL");
    map.insert(24, "MAX7456_SPICLOCK");
    map.insert(25, "SBUS");
    map.insert(26, "FPORT");
    map.insert(27, "RANGEFINDER");
    map.insert(28, "RANGEFINDER_QUALITY");
    map.insert(29, "LIDAR_TF");
    map.insert(30, "ADC_INTERNAL");
    map.insert(31, "RUNAWAY_TAKEOFF");
    map.insert(32, "SDIO");
    map.insert(33, "CURRENT_SENSOR");
    map.insert(34, "USB");
    map.insert(35, "SMARTAUDIO");
    map.insert(36, "RTH");
    map.insert(37, "ITERM_RELAX");
    map.insert(38, "ACRO_TRAINER");
    map.insert(39, "RC_SMOOTHING");
    map.insert(40, "RX_SIGNAL_LOSS");
    map.insert(41, "RC_SMOOTHING_RATE");
    map.insert(42, "ANTI_GRAVITY");
    map.insert(43, "DYN_LPF");
    map.insert(44, "RX_SPEKTRUM_SPI");
    map.insert(45, "DSHOT_RPM_TELEMETRY");
    map.insert(46, "RPM_FILTER");
    map.insert(47, "D_MIN");
    map.insert(48, "AC_CORRECTION");
    map.insert(49, "AC_ERROR");
    map.insert(50, "DUAL_GYRO_SCALED");
    map.insert(51, "DSHOT_RPM_ERRORS");
    map.insert(52, "CRSF_LINK_STATISTICS_UPLINK");
    map.insert(53, "CRSF_LINK_STATISTICS_PWR");
    map.insert(54, "CRSF_LINK_STATISTICS_DOWN");
    map.insert(55, "BARO");
    map.insert(56, "GPS_RESCUE_THROTTLE_PID");
    map.insert(57, "DYN_IDLE");
    map.insert(58, "FEEDFORWARD_LIMIT");
    map.insert(59, "FEEDFORWARD");
    map.insert(60, "BLACKBOX_OUTPUT");
    map.insert(61, "GYRO_SAMPLE");
    map.insert(62, "RX_TIMING");
    map.insert(63, "D_LPF");
    map.insert(64, "VTX_TRAMP");
    map.insert(65, "GHST");
    map.insert(66, "GHST_MSP");
    map.insert(67, "SCHEDULER_DETERMINISM");
    map.insert(68, "TIMING_ACCURACY");
    map.insert(69, "RX_EXPRESSLRS_SPI");
    map.insert(70, "RX_EXPRESSLRS_PHASELOCK");
    map.insert(71, "RX_STATE_TIME");
    map.insert(72, "GPS_RESCUE_VELOCITY");
    map.insert(73, "GPS_RESCUE_HEADING");
    map.insert(74, "GPS_RESCUE_TRACKING");
    map.insert(75, "ATTITUDE");
    map.insert(76, "VTX_MSP");
    map.insert(77, "GPS_DOP");
    map.insert(78, "FAILSAFE");
    map.insert(79, "DSHOT_TELEMETRY_COUNTS");
    map
}

/// Debug mode lookup for Betaflight 4.5.x
fn betaflight_45x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = betaflight_44x_debug_modes();
    // Add new modes introduced in 4.5.x
    map.insert(80, "MAG_CALIB");
    map.insert(81, "MAG_TASK_RATE");
    map.insert(82, "EZLANDING");
    map
}

/// Debug mode lookup for Betaflight 4.6.x (2025.12.0 and later)
fn betaflight_46x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = HashMap::new();
    map.insert(0, "NONE");
    map.insert(1, "CYCLETIME");
    map.insert(2, "BATTERY");
    map.insert(3, "GYRO_FILTERED");
    map.insert(4, "ACCELEROMETER");
    map.insert(5, "PIDLOOP");
    map.insert(6, "RC_INTERPOLATION");
    map.insert(7, "ANGLERATE");
    map.insert(8, "ESC_SENSOR");
    map.insert(9, "SCHEDULER");
    map.insert(10, "STACK");
    map.insert(11, "ESC_SENSOR_RPM");
    map.insert(12, "ESC_SENSOR_TMP");
    map.insert(13, "ALTITUDE");
    map.insert(14, "FFT");
    map.insert(15, "FFT_TIME");
    map.insert(16, "FFT_FREQ");
    map.insert(17, "RX_FRSKY_SPI");
    map.insert(18, "RX_SFHSS_SPI");
    map.insert(19, "GYRO_RAW");
    map.insert(20, "MULTI_GYRO_RAW");
    map.insert(21, "MULTI_GYRO_DIFF");
    map.insert(22, "MAX7456_SIGNAL");
    map.insert(23, "MAX7456_SPICLOCK");
    map.insert(24, "SBUS");
    map.insert(25, "FPORT");
    map.insert(26, "RANGEFINDER");
    map.insert(27, "RANGEFINDER_QUALITY");
    map.insert(28, "OPTICALFLOW");
    map.insert(29, "LIDAR_TF");
    map.insert(30, "ADC_INTERNAL");
    map.insert(31, "RUNAWAY_TAKEOFF");
    map.insert(32, "SDIO");
    map.insert(33, "CURRENT_SENSOR");
    map.insert(34, "USB");
    map.insert(35, "SMARTAUDIO");
    map.insert(36, "RTH");
    map.insert(37, "ITERM_RELAX");
    map.insert(38, "ACRO_TRAINER");
    map.insert(39, "RC_SMOOTHING");
    map.insert(40, "RX_SIGNAL_LOSS");
    map.insert(41, "RC_SMOOTHING_RATE");
    map.insert(42, "ANTI_GRAVITY");
    map.insert(43, "DYN_LPF");
    map.insert(44, "RX_SPEKTRUM_SPI");
    map.insert(45, "DSHOT_RPM_TELEMETRY");
    map.insert(46, "RPM_FILTER");
    map.insert(47, "D_MAX");
    map.insert(48, "AC_CORRECTION");
    map.insert(49, "AC_ERROR");
    map.insert(50, "MULTI_GYRO_SCALED");
    map.insert(51, "DSHOT_RPM_ERRORS");
    map.insert(52, "CRSF_LINK_STATISTICS_UPLINK");
    map.insert(53, "CRSF_LINK_STATISTICS_PWR");
    map.insert(54, "CRSF_LINK_STATISTICS_DOWN");
    map.insert(55, "BARO");
    map.insert(56, "AUTOPILOT_ALTITUDE");
    map.insert(57, "DYN_IDLE");
    map.insert(58, "FEEDFORWARD_LIMIT");
    map.insert(59, "FEEDFORWARD");
    map.insert(60, "BLACKBOX_OUTPUT");
    map.insert(61, "GYRO_SAMPLE");
    map.insert(62, "RX_TIMING");
    map.insert(63, "D_LPF");
    map.insert(64, "VTX_TRAMP");
    map.insert(65, "GHST");
    map.insert(66, "GHST_MSP");
    map.insert(67, "SCHEDULER_DETERMINISM");
    map.insert(68, "TIMING_ACCURACY");
    map.insert(69, "RX_EXPRESSLRS_SPI");
    map.insert(70, "RX_EXPRESSLRS_PHASELOCK");
    map.insert(71, "RX_STATE_TIME");
    map.insert(72, "GPS_RESCUE_VELOCITY");
    map.insert(73, "GPS_RESCUE_HEADING");
    map.insert(74, "GPS_RESCUE_TRACKING");
    map.insert(75, "GPS_CONNECTION");
    map.insert(76, "ATTITUDE");
    map.insert(77, "VTX_MSP");
    map.insert(78, "GPS_DOP");
    map.insert(79, "FAILSAFE");
    map.insert(80, "GYRO_CALIBRATION");
    map.insert(81, "ANGLE_MODE");
    map.insert(82, "ANGLE_TARGET");
    map.insert(83, "CURRENT_ANGLE");
    map.insert(84, "DSHOT_TELEMETRY_COUNTS");
    map.insert(85, "RPM_LIMIT");
    map.insert(86, "RC_STATS");
    map.insert(87, "MAG_CALIB");
    map.insert(88, "MAG_TASK_RATE");
    map.insert(89, "EZLANDING");
    map.insert(90, "TPA");
    map.insert(91, "S_TERM");
    map.insert(92, "SPA");
    map.insert(93, "TASK");
    map.insert(94, "GIMBAL");
    map.insert(95, "WING_SETPOINT");
    map.insert(96, "AUTOPILOT_POSITION");
    map.insert(97, "CHIRP");
    map.insert(98, "FLASH_TEST_PRBS");
    map.insert(99, "MAVLINK_TELEMETRY");
    map
}

/// Debug mode lookup for INAV 7.x
fn inav_7x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = HashMap::new();
    map.insert(0, "NONE");
    map.insert(1, "AGL");
    map.insert(2, "FLOW_RAW");
    map.insert(3, "FLOW");
    map.insert(4, "ALWAYS");
    map.insert(5, "SAG_COMP_VOLTAGE");
    map.insert(6, "VIBE");
    map.insert(7, "CRUISE");
    map.insert(8, "REM_FLIGHT_TIME");
    map.insert(9, "SMARTAUDIO");
    map.insert(10, "ACC");
    map.insert(11, "NAV_YAW");
    map.insert(12, "PCF8574");
    map.insert(13, "DYN_GYRO_LPF");
    map.insert(14, "AUTOLEVEL");
    map.insert(15, "ALTITUDE");
    map.insert(16, "AUTOTRIM");
    map.insert(17, "AUTOTUNE");
    map.insert(18, "RATE_DYNAMICS");
    map.insert(19, "LANDING");
    map.insert(20, "POS_EST");
    map
}

/// Debug mode lookup for INAV 8.x (8.0.1 and later)
fn inav_8x_debug_modes() -> HashMap<u32, &'static str> {
    let mut map = inav_7x_debug_modes();
    // INAV 8.x adds new debug modes after POS_EST
    map.insert(21, "ADAPTIVE_FILTER");
    map.insert(22, "HEADTRACKER");
    map.insert(23, "GPS");
    map.insert(24, "LULU");
    map.insert(25, "SBUS2");
    map
}

/// Detect firmware type and version from the firmware_revision header metadata
/// Returns (firmware_type, major_version, minor_version)
fn parse_firmware_revision(firmware_revision: &str) -> (&str, u32, u32) {
    // INAV format from header metadata: "INAV x.x.x (<HASH>) <TARGET>"
    // Example: "INAV 8.0.0 (ec2106af) FLYWOOF745"
    // Check INAV first before others
    if let Some(after_inav) = firmware_revision.strip_prefix("INAV ") {
        // Extract version directly after "INAV "
        if let Some(space_pos) = after_inav.find(char::is_whitespace) {
            let version_str = &after_inav[..space_pos];
            let version_parts: Vec<&str> = version_str.split('.').collect();
            let major = version_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let minor = version_parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return ("INAV", major, minor);
        }
        return ("INAV", 0, 0);
    }

    // EmuFlight format: "EmuFlight / <TARGET> (<CODE>) 0.4.3 <DATE> / <TIME> (<HASH>) MSP API: <VERSION>"
    if firmware_revision.contains("EmuFlight") || firmware_revision.contains("Emuflight") {
        // Extract version number like "0.4.3"
        if let Some(version_start) = firmware_revision.find(|c: char| c.is_ascii_digit()) {
            let version_str = &firmware_revision[version_start..];
            let version_parts: Vec<&str> = version_str
                .split_whitespace()
                .next()
                .unwrap_or("0.0.0")
                .split('.')
                .collect();
            let major = version_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let minor = version_parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return ("EmuFlight", major, minor);
        }
        return ("EmuFlight", 0, 0);
    }

    // Betaflight format can be:
    // Old: "Betaflight / <TARGET> (<CODE>) 4.5.2 <DATE> / <TIME> (<HASH>) MSP API: <VERSION>"
    // New: "Betaflight / <TARGET> (<CODE>) 2025.12.0-beta <DATE> / <TIME> (<HASH>) MSP API: <VERSION>"
    if firmware_revision.contains("Betaflight") {
        // Find the closing parenthesis after the target code, version comes after that
        if let Some(paren_pos) = firmware_revision.find(") ") {
            let after_paren = &firmware_revision[paren_pos + 2..]; // Skip ") "

            // Check for new date-based versioning (YYYY.mm.x[-suffix]) → treat YYYY >= 2025 as 4.6+
            if let Some(ver_token) = after_paren.split_whitespace().next() {
                if let Some(year_part) = ver_token.split('.').next() {
                    if let Ok(year) = year_part.parse::<u32>() {
                        if year >= 2025 {
                            return ("Betaflight", 4, 6);
                        }
                    }
                }
            }

            // Old numeric versioning (4.x.x)
            if let Some(space_pos) = after_paren.find(char::is_whitespace) {
                let version_str = &after_paren[..space_pos];
                let version_parts: Vec<&str> = version_str.split('.').collect();
                let major = version_parts
                    .first()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let minor = version_parts
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return ("Betaflight", major, minor);
            }
        }
        return ("Betaflight", 0, 0);
    }

    ("Unknown", 0, 0)
}

/// Lookup debug mode name from integer value
/// Returns the debug mode name string, or None if not found
pub fn lookup_debug_mode(firmware_revision: &str, debug_mode_value: u32) -> Option<&'static str> {
    let (fw_type, major, minor) = parse_firmware_revision(firmware_revision);

    let mode_map = match fw_type {
        "EmuFlight" => {
            // Version-aware selection for EmuFlight
            if major == 0 && minor <= 3 {
                // EmuFlight 0.3.5 and earlier use SMART_SMOOTHING at 45
                emuflight_035_debug_modes()
            } else {
                // EmuFlight 0.4.x and later use ANGLE at 45, HORIZON at 46
                emuflight_04x_debug_modes()
            }
        }
        "Betaflight" => {
            if major == 4 && minor >= 6 {
                betaflight_46x_debug_modes()
            } else if major == 4 && minor >= 5 {
                betaflight_45x_debug_modes()
            } else if major == 4 && minor >= 4 {
                betaflight_44x_debug_modes()
            } else {
                // Fallback to 4.4.x for older versions
                betaflight_44x_debug_modes()
            }
        }
        "INAV" => {
            if major >= 8 {
                inav_8x_debug_modes()
            } else if major >= 7 {
                inav_7x_debug_modes()
            } else {
                // Unknown INAV version - return None
                return None;
            }
        }
        _ => return None, // Unknown firmware
    };

    mode_map.get(&debug_mode_value).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emuflight_parse() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.4.3 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let (fw_type, major, minor) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "EmuFlight");
        assert_eq!(major, 0);
        assert_eq!(minor, 4);
    }

    #[test]
    fn test_betaflight_old_parse() {
        let fw =
            "Betaflight / STM32F7X2 (S7X2) 4.5.2 Jun 10 2025 / 09:47:40 (024f8e13d) MSP API: 1.46";
        let (fw_type, major, minor) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Betaflight");
        assert_eq!(major, 4);
        assert_eq!(minor, 5);
    }

    #[test]
    fn test_betaflight_new_parse() {
        let fw = "Betaflight / STM32F7X2 (S7X2) 2025.12.0-beta Sep 13 2025 / 15:33:21 (aafd969ec) MSP API: 1.47";
        let (fw_type, major, minor) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Betaflight");
        assert_eq!(major, 4);
        assert_eq!(minor, 6);
    }

    #[test]
    fn test_emuflight_gyro_scaled() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.4.3 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let mode = lookup_debug_mode(fw, 6);
        assert_eq!(mode, Some("GYRO_SCALED"));
    }

    #[test]
    fn test_emuflight_035_smart_smoothing() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.3.5 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let mode = lookup_debug_mode(fw, 45);
        assert_eq!(mode, Some("SMART_SMOOTHING"));
    }

    #[test]
    fn test_emuflight_035_no_mode_46() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.3.5 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let mode = lookup_debug_mode(fw, 46);
        assert_eq!(mode, None);
    }

    #[test]
    fn test_emuflight_04x_angle() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.4.3 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let mode = lookup_debug_mode(fw, 45);
        assert_eq!(mode, Some("ANGLE"));
    }

    #[test]
    fn test_emuflight_04x_horizon() {
        let fw =
            "EmuFlight / HELIOSPRING (HESP) 0.4.3 Jul 12 2024 / 17:13:23 (179c0bb86) MSP API: 1.54";
        let mode = lookup_debug_mode(fw, 46);
        assert_eq!(mode, Some("HORIZON"));
    }

    #[test]
    fn test_betaflight_multi_gyro_diff() {
        let fw = "Betaflight / STM32F7X2 (S7X2) 2025.12.0-beta Sep 13 2025 / 15:33:21 (aafd969ec) MSP API: 1.47";
        let mode = lookup_debug_mode(fw, 21);
        assert_eq!(mode, Some("MULTI_GYRO_DIFF"));
    }

    #[test]
    fn test_inav_7x_parse() {
        let fw = "INAV 7.1.2 (4e1e59eb) FOXEERF722V4";
        let (fw_type, major, minor) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "INAV");
        assert_eq!(major, 7);
        assert_eq!(minor, 1);
    }

    #[test]
    fn test_inav_8x_parse() {
        let fw = "INAV 8.0.0 (ec2106af) FLYWOOF745";
        let (fw_type, major, minor) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "INAV");
        assert_eq!(major, 8);
        assert_eq!(minor, 0);
    }

    #[test]
    fn test_inav_7x_agl() {
        let fw = "INAV 7.1.2 (4e1e59eb) FOXEERF722V4";
        let mode = lookup_debug_mode(fw, 1);
        assert_eq!(mode, Some("AGL"));
    }

    #[test]
    fn test_inav_7x_pos_est() {
        let fw = "INAV 7.1.2 (4e1e59eb) FOXEERF722V4";
        let mode = lookup_debug_mode(fw, 20);
        assert_eq!(mode, Some("POS_EST"));
    }

    #[test]
    fn test_inav_8x_sbus2() {
        let fw = "INAV 8.0.0 (ec2106af) FLYWOOF745";
        let mode = lookup_debug_mode(fw, 25);
        assert_eq!(mode, Some("SBUS2"));
    }

    #[test]
    fn test_inav_8x_adaptive_filter() {
        let fw = "INAV 8.0.0 (ec2106af) FLYWOOF745";
        let mode = lookup_debug_mode(fw, 21);
        assert_eq!(mode, Some("ADAPTIVE_FILTER"));
    }
}
