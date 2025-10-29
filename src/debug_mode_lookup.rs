// src/debug_mode_lookup.rs
//
// Debug mode lookup tables for various firmware types and versions.
// Used to decode the debug_mode integer from header metadata into human-readable names.

use std::collections::HashMap;
use std::fmt;

/// Representation of firmware version type returned from parsing the header
/// - Semver(major, minor) for traditional numeric Betaflight/EmuFlight/INAV
/// - Datever(year, month, patch) for Betaflight 2025+ date-versioning
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FirmwareVersion {
    Semver(u32, u32),
    Datever(u32, u32, u32),
    Unknown,
}

impl fmt::Display for FirmwareVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FirmwareVersion::Semver(major, minor) => write!(f, "{}.{}", major, minor),
            FirmwareVersion::Datever(year, month, patch) => {
                write!(f, "{}.{}.{}", year, month, patch)
            }
            FirmwareVersion::Unknown => write!(f, "Unknown"),
        }
    }
}

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
/// Returns (firmware_type, version_string, FirmwareVersion)
fn parse_firmware_revision(firmware_revision: &str) -> (&str, &str, FirmwareVersion) {
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
            return ("INAV", version_str, FirmwareVersion::Semver(major, minor));
        }
        return ("INAV", "", FirmwareVersion::Unknown);
    }

    // EmuFlight format from header metadata: "EmuFlight VERSION (HASH) TARGET"
    // Example: "EmuFlight 0.4.3 (784cd2b6b) HELIOSPRING"
    // Check EmuFlight first before others (case-insensitive)
    if firmware_revision.strip_prefix("EmuFlight ").is_some()
        || firmware_revision.strip_prefix("Emuflight ").is_some()
    {
        // Extract version after the prefix (case-insensitive)
        let after_emuflight = if let Some(after) = firmware_revision.strip_prefix("EmuFlight ") {
            after
        } else {
            firmware_revision.strip_prefix("Emuflight ").unwrap()
        };

        // Extract version directly after the prefix
        if let Some(space_pos) = after_emuflight.find(char::is_whitespace) {
            let version_str = &after_emuflight[..space_pos];
            let version_parts: Vec<&str> = version_str.split('.').collect();
            let major = version_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let minor = version_parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return (
                "EmuFlight",
                version_str,
                FirmwareVersion::Semver(major, minor),
            );
        }
        return ("EmuFlight", "", FirmwareVersion::Unknown);
    }

    // Betaflight format from header metadata: "Betaflight VERSION (HASH) TARGET"
    // Examples:
    //   Old: "Betaflight 4.5.2 (024f8e13d) STM32F7X2"
    //   New: "Betaflight 2025.12.0-beta (aafd969ec) STM32F7X2"
    if let Some(after_betaflight) = firmware_revision.strip_prefix("Betaflight ") {
        // Extract version directly after "Betaflight "
        if let Some(space_pos) = after_betaflight.find(char::is_whitespace) {
            let version_str = &after_betaflight[..space_pos];

            // Check for new date-based versioning (YYYY.mm.x[-suffix]) → treat YYYY >= 2025 as 4.6+
            if let Some(year_part) = version_str.split('.').next() {
                if let Ok(year) = year_part.parse::<u32>() {
                    if year >= 2025 {
                        // Parse datever components as best-effort: YYYY.MM.PP or YYYY.MM.DD
                        let parts: Vec<&str> = version_str.split('.').collect();
                        let month = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                        let patch = parts
                            .get(2)
                            .and_then(|s| {
                                // patch may include suffix like "0-beta"; take numeric prefix
                                s.split('-').next().and_then(|p| p.parse().ok())
                            })
                            .unwrap_or(0);
                        return (
                            "Betaflight",
                            version_str,
                            FirmwareVersion::Datever(year, month, patch),
                        );
                    }
                }
            }

            // Old numeric versioning (4.x.x)
            let version_parts: Vec<&str> = version_str.split('.').collect();
            let major = version_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let minor = version_parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return (
                "Betaflight",
                version_str,
                FirmwareVersion::Semver(major, minor),
            );
        }
        return ("Betaflight", "", FirmwareVersion::Unknown);
    }

    ("Unknown", "", FirmwareVersion::Unknown)
}

/// Lookup debug mode name from integer value
/// Returns the debug mode name string, or None if not found
pub fn lookup_debug_mode(firmware_revision: &str, debug_mode_value: u32) -> Option<&'static str> {
    let (fw_type, _version_str, version) = parse_firmware_revision(firmware_revision);

    let mode_map = match fw_type {
        "EmuFlight" => {
            // Version-aware selection for EmuFlight
            match version {
                FirmwareVersion::Semver(0, minor) if minor <= 3 => emuflight_035_debug_modes(),
                FirmwareVersion::Semver(_, _) => emuflight_04x_debug_modes(),
                _ => emuflight_04x_debug_modes(),
            }
        }
        "Betaflight" => {
            match version {
                // Datever (2025+) → use Betaflight 4.6.x table as the best mapping
                FirmwareVersion::Datever(year, _month, _patch) if year >= 2025 => {
                    betaflight_46x_debug_modes()
                }
                FirmwareVersion::Semver(4, minor) if minor >= 6 => betaflight_46x_debug_modes(),
                FirmwareVersion::Semver(4, minor) if minor >= 5 => betaflight_45x_debug_modes(),
                FirmwareVersion::Semver(4, minor) if minor >= 4 => betaflight_44x_debug_modes(),
                // Fallback to 4.4.x
                _ => betaflight_44x_debug_modes(),
            }
        }
        "INAV" => match version {
            FirmwareVersion::Semver(major, _) if major >= 8 => inav_8x_debug_modes(),
            FirmwareVersion::Semver(major, _) if major >= 7 => inav_7x_debug_modes(),
            _ => return None,
        },
        _ => return None, // Unknown firmware
    };

    mode_map.get(&debug_mode_value).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emuflight_parse() {
        let fw = "EmuFlight 0.4.3 (784cd2b6b) HELIOSPRING";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "EmuFlight");
        assert_eq!(version_str, "0.4.3");
        match version {
            FirmwareVersion::Semver(major, minor) => {
                assert_eq!(major, 0);
                assert_eq!(minor, 4);
            }
            other => panic!("unexpected version: {:?}", other),
        }
    }

    #[test]
    fn test_betaflight_old_parse() {
        let fw = "Betaflight 4.5.2 (024f8e13d) STM32F7X2";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Betaflight");
        assert_eq!(version_str, "4.5.2");
        match version {
            FirmwareVersion::Semver(major, minor) => {
                assert_eq!(major, 4);
                assert_eq!(minor, 5);
            }
            other => panic!("unexpected version: {:?}", other),
        }
    }

    #[test]
    fn test_betaflight_new_parse() {
        let fw = "Betaflight 2025.12.0-beta (aafd969ec) STM32F7X2";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Betaflight");
        assert_eq!(version_str, "2025.12.0-beta");
        match version {
            FirmwareVersion::Datever(year, month, patch) => {
                assert_eq!(year, 2025);
                assert_eq!(month, 12);
                // patch may be 0 for the "0-beta" case where numeric prefix is 0
                assert_eq!(patch, 0);
            }
            other => panic!("unexpected version: {:?}", other),
        }
    }

    #[test]
    fn test_emuflight_gyro_scaled() {
        let fw = "EmuFlight 0.4.3 (784cd2b6b) HELIOSPRING";
        let mode = lookup_debug_mode(fw, 6);
        assert_eq!(mode, Some("GYRO_SCALED"));
    }

    #[test]
    fn test_emuflight_035_smart_smoothing() {
        let fw = "EmuFlight 0.3.5 (784cd2b6b) HELIOSPRING";
        let mode = lookup_debug_mode(fw, 45);
        assert_eq!(mode, Some("SMART_SMOOTHING"));
    }

    #[test]
    fn test_emuflight_035_no_mode_46() {
        let fw = "EmuFlight 0.3.5 (784cd2b6b) HELIOSPRING";
        let mode = lookup_debug_mode(fw, 46);
        assert_eq!(mode, None);
    }

    #[test]
    fn test_emuflight_04x_angle() {
        let fw = "EmuFlight 0.4.3 (784cd2b6b) HELIOSPRING";
        let mode = lookup_debug_mode(fw, 45);
        assert_eq!(mode, Some("ANGLE"));
    }

    #[test]
    fn test_emuflight_04x_horizon() {
        let fw = "EmuFlight 0.4.3 (784cd2b6b) HELIOSPRING";
        let mode = lookup_debug_mode(fw, 46);
        assert_eq!(mode, Some("HORIZON"));
    }

    #[test]
    fn test_betaflight_multi_gyro_diff() {
        let fw = "Betaflight 2025.12.0-beta (aafd969ec) STM32F7X2";
        let mode = lookup_debug_mode(fw, 21);
        assert_eq!(mode, Some("MULTI_GYRO_DIFF"));
    }

    #[test]
    fn test_inav_7x_parse() {
        let fw = "INAV 7.1.2 (4e1e59eb) FOXEERF722V4";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "INAV");
        assert_eq!(version_str, "7.1.2");
        match version {
            FirmwareVersion::Semver(major, minor) => {
                assert_eq!(major, 7);
                assert_eq!(minor, 1);
            }
            other => panic!("unexpected version: {:?}", other),
        }
    }

    #[test]
    fn test_inav_8x_parse() {
        let fw = "INAV 8.0.0 (ec2106af) FLYWOOF745";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "INAV");
        assert_eq!(version_str, "8.0.0");
        match version {
            FirmwareVersion::Semver(major, minor) => {
                assert_eq!(major, 8);
                assert_eq!(minor, 0);
            }
            other => panic!("unexpected version: {:?}", other),
        }
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

    #[test]
    fn test_unknown_firmware() {
        let fw = "SomethingElse 1.0.0 (a1b2c3d4e) TARGET";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Unknown");
        assert_eq!(version_str, "");
        assert_eq!(version, FirmwareVersion::Unknown);
    }

    #[test]
    fn test_empty_firmware_string() {
        let result = lookup_debug_mode("", 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_unknown_debug_mode_value() {
        let fw = "Betaflight 4.5.0 (f7e8d9c6b) TARGET";
        let result = lookup_debug_mode(fw, 9999);
        assert_eq!(result, None);
    }

    #[test]
    fn test_betaflight_future_calver() {
        // Betaflight uses YYYY.MM.PP format (patch level, not day)
        let fw = "Betaflight 2027.06.15 (a1b2c3d4e) TARGET";
        let (fw_type, version_str, version) = parse_firmware_revision(fw);
        assert_eq!(fw_type, "Betaflight");
        assert_eq!(version_str, "2027.06.15");
        match version {
            FirmwareVersion::Datever(year, month, patch) => {
                assert_eq!(year, 2027);
                assert_eq!(month, 6);
                assert_eq!(patch, 15);
            }
            other => panic!("unexpected version: {:?}", other),
        }
    }

    #[test]
    fn test_firmware_version_display() {
        assert_eq!(format!("{}", FirmwareVersion::Semver(4, 5)), "4.5");
        assert_eq!(
            format!("{}", FirmwareVersion::Datever(2025, 12, 15)),
            "2025.12.15"
        );
        assert_eq!(format!("{}", FirmwareVersion::Unknown), "Unknown");
    }
}
