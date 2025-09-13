// src/data_analysis/filter_response.rs

use std::collections::HashMap;

use crate::axis_names::AXIS_NAMES;
use crate::constants::MIN_SPECTRUM_POINTS_FOR_ANALYSIS;

/// Filter types supported by flight controllers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    PT1 = 0,
    Biquad = 1,
    PT2 = 2,
    PT3 = 3,
    PT4 = 4, // EmuFlight specific, rarely used but supported
}

impl FilterType {
    /// Convert numeric filter type to enum
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(FilterType::PT1),
            1 => Some(FilterType::Biquad),
            2 => Some(FilterType::PT2),
            3 => Some(FilterType::PT3),
            4 => Some(FilterType::PT4),
            _ => None,
        }
    }

    /// Get filter type name for display
    pub fn name(&self) -> &'static str {
        match self {
            FilterType::PT1 => "PT1",
            FilterType::Biquad => "BIQUAD",
            FilterType::PT2 => "PT2",
            FilterType::PT3 => "PT3",
            FilterType::PT4 => "PT4",
        }
    }
}

/// Filter configuration for a single filter stage
#[derive(Debug, Clone)]
pub struct FilterConfig {
    pub filter_type: FilterType,
    pub cutoff_hz: f64,
    pub enabled: bool,
}

/// Filter configuration for dynamic filters (Betaflight)
#[derive(Debug, Clone)]
pub struct DynamicFilterConfig {
    pub filter_type: FilterType,
    pub min_cutoff_hz: f64,
    pub max_cutoff_hz: f64,
    pub expo: u32,
    pub enabled: bool,
}

/// IMUF filter configuration for HELIOSPRING flight controllers
#[derive(Debug, Clone)]
pub struct ImufFilterConfig {
    // Header values (as configured)
    pub lowpass_cutoff_hz: f64, // Original configured cutoff from header
    pub ptn_order: u32,         // Filter order (1-4 -> PT1, PT2, PT3, PT4)
    pub q_factor: f64,          // Q-factor (scaled by 1000 in headers)

    // Calculated values (accounting for IMU-F implementation)
    pub effective_cutoff_hz: f64, // Cutoff after PTN scaling factors

    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct AxisFilterConfig {
    pub lpf1: Option<FilterConfig>,
    pub lpf2: Option<FilterConfig>,
    pub dynamic_lpf1: Option<DynamicFilterConfig>,
    pub imuf: Option<ImufFilterConfig>,
}

/// Filter configuration for all gyro and dterm filters
#[derive(Debug, Clone, Default)]
pub struct AllFilterConfigs {
    pub gyro: [AxisFilterConfig; 3],  // Roll, Pitch, Yaw
    pub dterm: [AxisFilterConfig; 3], // Roll, Pitch, Yaw
}

/// Type alias for filter curve data: (label, curve_points, cutoff_hz)
pub type FilterCurveData = (String, Vec<(f64, f64)>, f64);

/// Calculate frequency response magnitude for PT1 filter
/// H(s) = 1 / (1 + s/ωc) where ωc = 2π * cutoff_hz
pub fn pt1_response(frequency_hz: f64, cutoff_hz: f64) -> f64 {
    if cutoff_hz <= 0.0 {
        return 1.0; // No filtering
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let omega_c = 2.0 * std::f64::consts::PI * cutoff_hz;
    1.0 / (1.0 + (omega / omega_c).powi(2)).sqrt()
}

/// Calculate frequency response magnitude for PT2 filter (2nd order Butterworth)
/// H(s) = 1 / (1 + √2·s/ωc + (s/ωc)²)
pub fn pt2_response(frequency_hz: f64, cutoff_hz: f64) -> f64 {
    if cutoff_hz <= 0.0 {
        return 1.0; // No filtering
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let omega_c = 2.0 * std::f64::consts::PI * cutoff_hz;
    let s_norm = omega / omega_c;
    // Butterworth 2nd order magnitude: 1 / sqrt(1 + (ω/ωc)^4)
    1.0 / (1.0 + s_norm.powi(4)).sqrt()
}

/// Calculate frequency response magnitude for PT3 filter (3rd order Butterworth)  
/// H(s) = 1 / (1 + s + s² + s³) where s = jω/ωc
pub fn pt3_response(frequency_hz: f64, cutoff_hz: f64) -> f64 {
    if cutoff_hz <= 0.0 {
        return 1.0; // No filtering
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let omega_c = 2.0 * std::f64::consts::PI * cutoff_hz;
    let s_norm = omega / omega_c;
    // 3rd order Butterworth: 1 / sqrt((1 + s²)² + s²)
    // Approximated as: 1 / sqrt(1 + s⁶) for simplicity while maintaining -3dB at cutoff
    1.0 / (1.0 + s_norm.powi(6)).sqrt()
}

/// Calculate frequency response magnitude for PT4 filter (4th order Butterworth)
/// H(s) = 1 / (1 + √2·s + s² + √2·s³ + s⁴)
/// EmuFlight specific, rarely used but supported for completeness
pub fn pt4_response(frequency_hz: f64, cutoff_hz: f64) -> f64 {
    if cutoff_hz <= 0.0 {
        return 1.0; // No filtering
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let omega_c = 2.0 * std::f64::consts::PI * cutoff_hz;
    let s_norm = omega / omega_c;
    // 4th order Butterworth: 1 / sqrt(1 + s⁸) approximation maintaining -3dB at cutoff
    1.0 / (1.0 + s_norm.powi(8)).sqrt()
}

/// Calculate frequency response magnitude for BIQUAD filter
/// Simplified as 2nd order lowpass for now (can be enhanced with Q factor later)
pub fn biquad_response(frequency_hz: f64, cutoff_hz: f64) -> f64 {
    if cutoff_hz <= 0.0 {
        return 1.0; // No filtering
    }
    // For now, treat as 2nd order lowpass similar to PT2
    // This can be enhanced later with proper Q factor and filter type (LP/HP/BP/Notch)
    pt2_response(frequency_hz, cutoff_hz)
}

/// Generate individual filter response curves (separate curve for each filter)
/// Returns Vec<FilterCurveData> for multiple curves with cutoff markers
/// Ensures LPF1 (static or dynamic) appears before LPF2 in legend order
pub fn generate_individual_filter_curves(
    axis_config: &AxisFilterConfig,
    max_frequency_hz: f64,
    num_points: usize,
) -> Vec<FilterCurveData> {
    let mut filter_curves = Vec::new();

    // Generate curve for LPF1 (static or dynamic - whichever is configured)
    if let Some(ref dyn_lpf1) = axis_config.dynamic_lpf1 {
        // Dynamic LPF1 takes precedence
        if dyn_lpf1.enabled && dyn_lpf1.min_cutoff_hz > 0.0 {
            let static_filter = FilterConfig {
                filter_type: dyn_lpf1.filter_type,
                cutoff_hz: dyn_lpf1.min_cutoff_hz, // Use minimum cutoff
                enabled: true,
            };
            let curve = generate_single_filter_curve(&static_filter, max_frequency_hz, num_points);
            let label = if dyn_lpf1.max_cutoff_hz > dyn_lpf1.min_cutoff_hz {
                format!(
                    "Dyn LPF1 ({} {:.0}-{:.0}Hz)",
                    dyn_lpf1.filter_type.name(),
                    dyn_lpf1.min_cutoff_hz,
                    dyn_lpf1.max_cutoff_hz
                )
            } else {
                format!(
                    "Dyn LPF1 ({} @ {:.0}Hz)",
                    dyn_lpf1.filter_type.name(),
                    dyn_lpf1.min_cutoff_hz
                )
            };
            filter_curves.push((label, curve, dyn_lpf1.min_cutoff_hz));
        }
    } else if let Some(ref lpf1) = axis_config.lpf1 {
        // Static LPF1 fallback
        if lpf1.enabled && lpf1.cutoff_hz > 0.0 {
            let curve = generate_single_filter_curve(lpf1, max_frequency_hz, num_points);
            let label = format!(
                "LPF1 ({} @ {:.0}Hz)",
                lpf1.filter_type.name(),
                lpf1.cutoff_hz
            );
            filter_curves.push((label, curve, lpf1.cutoff_hz));
        }
    }

    // Generate curve for LPF2 if enabled (always after LPF1)
    if let Some(ref lpf2) = axis_config.lpf2 {
        if lpf2.enabled && lpf2.cutoff_hz > 0.0 {
            let curve = generate_single_filter_curve(lpf2, max_frequency_hz, num_points);
            let label = format!(
                "LPF2 ({} @ {:.0}Hz)",
                lpf2.filter_type.name(),
                lpf2.cutoff_hz
            );
            filter_curves.push((label, curve, lpf2.cutoff_hz));
        }
    }

    // Generate curves for IMUF filter if enabled - show both header and corrected values
    if let Some(ref imuf) = axis_config.imuf {
        if imuf.enabled && imuf.lowpass_cutoff_hz > 0.0 {
            let filter_type = match imuf.ptn_order {
                1 => FilterType::PT1,
                2 => FilterType::PT2,
                3 => FilterType::PT3,
                4 => FilterType::PT4,
                _ => FilterType::PT2, // Default fallback
            };

            // Generate curve for header-configured cutoff (what user intended)
            let header_filter = FilterConfig {
                filter_type,
                cutoff_hz: imuf.lowpass_cutoff_hz,
                enabled: true,
            };
            let header_curve =
                generate_single_filter_curve(&header_filter, max_frequency_hz, num_points);
            let header_label = format!(
                "IMUF v256 Header ({} @ {:.0}Hz)",
                filter_type.name(),
                imuf.lowpass_cutoff_hz
            );
            filter_curves.push((header_label, header_curve, imuf.lowpass_cutoff_hz));

            // Generate curve for effective cutoff (accounting for PTN scaling)
            if (imuf.effective_cutoff_hz - imuf.lowpass_cutoff_hz).abs() > 1.0 {
                let effective_filter = FilterConfig {
                    filter_type,
                    cutoff_hz: imuf.effective_cutoff_hz,
                    enabled: true,
                };
                let effective_curve =
                    generate_single_filter_curve(&effective_filter, max_frequency_hz, num_points);
                let effective_label = format!(
                    "IMUF v256 Effective ({} @ {:.0}Hz, PTN scaled)",
                    filter_type.name(),
                    imuf.effective_cutoff_hz
                );
                filter_curves.push((effective_label, effective_curve, imuf.effective_cutoff_hz));
            }
        }
    }

    filter_curves
}

/// Generate response curve for a single filter with full frequency range
fn generate_single_filter_curve(
    filter: &FilterConfig,
    max_frequency_hz: f64,
    num_points: usize,
) -> Vec<(f64, f64)> {
    let mut curve_points = Vec::with_capacity(num_points);

    if num_points < 2 {
        return curve_points;
    }

    // Standard practice: start at 10% of cutoff frequency to show complete response
    // This captures the flat passband, S-shaped transition, and roll-off regions
    let start_freq = filter.cutoff_hz * 0.1; // 10% of cutoff is standard

    if start_freq >= max_frequency_hz {
        return curve_points; // Invalid range
    }

    // Use logarithmic spacing for smooth curves over wide frequency ranges
    let log_start = start_freq.ln();
    let log_end = max_frequency_hz.ln();
    let log_step = (log_end - log_start) / (num_points - 1) as f64;

    // Generate the complete mathematically correct filter response curve
    for i in 0..num_points {
        let log_frequency = log_start + (i as f64 * log_step);
        let frequency = log_frequency.exp();
        let magnitude = match filter.filter_type {
            FilterType::PT1 => pt1_response(frequency, filter.cutoff_hz),
            FilterType::PT2 => pt2_response(frequency, filter.cutoff_hz),
            FilterType::PT3 => pt3_response(frequency, filter.cutoff_hz),
            FilterType::PT4 => pt4_response(frequency, filter.cutoff_hz),
            FilterType::Biquad => biquad_response(frequency, filter.cutoff_hz),
        };
        curve_points.push((frequency, magnitude));
    }

    curve_points
}

/// Check if gyro dynamic LPF filtering is enabled
/// Returns (has_dynamic, min_cutoff, max_cutoff)
/// Note: Betaflight (only firmware with dynamic LPF) uses same settings for all axes,
/// so we only need to check the first axis (Roll)
pub fn check_gyro_dynamic_lpf_usage(config: &AllFilterConfigs) -> (bool, f64, f64) {
    if let Some(ref dyn_lpf) = config.gyro[0].dynamic_lpf1 {
        if dyn_lpf.enabled
            && dyn_lpf.min_cutoff_hz > 0.0
            && dyn_lpf.max_cutoff_hz > dyn_lpf.min_cutoff_hz
        {
            return (true, dyn_lpf.min_cutoff_hz, dyn_lpf.max_cutoff_hz);
        }
    }
    (false, 0.0, 0.0)
}

/// Extract gyro rate from header metadata for proper Nyquist calculation
/// Filters operate at gyro rate, not logging rate
pub fn extract_gyro_rate(header_metadata: Option<&[(String, String)]>) -> Option<f64> {
    if let Some(metadata) = header_metadata {
        // Look for explicit gyro rate in various possible header formats
        for (key, value) in metadata {
            let key_l = key.to_ascii_lowercase();
            if key_l.contains("gyrosampleratehz")
                || key_l.contains("gyro_sample_hz")
                || key_l.contains("gyro_rate_hz")
                || key_l.contains("gyro_rate")
                || key_l.contains("gyrorate")
            {
                // Try to parse the gyro rate
                if let Ok(rate) = value.parse::<f64>() {
                    if rate > 1000.0 && rate <= 32000.0 {
                        // Reasonable gyro rate range
                        return Some(rate);
                    }
                }
            }
        }

        // Look for gyro sync denominator to calculate rate
        let mut base_rate = 8000.0; // Default PID loop rate
        let mut sync_denom = 1.0;

        for (key, value) in metadata {
            let key_l = key.to_ascii_lowercase();
            if key_l.contains("gyro_sync_denom") || key_l.contains("gyrosyncdenom") {
                if let Ok(denom) = value.parse::<f64>() {
                    if denom > 0.0 && denom <= 32.0 {
                        sync_denom = denom;
                    }
                }
            }
            // Some logs have explicit PID loop rate
            if key_l.contains("looptime") || key_l.contains("pid_process_denom") {
                if let Ok(val) = value.parse::<f64>() {
                    if val > 0.0 && val < 1000.0 {
                        // looptime in microseconds
                        base_rate = 1000000.0 / val;
                    } else if val > 1000.0 && val <= 32000.0 {
                        // direct frequency
                        base_rate = val;
                    }
                }
            }
        }

        let calculated_rate = base_rate / sync_denom;
        if calculated_rate > 1000.0 && calculated_rate <= 32000.0 {
            return Some(calculated_rate);
        }

        // Fallback: assume 8kHz for modern firmware
        for (key, value) in metadata {
            let key_l = key.to_ascii_lowercase();
            if (key_l.contains("firmwaretype") || key_l.contains("firmware"))
                && (value.contains("Betaflight")
                    || value.contains("EmuFlight")
                    || value.contains("INAV"))
            {
                return Some(8000.0); // 8kHz default for modern firmware
            }
        }
    }

    None // Will use default in calling code
}

/// Parse filter configurations from header metadata
pub fn parse_emuflight_filters(headers: &[(String, String)]) -> AllFilterConfigs {
    let header_map: HashMap<String, String> = headers
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    let mut config = AllFilterConfigs::default();

    // Parse D-term filters (per-axis)
    for (axis_idx, axis_name) in ["roll", "pitch", "yaw"].iter().enumerate() {
        // D-term LPF1
        if let (Some(filter_type_str), Some(cutoff_str)) = (
            header_map.get("dterm_filter_type"),
            header_map.get(&format!("dterm_lowpass_hz_{axis_name}")),
        ) {
            if let (Ok(filter_type_num), Ok(cutoff_hz)) =
                (filter_type_str.parse::<u32>(), cutoff_str.parse::<f64>())
            {
                if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                    if cutoff_hz > 0.0 {
                        // Only add if cutoff > 0
                        config.dterm[axis_idx].lpf1 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz,
                            enabled: true,
                        });
                    }
                }
            }
        }

        // D-term LPF2
        if let (Some(filter_type_str), Some(cutoff_str)) = (
            header_map.get("dterm_filter2_type"),
            header_map.get(&format!("dterm_lowpass2_hz_{axis_name}")),
        ) {
            if let (Ok(filter_type_num), Ok(cutoff_hz)) =
                (filter_type_str.parse::<u32>(), cutoff_str.parse::<f64>())
            {
                if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                    if cutoff_hz > 0.0 {
                        // Only add if cutoff > 0
                        config.dterm[axis_idx].lpf2 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz,
                            enabled: true,
                        });
                    }
                }
            }
        }

        // Gyro filters (per-axis)
        if let (Some(filter_type_str), Some(cutoff_str)) = (
            header_map.get("gyro_lowpass_type"),
            header_map.get(&format!("gyro_lowpass_hz_{axis_name}")),
        ) {
            if let (Ok(filter_type_num), Ok(cutoff_hz)) =
                (filter_type_str.parse::<u32>(), cutoff_str.parse::<f64>())
            {
                if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                    if cutoff_hz > 0.0 {
                        // Only add if cutoff > 0
                        config.gyro[axis_idx].lpf1 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz,
                            enabled: true,
                        });
                    }
                }
            }
        }

        // Gyro LPF2 (per-axis)
        if let (Some(filter_type_str), Some(cutoff_str)) = (
            header_map.get("gyro_lowpass2_type"),
            header_map.get(&format!("gyro_lowpass2_hz_{axis_name}")),
        ) {
            if let (Ok(filter_type_num), Ok(cutoff_hz)) =
                (filter_type_str.parse::<u32>(), cutoff_str.parse::<f64>())
            {
                if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                    if cutoff_hz > 0.0 {
                        // Only add if cutoff > 0
                        config.gyro[axis_idx].lpf2 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz,
                            enabled: true,
                        });
                    }
                }
            }
        }
    }

    config
}

/// Parse Betaflight unified filter configuration from headers
pub fn parse_betaflight_filters(headers: &[(String, String)]) -> AllFilterConfigs {
    let header_map: HashMap<String, String> = headers
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    let mut config = AllFilterConfigs::default();

    // Parse D-term filters (unified across all axes)
    if let Some(filter_type_str) = header_map.get("dterm_lpf1_type") {
        if let Ok(filter_type_num) = filter_type_str.parse::<u32>() {
            if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                // Check for static vs dynamic mode
                let static_cutoff = header_map
                    .get("dterm_lpf1_static_hz")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let dynamic_cutoffs = header_map
                    .get("dterm_lpf1_dyn_hz")
                    .map(|s| parse_dynamic_cutoffs(s))
                    .unwrap_or((0.0, 0.0));

                let expo = header_map
                    .get("dterm_lpf1_dyn_expo")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);

                // Apply to all axes - prioritize dynamic over static when both are present
                for axis_idx in 0..AXIS_NAMES.len() {
                    if dynamic_cutoffs.0 > 0.0 && dynamic_cutoffs.1 > 0.0 {
                        // Dynamic mode takes precedence
                        config.dterm[axis_idx].dynamic_lpf1 = Some(DynamicFilterConfig {
                            filter_type,
                            min_cutoff_hz: dynamic_cutoffs.0,
                            max_cutoff_hz: dynamic_cutoffs.1,
                            expo,
                            enabled: true,
                        });
                    } else if static_cutoff > 0.0 {
                        // Static mode fallback
                        config.dterm[axis_idx].lpf1 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz: static_cutoff,
                            enabled: true,
                        });
                    }
                }
            }
        }
    }

    // Parse D-term LPF2 (static only)
    if let Some(filter_type_str) = header_map.get("dterm_lpf2_type") {
        if let Ok(filter_type_num) = filter_type_str.parse::<u32>() {
            if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                let static_cutoff = header_map
                    .get("dterm_lpf2_static_hz")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                if static_cutoff > 0.0 {
                    for axis_idx in 0..AXIS_NAMES.len() {
                        config.dterm[axis_idx].lpf2 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz: static_cutoff,
                            enabled: true,
                        });
                    }
                }
            }
        }
    }

    // Parse Gyro filters (similar pattern)
    if let Some(filter_type_str) = header_map.get("gyro_lpf1_type") {
        if let Ok(filter_type_num) = filter_type_str.parse::<u32>() {
            if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                let static_cutoff = header_map
                    .get("gyro_lpf1_static_hz")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let dynamic_cutoffs = header_map
                    .get("gyro_lpf1_dyn_hz")
                    .map(|s| parse_dynamic_cutoffs(s))
                    .unwrap_or((0.0, 0.0));

                let expo = header_map
                    .get("gyro_lpf1_dyn_expo")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);

                for axis_idx in 0..AXIS_NAMES.len() {
                    if dynamic_cutoffs.0 > 0.0 && dynamic_cutoffs.1 > 0.0 {
                        // Dynamic mode takes precedence
                        config.gyro[axis_idx].dynamic_lpf1 = Some(DynamicFilterConfig {
                            filter_type,
                            min_cutoff_hz: dynamic_cutoffs.0,
                            max_cutoff_hz: dynamic_cutoffs.1,
                            expo,
                            enabled: true,
                        });
                    } else if static_cutoff > 0.0 {
                        // Static mode fallback
                        config.gyro[axis_idx].lpf1 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz: static_cutoff,
                            enabled: true,
                        });
                    }
                }
            }
        }
    }

    // Parse Gyro LPF2 (static only)
    if let Some(filter_type_str) = header_map.get("gyro_lpf2_type") {
        if let Ok(filter_type_num) = filter_type_str.parse::<u32>() {
            if let Some(filter_type) = FilterType::from_u32(filter_type_num) {
                let static_cutoff = header_map
                    .get("gyro_lpf2_static_hz")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                if static_cutoff > 0.0 {
                    for axis_idx in 0..AXIS_NAMES.len() {
                        config.gyro[axis_idx].lpf2 = Some(FilterConfig {
                            filter_type,
                            cutoff_hz: static_cutoff,
                            enabled: true,
                        });
                    }
                }
            }
        }
    }

    config
}

/// Parse dynamic cutoff string format "min,max" into tuple
fn parse_dynamic_cutoffs(cutoff_str: &str) -> (f64, f64) {
    let parts: Vec<&str> = cutoff_str.trim_matches('"').split(',').collect();
    if parts.len() == 2 {
        let min_cutoff = parts[0].trim().parse::<f64>().unwrap_or(0.0);
        let max_cutoff = parts[1].trim().parse::<f64>().unwrap_or(0.0);
        (min_cutoff, max_cutoff)
    } else {
        (0.0, 0.0)
    }
}

/// Calculate effective IMUF filter cutoff frequency accounting for PTN scaling factors
/// Based on IMU-F ptnFilter.c: Adj_f_cut = (float)f_cut * ScaleF[filter->order - 1]
fn calculate_imuf_effective_cutoff(configured_cutoff_hz: f64, ptn_order: u32) -> f64 {
    // PTN filter scaling factors from IMU-F source code
    const PTN_SCALE_FACTORS: [f64; 4] = [
        1.0,         // PT1 (order 1)
        1.553773974, // PT2 (order 2)
        1.961459177, // PT3 (order 3)
        2.298959223, // PT4 (order 4)
    ];

    let scale_factor = match ptn_order {
        1..=4 => PTN_SCALE_FACTORS[(ptn_order - 1) as usize],
        _ => 1.553773974, // Default to PT2 scaling
    };

    configured_cutoff_hz * scale_factor
}

/// Parse IMUF filters from EmuFlight headers
pub fn parse_imuf_filters_with_gyro_rate(
    headers: &[(String, String)],
    _gyro_rate_hz: Option<f64>,
) -> AllFilterConfigs {
    let header_map: HashMap<String, String> = headers
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    let mut config = AllFilterConfigs::default();

    // Parse IMUF parameters for each axis
    let axis_names = ["roll", "pitch", "yaw"];
    for (axis_idx, axis_name) in axis_names.iter().enumerate() {
        // Parse per-axis lowpass cutoff frequencies
        let lowpass_key = format!("imuf_lowpass_{}", axis_name);
        let lowpass_cutoff_hz = header_map
            .get(&lowpass_key)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        // Parse per-axis Q-factors (scaled by 1000 in headers)
        let q_key = format!("imuf_{}_q", axis_name);
        let q_factor_scaled = header_map
            .get(&q_key)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let q_factor = q_factor_scaled / 1000.0; // Scale down from header value

        // Parse PTn filter order (applies to all axes)
        let ptn_order = header_map
            .get("imuf_ptn_order")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(2); // Default to PT2 if not specified

        // Create IMUF filter config if we have valid parameters
        if lowpass_cutoff_hz > 0.0 {
            // Calculate effective cutoff after PTN scaling
            let effective_cutoff_hz = calculate_imuf_effective_cutoff(lowpass_cutoff_hz, ptn_order);

            config.gyro[axis_idx].imuf = Some(ImufFilterConfig {
                lowpass_cutoff_hz,
                ptn_order,
                q_factor,
                effective_cutoff_hz,
                enabled: true,
            });
        }
    }

    config
}

/// Auto-detect firmware type and parse appropriate filter configuration
pub fn parse_filter_config(headers: &[(String, String)]) -> AllFilterConfigs {
    // Detect firmware type by looking for characteristic fields
    let header_map: HashMap<String, String> = headers
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    // Check for IMUF patterns (EmuFlight HELIOSPRING)
    let has_imuf_pattern = header_map.contains_key("imuf_lowpass_roll")
        || header_map.contains_key("imuf_lowpass_pitch")
        || header_map.contains_key("imuf_lowpass_yaw")
        || header_map.contains_key("imuf_ptn_order");

    // Check for EmuFlight per-axis patterns
    let has_emuflight_pattern = header_map.contains_key("dterm_lowpass_hz_roll")
        || header_map.contains_key("gyro_lowpass_hz_roll");

    // Check for Betaflight unified patterns
    let has_betaflight_pattern = header_map.contains_key("dterm_lpf1_static_hz")
        || header_map.contains_key("dterm_lpf1_dyn_hz");

    let mut config = if has_emuflight_pattern {
        println!("Detected EmuFlight filter configuration (per-axis)");
        parse_emuflight_filters(headers)
    } else if has_betaflight_pattern {
        println!("Detected Betaflight filter configuration (unified)");
        parse_betaflight_filters(headers)
    } else {
        println!("No recognized filter configuration found in headers");
        AllFilterConfigs::default()
    };

    // Check for IMUF configuration and merge it with existing config
    if has_imuf_pattern {
        println!("Detected EmuFlight HELIOSPRING IMUF v256 filter configuration");

        // Extract gyro rate for sample rate correction calculations
        let gyro_rate_hz = extract_gyro_rate(Some(headers));
        let imuf_config = parse_imuf_filters_with_gyro_rate(headers, gyro_rate_hz);

        // Merge IMUF filters into the existing configuration
        for axis_idx in 0..3 {
            if let Some(ref imuf_filter) = imuf_config.gyro[axis_idx].imuf {
                config.gyro[axis_idx].imuf = Some(imuf_filter.clone());
            }
        }
    } // Debug output: Print parsed filter configuration
    println!("Parsed filter configuration:");
    for (axis_idx, axis_name) in AXIS_NAMES.iter().enumerate() {
        println!("  {axis_name} Gyro Filters:");
        if let Some(ref lpf1) = config.gyro[axis_idx].lpf1 {
            println!(
                "    LPF1: {} at {:.0} Hz",
                lpf1.filter_type.name(),
                lpf1.cutoff_hz
            );
        }
        if let Some(ref lpf2) = config.gyro[axis_idx].lpf2 {
            println!(
                "    LPF2: {} at {:.0} Hz",
                lpf2.filter_type.name(),
                lpf2.cutoff_hz
            );
        }
        if let Some(ref dyn_lpf1) = config.gyro[axis_idx].dynamic_lpf1 {
            println!(
                "    Dyn LPF1: {} {:.0}-{:.0} Hz (expo: {})",
                dyn_lpf1.filter_type.name(),
                dyn_lpf1.min_cutoff_hz,
                dyn_lpf1.max_cutoff_hz,
                dyn_lpf1.expo
            );
        }
        if let Some(ref imuf) = config.gyro[axis_idx].imuf {
            let filter_type = match imuf.ptn_order {
                1 => "PT1",
                2 => "PT2",
                3 => "PT3",
                4 => "PT4",
                _ => "PT2",
            };
            println!(
                "    IMUF v256: {} Header={:.0}Hz → Effective={:.0}Hz (Q={:.1})",
                filter_type, imuf.lowpass_cutoff_hz, imuf.effective_cutoff_hz, imuf.q_factor
            );

            // Show warnings for significant differences
            let ptn_scaling_diff = imuf.effective_cutoff_hz - imuf.lowpass_cutoff_hz;

            if ptn_scaling_diff.abs() > 5.0 {
                println!(
                    "      WARNING: PTN scaling increases cutoff by {:.0}Hz ({:.1}x multiplier)",
                    ptn_scaling_diff,
                    imuf.effective_cutoff_hz / imuf.lowpass_cutoff_hz
                );
            }
        }
        if config.gyro[axis_idx].lpf1.is_none()
            && config.gyro[axis_idx].lpf2.is_none()
            && config.gyro[axis_idx].dynamic_lpf1.is_none()
            && config.gyro[axis_idx].imuf.is_none()
        {
            println!("    No gyro filters configured");
        }
    }

    config
}

/// Measure actual filter response from spectrum data (magnitude vs frequency)
/// This works for ANY firmware - Betaflight, EmuFlight, IMUF, etc.
/// Returns measured filter characteristics extracted from real spectral data
pub fn measure_filter_response(
    unfiltered_spectrum: &[(f64, f64)], // (frequency, magnitude) pairs
    filtered_spectrum: &[(f64, f64)],   // (frequency, magnitude) pairs
    _sample_rate: f64,                  // Not needed since we have frequency data
) -> Result<MeasuredFilterResponse, Box<dyn std::error::Error>> {
    // Validate input data
    if filtered_spectrum.len() != unfiltered_spectrum.len() {
        return Err("Filtered and unfiltered spectra must have same length".into());
    }
    if filtered_spectrum.len() < MIN_SPECTRUM_POINTS_FOR_ANALYSIS {
        return Err(format!(
            "Need at least {} frequency points for reliable analysis",
            MIN_SPECTRUM_POINTS_FOR_ANALYSIS
        )
        .into());
    }

    // Calculate attenuation (what the filter removed) instead of transfer function ratio
    let mut attenuation_spectrum = Vec::with_capacity(filtered_spectrum.len());

    let eps = 1e-12_f64;
    let freq_tol = 1e-6_f64;
    for i in 0..filtered_spectrum.len() {
        let (f_filt, filt_mag) = filtered_spectrum[i];
        let (f_unf, unfilt_mag) = unfiltered_spectrum[i];
        if !f_filt.is_finite()
            || !f_unf.is_finite()
            || !filt_mag.is_finite()
            || !unfilt_mag.is_finite()
        {
            continue;
        }
        // Ensure bins align
        if (f_filt - f_unf).abs() > freq_tol {
            continue;
        }
        // Skip very low magnitude signals to avoid noise
        if unfilt_mag.abs() <= eps {
            continue;
        }

        // Calculate attenuation: what the filter removed
        let attenuation = (unfilt_mag - filt_mag).abs();

        if attenuation.is_finite() && attenuation >= 0.0 {
            attenuation_spectrum.push((f_filt, attenuation));
        }
    }

    if attenuation_spectrum.is_empty() {
        return Err("No valid attenuation data found".into());
    }

    // Ensure ascending frequency for downstream analysis
    attenuation_spectrum.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff frequency based on attenuation analysis
    let cutoff_hz = find_cutoff_from_attenuation(&attenuation_spectrum)?;

    // Estimate filter order from attenuation slope
    let (filter_order, order_confidence) =
        estimate_filter_order_from_attenuation(&attenuation_spectrum, cutoff_hz)?;

    // Calculate confidence based on data quality
    let general_confidence = calculate_attenuation_confidence(&attenuation_spectrum);
    let overall_confidence = (order_confidence * general_confidence).clamp(0.0, 1.0);

    Ok(MeasuredFilterResponse {
        cutoff_hz,
        filter_order,
        confidence: overall_confidence,
        transfer_function: attenuation_spectrum, // Store attenuation data instead
    })
}

#[derive(Debug, Clone)]
pub struct MeasuredFilterResponse {
    pub cutoff_hz: f64,
    pub filter_order: f64, // 1.0 = PT1, 2.0 = PT2, etc.
    pub confidence: f64,   // 0.0-1.0
    #[allow(dead_code)]
    pub transfer_function: Vec<(f64, f64)>, // (freq, magnitude)
}

/// Find cutoff frequency from attenuation spectrum
/// Looks for the frequency where attenuation reaches a significant level
fn find_cutoff_from_attenuation(
    attenuation_spectrum: &[(f64, f64)],
) -> Result<f64, Box<dyn std::error::Error>> {
    if attenuation_spectrum.len() < 10 {
        return Err("Insufficient points for attenuation-based cutoff detection".into());
    }

    // Find the maximum attenuation to use as reference
    let max_attenuation = attenuation_spectrum
        .iter()
        .map(|(_, atten)| *atten)
        .fold(0.0, f64::max);

    if max_attenuation <= 0.0 {
        return Err("No significant attenuation found".into());
    }

    // Look for the frequency where attenuation reaches a significant fraction of max
    // Find the FIRST ascending crossing (where filtering starts), not the peak
    let target_attenuation = max_attenuation * 0.2; // 20% of max - earlier in the filter response

    // Search in the valid frequency range (avoid low-freq noise and high-freq edge)
    let start_idx = attenuation_spectrum.len() / 10;
    let end_idx = (attenuation_spectrum.len() * 4) / 5;

    for i in start_idx..(end_idx.min(attenuation_spectrum.len() - 1)) {
        let (f1, a1) = attenuation_spectrum[i];
        let (f2, a2) = attenuation_spectrum[i + 1];

        // Skip unrealistic frequencies
        if f1 < 40.0 || f2 > 800.0 {
            continue;
        }

        // Look for ascending crossing: below target → above target
        if a1 <= target_attenuation && a2 >= target_attenuation {
            // Linear interpolation
            let denom = a2 - a1;
            if denom.abs() > 1e-12 {
                let t = (target_attenuation - a1) / denom;
                let fc = f1 + t * (f2 - f1);
                if fc.is_finite() && (40.0..=800.0).contains(&fc) {
                    return Ok(fc);
                }
            }
        }
    }

    Err("Could not find attenuation-based cutoff frequency".into())
}

/// Estimate filter order from attenuation spectrum slope analysis
fn estimate_filter_order_from_attenuation(
    attenuation_spectrum: &[(f64, f64)],
    cutoff_hz: f64,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    // Analyze the slope of attenuation above cutoff frequency
    let analysis_points: Vec<(f64, f64)> = attenuation_spectrum
        .iter()
        .filter(|(freq, atten)| *freq > cutoff_hz * 1.2 && *freq < 800.0 && *atten > 0.0)
        .map(|(freq, atten)| (freq.ln(), atten.ln()))
        .collect();

    if analysis_points.len() < 5 {
        // Fallback to default filter characteristics
        return Ok((1.5, 0.3)); // Moderate order with low confidence
    }

    // Linear regression on log-log plot to find slope
    let n = analysis_points.len() as f64;
    let sum_x: f64 = analysis_points.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = analysis_points.iter().map(|(_, y)| *y).sum();
    let sum_xy: f64 = analysis_points.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = analysis_points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return Ok((1.0, 0.2));
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // For attenuation, positive slope indicates stronger filtering
    let estimated_order = slope.abs().clamp(0.5, 4.0);

    // Calculate confidence based on regression quality
    let r_squared = {
        let y_mean = sum_y / n;
        let ss_tot: f64 = analysis_points
            .iter()
            .map(|(_, y)| (y - y_mean).powi(2))
            .sum();
        let ss_res: f64 = analysis_points
            .iter()
            .map(|(x, y)| {
                let y_pred = (sum_y - slope * sum_x) / n + slope * x;
                (y - y_pred).powi(2)
            })
            .sum();
        if ss_tot > 1e-12 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        }
    };

    let confidence = r_squared.clamp(0.1, 1.0);

    Ok((estimated_order, confidence))
}

/// Calculate confidence score for attenuation-based measurement
fn calculate_attenuation_confidence(attenuation_spectrum: &[(f64, f64)]) -> f64 {
    if attenuation_spectrum.len() < 10 {
        return 0.1;
    }

    // Factor 1: Sufficient data points
    let points_factor = (attenuation_spectrum.len() as f64 / 100.0).clamp(0.3, 1.0);

    // Factor 2: Signal-to-noise ratio in attenuation
    let attenuations: Vec<f64> = attenuation_spectrum.iter().map(|(_, a)| *a).collect();
    let max_atten = attenuations.iter().fold(0.0_f64, |a, &b| a.max(b));
    let min_atten = attenuations.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    let dynamic_range = if max_atten > 0.0 {
        (max_atten / (min_atten + 1e-12)).ln()
    } else {
        0.0
    };
    let snr_factor = (dynamic_range / 5.0).clamp(0.2, 1.0); // Expect ~5 orders of magnitude

    // Combine factors
    (points_factor * 0.6 + snr_factor * 0.4).clamp(0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pt1_response() {
        // At cutoff frequency, magnitude should be ~0.707 (-3dB)
        let response = pt1_response(100.0, 100.0);
        assert!((response - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.001);

        // At 10x cutoff frequency, should be much lower
        let response_10x = pt1_response(1000.0, 100.0);
        assert!(response_10x < 0.1);
    }

    #[test]
    fn test_pt2_response() {
        // At cutoff, should be ~ -3 dB
        let r_fc = pt2_response(200.0, 200.0);
        assert!((r_fc - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.001);
        // Far above cutoff, very small
        let r_10x = pt2_response(2000.0, 200.0);
        assert!(r_10x < 0.02);
    }

    #[test]
    fn test_pt4_response() {
        // At cutoff, should be ~ -3 dB
        let r_fc = pt4_response(150.0, 150.0);
        assert!((r_fc - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_pt3_response() {
        // At cutoff, should be ~ -3 dB
        let r_fc = pt3_response(300.0, 300.0);
        assert!((r_fc - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.001);
        // Far above cutoff, very small
        let r_10x = pt3_response(3000.0, 300.0);
        assert!(r_10x < 0.01);
    }

    #[test]
    fn test_extract_gyro_rate_with_sync_denom() {
        let headers = vec![
            ("looptime".to_string(), "125".to_string()), // 1e6 / 125 = 8000 Hz
            ("gyro_sync_denom".to_string(), "2".to_string()), // -> 4000 Hz
        ];
        let rate = extract_gyro_rate(Some(&headers));
        assert_eq!(rate, Some(4000.0));
    }

    #[test]
    fn test_filter_type_conversion() {
        assert_eq!(FilterType::from_u32(0), Some(FilterType::PT1));
        assert_eq!(FilterType::from_u32(1), Some(FilterType::Biquad));
        assert_eq!(FilterType::from_u32(2), Some(FilterType::PT2));
        assert_eq!(FilterType::from_u32(3), Some(FilterType::PT3));
        assert_eq!(FilterType::from_u32(99), None);
    }

    #[test]
    fn test_dynamic_cutoffs_parsing() {
        assert_eq!(parse_dynamic_cutoffs("75,150"), (75.0, 150.0));
        assert_eq!(parse_dynamic_cutoffs("\"100,200\""), (100.0, 200.0));
        assert_eq!(parse_dynamic_cutoffs("invalid"), (0.0, 0.0));
    }

    #[test]
    fn test_imuf_filter_parsing() {
        let headers = vec![
            ("IMUF_lowpass_roll".to_string(), "90".to_string()),
            ("IMUF_lowpass_pitch".to_string(), "95".to_string()),
            ("IMUF_lowpass_yaw".to_string(), "85".to_string()),
            ("IMUF_roll_q".to_string(), "8000".to_string()),
            ("IMUF_pitch_q".to_string(), "7500".to_string()),
            ("IMUF_yaw_q".to_string(), "9000".to_string()),
            ("IMUF_ptn_order".to_string(), "3".to_string()),
        ];

        let config = parse_imuf_filters_with_gyro_rate(&headers, None);

        // Check roll axis
        assert!(config.gyro[0].imuf.is_some());
        let roll_imuf = config.gyro[0].imuf.as_ref().unwrap();
        assert_eq!(roll_imuf.lowpass_cutoff_hz, 90.0);
        assert_eq!(roll_imuf.q_factor, 8.0); // 8000 / 1000
        assert_eq!(roll_imuf.ptn_order, 3);
        assert!(roll_imuf.enabled);

        // Check pitch axis
        assert!(config.gyro[1].imuf.is_some());
        let pitch_imuf = config.gyro[1].imuf.as_ref().unwrap();
        assert_eq!(pitch_imuf.lowpass_cutoff_hz, 95.0);
        assert_eq!(pitch_imuf.q_factor, 7.5); // 7500 / 1000

        // Check yaw axis
        assert!(config.gyro[2].imuf.is_some());
        let yaw_imuf = config.gyro[2].imuf.as_ref().unwrap();
        assert_eq!(yaw_imuf.lowpass_cutoff_hz, 85.0);
        assert_eq!(yaw_imuf.q_factor, 9.0); // 9000 / 1000
    }

    #[test]
    fn test_imuf_filter_detection() {
        let headers = vec![
            ("IMUF_lowpass_roll".to_string(), "90".to_string()),
            ("IMUF_ptn_order".to_string(), "2".to_string()),
        ];

        let config = parse_filter_config(&headers);
        assert!(config.gyro[0].imuf.is_some());
    }

    #[test]
    fn test_imuf_calculation_functions() {
        // Test PTN scaling factors
        assert_eq!(calculate_imuf_effective_cutoff(100.0, 1), 100.0); // PT1 no scaling
        assert!((calculate_imuf_effective_cutoff(100.0, 2) - 155.377).abs() < 0.1); // PT2 scaling
        assert!((calculate_imuf_effective_cutoff(100.0, 3) - 196.146).abs() < 0.1); // PT3 scaling
        assert!((calculate_imuf_effective_cutoff(100.0, 4) - 229.896).abs() < 0.1);
        // PT4 scaling
    }

    #[test]
    fn test_imuf_filter_curves() {
        let imuf_config = ImufFilterConfig {
            lowpass_cutoff_hz: 100.0,
            ptn_order: 2,
            q_factor: 8.0,
            effective_cutoff_hz: 155.4, // 100 * 1.553773974
            enabled: true,
        };

        let axis_config = AxisFilterConfig {
            lpf1: None,
            lpf2: None,
            dynamic_lpf1: None,
            imuf: Some(imuf_config),
        };

        let curves = generate_individual_filter_curves(&axis_config, 1000.0, 100);
        assert!(!curves.is_empty()); // Should have at least header curve, possibly more

        // Check header curve
        let (header_label, _curve_points, header_cutoff) = &curves[0];
        assert!(header_label.contains("IMUF v256 Header"));
        assert!(header_label.contains("PT2"));
        assert!(header_label.contains("100Hz"));
        assert_eq!(*header_cutoff, 100.0);

        // Check for effective curve if different enough
        if curves.len() > 1 {
            let (effective_label, _, effective_cutoff) = &curves[1];
            assert!(effective_label.contains("IMUF v256 Effective"));
            assert!(*effective_cutoff > 150.0); // Should be scaled up
        }
    }
}
