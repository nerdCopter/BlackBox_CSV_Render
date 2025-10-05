// src/data_analysis/filter_response.rs

use std::collections::HashMap;

use crate::axis_names::AXIS_NAMES;

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

/// IMUF filter configuration with Butterworth correction (Betaflight/EmuFlight)
#[derive(Debug, Clone)]
pub struct ImufFilterConfig {
    // Header values (as configured)
    pub lowpass_cutoff_hz: f64, // Configured combined cutoff from header (effective -3dB point)
    pub ptn_order: u32,         // Filter order (1-4 -> PT1, PT2, PT3, PT4)
    pub q_factor: f64,          // Q-factor (scaled by 1000 in headers)
    pub revision: Option<u32>,  // IMUF revision from header (e.g., 256, 257)

    // Calculated values (Butterworth correction for cascaded stages)
    pub effective_cutoff_hz: f64, // Per-stage PT1 cutoff (scaled up to achieve configured combined response)

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

/// Helper function to add PTn filter curves with Butterworth correction
/// Generates both the user-configured curve and the per-stage implementation curve
fn add_ptn_filter_curves(
    filter_curves: &mut Vec<FilterCurveData>,
    filter_type: FilterType,
    cutoff_hz: f64,
    label_prefix: &str,
    max_frequency_hz: f64,
    num_points: usize,
    show_butterworth: bool,
) {
    // Generate curve for user-configured cutoff (e.g., PT2 @ 90Hz)
    let main_filter = FilterConfig {
        filter_type,
        cutoff_hz,
        enabled: true,
    };
    let main_curve = generate_single_filter_curve(&main_filter, max_frequency_hz, num_points);
    let main_label = format!(
        "{} ({} @ {:.0}Hz)",
        label_prefix,
        filter_type.name(),
        cutoff_hz
    );
    filter_curves.push((main_label, main_curve, cutoff_hz));

    // Generate per-stage curve for PT2/PT3/PT4 (Butterworth correction) only if --butterworth flag is set
    if !show_butterworth {
        return;
    }

    let ptn_order = match filter_type {
        FilterType::PT2 => 2,
        FilterType::PT3 => 3,
        FilterType::PT4 => 4,
        _ => 0, // PT1 and BIQUAD don't need per-stage curves
    };

    if ptn_order > 1 {
        let per_stage_cutoff = calculate_ptn_per_stage_cutoff(cutoff_hz, ptn_order);

        // Only show per-stage if it's meaningfully different (> 1 Hz)
        if (per_stage_cutoff - cutoff_hz).abs() > 1.0 {
            let stage_count_text = match ptn_order {
                2 => "Two PT1",
                3 => "Three PT1",
                4 => "Four PT1",
                _ => "PT1",
            };

            let stage_filter = FilterConfig {
                filter_type,
                cutoff_hz: per_stage_cutoff,
                enabled: true,
            };
            let stage_curve =
                generate_single_filter_curve(&stage_filter, max_frequency_hz, num_points);
            let stage_label = format!(
                "≈ {} ({} @ {:.0}Hz per-stage)",
                label_prefix, stage_count_text, per_stage_cutoff
            );
            filter_curves.push((stage_label, stage_curve, per_stage_cutoff));
        }
    }
}

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
    show_butterworth: bool,
) -> Vec<FilterCurveData> {
    let mut filter_curves = Vec::new();

    // Generate curve for LPF1 (static or dynamic - whichever is configured)
    if let Some(ref dyn_lpf1) = axis_config.dynamic_lpf1 {
        // Dynamic LPF1 takes precedence
        if dyn_lpf1.enabled && dyn_lpf1.min_cutoff_hz > 0.0 {
            let label_prefix = if dyn_lpf1.max_cutoff_hz > dyn_lpf1.min_cutoff_hz {
                format!(
                    "Dyn LPF1 {:.0}-{:.0}Hz",
                    dyn_lpf1.min_cutoff_hz, dyn_lpf1.max_cutoff_hz
                )
            } else {
                "Dyn LPF1".to_string()
            };

            // Use helper to add both main and per-stage curves for PTn filters
            add_ptn_filter_curves(
                &mut filter_curves,
                dyn_lpf1.filter_type,
                dyn_lpf1.min_cutoff_hz,
                &label_prefix,
                max_frequency_hz,
                num_points,
                show_butterworth,
            );
        }
    } else if let Some(ref lpf1) = axis_config.lpf1 {
        // Static LPF1 fallback
        if lpf1.enabled && lpf1.cutoff_hz > 0.0 {
            // Use helper to add both main and per-stage curves for PTn filters
            add_ptn_filter_curves(
                &mut filter_curves,
                lpf1.filter_type,
                lpf1.cutoff_hz,
                "LPF1",
                max_frequency_hz,
                num_points,
                show_butterworth,
            );
        }
    }

    // Generate curve for LPF2 if enabled (always after LPF1)
    if let Some(ref lpf2) = axis_config.lpf2 {
        if lpf2.enabled && lpf2.cutoff_hz > 0.0 {
            // Use helper to add both main and per-stage curves for PTn filters
            add_ptn_filter_curves(
                &mut filter_curves,
                lpf2.filter_type,
                lpf2.cutoff_hz,
                "LPF2",
                max_frequency_hz,
                num_points,
                show_butterworth,
            );
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

            // Helper to get stage count text for per-stage labels
            let stage_count_text = match imuf.ptn_order {
                1 => "One PT1",
                2 => "Two PT1",
                3 => "Three PT1",
                4 => "Four PT1",
                _ => "Two PT1",
            };

            // Build version string
            let version_str = if let Some(rev) = imuf.revision {
                format!("IMUF v{}", rev)
            } else {
                "IMUF".to_string()
            };

            // Generate curve for header-configured cutoff (user sees PT2 @ 90Hz)
            let header_filter = FilterConfig {
                filter_type,
                cutoff_hz: imuf.lowpass_cutoff_hz,
                enabled: true,
            };
            let header_curve =
                generate_single_filter_curve(&header_filter, max_frequency_hz, num_points);
            let header_label = format!(
                "{} ({} @ {:.0}Hz)",
                version_str,
                filter_type.name(),
                imuf.lowpass_cutoff_hz
            );
            filter_curves.push((header_label, header_curve, imuf.lowpass_cutoff_hz));

            // Generate curve for per-stage PT1 cutoff (Butterworth correction - shows Two PT1 @ 140Hz) only if --butterworth flag is set
            if show_butterworth && (imuf.effective_cutoff_hz - imuf.lowpass_cutoff_hz).abs() > 1.0 {
                let stage_filter = FilterConfig {
                    filter_type,
                    cutoff_hz: imuf.effective_cutoff_hz,
                    enabled: true,
                };
                let stage_curve =
                    generate_single_filter_curve(&stage_filter, max_frequency_hz, num_points);
                let stage_label = format!(
                    "≈ {} ({} @ {:.0}Hz per-stage)",
                    version_str, stage_count_text, imuf.effective_cutoff_hz
                );
                filter_curves.push((stage_label, stage_curve, imuf.effective_cutoff_hz));
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

/// Calculate per-stage PT1 cutoff frequency for IMUF filters
/// Uses Butterworth correction factors to achieve proper combined response
/// Based on IMU-F ptnFilter.c: Adj_f_cut = (float)f_cut * ScaleF[filter->order - 1]
///
/// Example: For PT2 at 90 Hz combined:
/// - Each PT1 stage runs at 90 * 1.554 = 140 Hz
/// - Two stages at 140 Hz produce -3dB at 90 Hz combined
fn calculate_ptn_per_stage_cutoff(configured_cutoff_hz: f64, ptn_order: u32) -> f64 {
    // PTn cutoff correction factors from Betaflight/EmuFlight source
    // Formula: 1 / sqrt(2^(1/n) - 1) where n is the filter order
    const PTN_SCALE_FACTORS: [f64; 4] = [
        1.0,         // PT1 (order 1) - no correction needed
        1.553773974, // PT2 (order 2) - CUTOFF_CORRECTION_PT2
        1.961459177, // PT3 (order 3) - CUTOFF_CORRECTION_PT3
        2.298959223, // PT4 (order 4) - CUTOFF_CORRECTION_PT4
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

    // Parse IMUF revision (applies to all axes)
    let imuf_revision = header_map
        .get("imuf_revision")
        .and_then(|s| s.parse::<u32>().ok());

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
            // Calculate per-stage PT1 cutoff (Butterworth correction)
            let effective_cutoff_hz = calculate_ptn_per_stage_cutoff(lowpass_cutoff_hz, ptn_order);

            config.gyro[axis_idx].imuf = Some(ImufFilterConfig {
                lowpass_cutoff_hz,
                ptn_order,
                q_factor,
                revision: imuf_revision,
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

    // Check for IMUF patterns (Betaflight/EmuFlight with PTn Butterworth correction)
    let has_imuf_pattern = header_map.contains_key("imuf_lowpass_roll")
        || header_map.contains_key("imuf_lowpass_pitch")
        || header_map.contains_key("imuf_lowpass_yaw")
        || header_map.contains_key("imuf_ptn_order")
        || header_map.contains_key("imuf_revision");

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
        println!("Detected PTn filters with Butterworth correction (IMUF)");

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
            let stage_count_text = match imuf.ptn_order {
                1 => "One PT1",
                2 => "Two PT1",
                3 => "Three PT1",
                4 => "Four PT1",
                _ => "Two PT1",
            };

            let version_str = if let Some(rev) = imuf.revision {
                format!("IMUF v{}", rev)
            } else {
                "IMUF".to_string()
            };

            println!(
                "    {}: {} Combined={:.0}Hz → per-stage={:.0}Hz (Q={:.1})",
                version_str,
                stage_count_text,
                imuf.lowpass_cutoff_hz,
                imuf.effective_cutoff_hz,
                imuf.q_factor
            );

            // Show info about Butterworth correction
            let ptn_scaling_diff = imuf.effective_cutoff_hz - imuf.lowpass_cutoff_hz;

            if ptn_scaling_diff.abs() > 5.0 {
                println!(
                    "      Note: Butterworth correction requires PT1 stages at {:.0}Hz to achieve {:.0}Hz combined response",
                    imuf.effective_cutoff_hz,
                    imuf.lowpass_cutoff_hz
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
    fn test_ptn_calculation_functions() {
        // Test PTN scaling factors
        assert_eq!(calculate_ptn_per_stage_cutoff(100.0, 1), 100.0); // PT1 no scaling
        assert!((calculate_ptn_per_stage_cutoff(100.0, 2) - 155.377).abs() < 0.1); // PT2 scaling
        assert!((calculate_ptn_per_stage_cutoff(100.0, 3) - 196.146).abs() < 0.1); // PT3 scaling
        assert!((calculate_ptn_per_stage_cutoff(100.0, 4) - 229.896).abs() < 0.1);
        // PT4 scaling
    }

    #[test]
    fn test_imuf_filter_curves() {
        let imuf_config = ImufFilterConfig {
            lowpass_cutoff_hz: 100.0,
            ptn_order: 2,
            q_factor: 8.0,
            revision: Some(256),
            effective_cutoff_hz: 155.4, // Per-stage: 100 * 1.553773974
            enabled: true,
        };

        let axis_config = AxisFilterConfig {
            lpf1: None,
            lpf2: None,
            dynamic_lpf1: None,
            imuf: Some(imuf_config),
        };

        let curves = generate_individual_filter_curves(&axis_config, 1000.0, 100, true);
        assert!(!curves.is_empty()); // Should have at least combined curve, possibly per-stage too

        // Check combined response curve (red line - shows user-configured PT2 @ 100Hz)
        let (combined_label, _curve_points, combined_cutoff) = &curves[0];
        assert!(combined_label.contains("IMUF v256"));
        assert!(combined_label.contains("PT2")); // Should say "PT2" for user-configured filter
        assert!(combined_label.contains("100Hz"));
        assert!(!combined_label.contains("per-stage")); // Combined curve shouldn't have per-stage
        assert_eq!(*combined_cutoff, 100.0);

        // Check for per-stage curve if different enough (gray line - shows Two PT1 @ 155Hz)
        if curves.len() > 1 {
            let (stage_label, _, stage_cutoff) = &curves[1];
            assert!(stage_label.contains("IMUF v256"));
            assert!(stage_label.contains("Two PT1")); // Should say "Two PT1" for per-stage breakdown
            assert!(stage_label.contains("per-stage"));
            assert!(*stage_cutoff > 150.0); // Should be scaled up for Butterworth correction
        }
    }

    #[test]
    fn test_ptn_filter_curves_generation() {
        // Test that PT2 generates both main and per-stage curves
        let lpf1_pt2 = FilterConfig {
            filter_type: FilterType::PT2,
            cutoff_hz: 100.0,
            enabled: true,
        };

        let axis_config_pt2 = AxisFilterConfig {
            lpf1: Some(lpf1_pt2),
            lpf2: None,
            dynamic_lpf1: None,
            imuf: None,
        };

        let curves = generate_individual_filter_curves(&axis_config_pt2, 1000.0, 100, true);
        assert_eq!(
            curves.len(),
            2,
            "PT2 should generate 2 curves (main + per-stage)"
        );

        let (main_label, _, main_cutoff) = &curves[0];
        assert!(main_label.contains("LPF1"));
        assert!(main_label.contains("PT2"));
        assert!(main_label.contains("100Hz"));
        assert!(!main_label.contains("per-stage"));
        assert_eq!(*main_cutoff, 100.0);

        let (stage_label, _, stage_cutoff) = &curves[1];
        assert!(stage_label.contains("LPF1"));
        assert!(stage_label.contains("Two PT1"));
        assert!(stage_label.contains("per-stage"));
        assert!(*stage_cutoff > 150.0);

        // Test that PT1 generates only main curve
        let lpf1_pt1 = FilterConfig {
            filter_type: FilterType::PT1,
            cutoff_hz: 100.0,
            enabled: true,
        };

        let axis_config_pt1 = AxisFilterConfig {
            lpf1: Some(lpf1_pt1),
            lpf2: None,
            dynamic_lpf1: None,
            imuf: None,
        };

        let curves_pt1 = generate_individual_filter_curves(&axis_config_pt1, 1000.0, 100, false);
        assert_eq!(
            curves_pt1.len(),
            1,
            "PT1 should generate only 1 curve (no per-stage)"
        );
        assert!(curves_pt1[0].0.contains("PT1"));
        assert!(!curves_pt1[0].0.contains("per-stage"));
    }
}
