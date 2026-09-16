// src/data_analysis/rate_curve.rs

use std::collections::HashMap;

use crate::constants::{
    RATE_CURVE_DEFAULT_RATE_LIMIT_DPS, RATE_CURVE_RC_RATE_INCREMENTAL,
    RATE_CURVE_SETPOINT_LIMIT_DPS, RATE_CURVE_SUPER_EXPO_THRESHOLD, RATE_CURVE_SUPER_FACTOR_MIN,
};

/// Stick position at which every `apply_*_rate` function below is evaluated — full deflection,
/// matching firmware's own `applyRates(axis, 1.0f, 1.0f)` call used to compute its max rate.
const FULL_STICK: f64 = 1.0;

/// `rates_type` header value. Betaflight/RaceFlight/KISS/Actual formulas are identical between
/// Betaflight and EmuFlight firmware (verified against both source trees, Sept 2026). Quick is
/// Betaflight-only — EmuFlight's `rates_type` enum tops out at `Actual` (3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatesType {
    Betaflight,
    RaceFlight,
    Kiss,
    Actual,
    Quick,
}

impl RatesType {
    fn from_header_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Betaflight),
            1 => Some(Self::RaceFlight),
            2 => Some(Self::Kiss),
            3 => Some(Self::Actual),
            4 => Some(Self::Quick),
            _ => None,
        }
    }
}

/// Per-axis rate curve configuration, parsed from blackbox header metadata.
pub struct RateCurveConfig {
    pub rates_type: RatesType,
    pub rc_rate: [f64; 3],
    pub rc_expo: [f64; 3],
    pub super_rate: [f64; 3],
    /// Per-axis absolute setpoint ceiling (deg/s). Defaults to
    /// `RATE_CURVE_DEFAULT_RATE_LIMIT_DPS` when the `rate_limits` header key is absent
    /// (EmuFlight never logs it; this is also Betaflight's own compiled-in default).
    pub rate_limit: [f64; 3],
}

/// Parses rate curve configuration from header metadata. Returns `None` when `rc_rates`,
/// `rc_expo`, or `rates` is missing or malformed — without all three, no rate curve can be
/// evaluated — or when `rate_limits` is present but malformed (a present-but-unparseable value
/// is rejected outright rather than silently treated the same as an absent key). `rates_type`
/// defaults to `Betaflight` when the header key is absent, matching both firmwares' compiled-in
/// default and predating the header key's own introduction.
pub fn parse_rate_curve_config(header_metadata: &[(String, String)]) -> Option<RateCurveConfig> {
    if header_metadata.is_empty() {
        return None;
    }

    let header_map: HashMap<String, String> = header_metadata
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();

    let rc_rate = parse_axis_triple(&header_map, "rc_rates")?;
    let rc_expo = parse_axis_triple(&header_map, "rc_expo")?;
    let super_rate = parse_axis_triple(&header_map, "rates")?;
    // Distinguish "key absent" (default to the firmware ceiling) from "key present but
    // malformed" (reject outright) — silently defaulting a malformed value would mask a real
    // parse failure behind a plausible-looking number.
    let rate_limit = if header_map.contains_key("rate_limits") {
        parse_axis_triple(&header_map, "rate_limits")?
    } else {
        [RATE_CURVE_DEFAULT_RATE_LIMIT_DPS; 3]
    };

    let rates_type = header_map
        .get("rates_type")
        .and_then(|v| v.parse::<u8>().ok())
        .and_then(RatesType::from_header_value)
        .unwrap_or(RatesType::Betaflight);

    Some(RateCurveConfig {
        rates_type,
        rc_rate,
        rc_expo,
        super_rate,
        rate_limit,
    })
}

/// Parses a `"roll,pitch,yaw"`-style header value. Requires exactly 3 comma-separated tokens,
/// each a finite number — a `filter_map`-style "drop what doesn't parse" approach would let one
/// malformed middle token silently shift the remaining values onto the wrong axis.
fn parse_axis_triple(header_map: &HashMap<String, String>, key: &str) -> Option<[f64; 3]> {
    let raw = header_map.get(key)?;
    let parts: Vec<f64> = raw
        .trim_matches('"')
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
        })
        .collect::<Option<_>>()?;
    (parts.len() == 3).then(|| [parts[0], parts[1], parts[2]])
}

/// Configured max rate (deg/s) for one axis — the setpoint firmware produces at full stick
/// deflection, after the final per-axis `rate_limit` clamp. `None` when `axis` is out of range,
/// the curve evaluates to a non-finite value (e.g. a zero `rc_rate` under Quick, which divides
/// by it), or `rate_limit` is negative or non-finite — a malformed header could parse to a
/// negative number, and `f64::clamp` panics when its min bound exceeds its max. An unrecognized
/// `rates_type` header value never reaches this function — see `parse_rate_curve_config`, which
/// resolves it to `Betaflight` at parse time.
pub fn configured_max_rate(config: &RateCurveConfig, axis: usize) -> Option<f64> {
    let (rc_rate, rc_expo, super_rate, rate_limit) = (
        *config.rc_rate.get(axis)?,
        *config.rc_expo.get(axis)?,
        *config.super_rate.get(axis)?,
        *config.rate_limit.get(axis)?,
    );

    let raw = match config.rates_type {
        RatesType::Betaflight => apply_betaflight_rate(rc_rate, rc_expo, super_rate),
        RatesType::RaceFlight => apply_raceflight_rate(rc_rate, rc_expo, super_rate),
        RatesType::Kiss => apply_kiss_rate(rc_rate, rc_expo, super_rate),
        RatesType::Actual => apply_actual_rate(rc_rate, super_rate),
        RatesType::Quick => apply_quick_rate(rc_rate, super_rate),
    };

    // `f64::clamp` panics when min > max — guard `rate_limit` explicitly rather than trust a
    // parsed header value to be non-negative before it becomes the clamp's upper bound.
    (raw.is_finite() && rate_limit.is_finite() && rate_limit >= 0.0)
        .then(|| raw.clamp(-rate_limit, rate_limit).abs())
}

/// Mirrors `applyBetaflightRates()`. Expo shapes only the mid-curve — at full stick,
/// `rcCommandf * rcCommandfAbs^3 * expof + rcCommandf * (1 - expof)` collapses to `rcCommandf`
/// regardless of `expof`, so `rc_expo` has no effect on the returned value.
fn apply_betaflight_rate(rc_rate_raw: f64, rc_expo: f64, super_rate: f64) -> f64 {
    let rc_commandf = if rc_expo != 0.0 {
        let expof = rc_expo / 100.0;
        FULL_STICK * FULL_STICK.powi(3) * expof + FULL_STICK * (1.0 - expof)
    } else {
        FULL_STICK
    };

    let mut rc_rate = rc_rate_raw / 100.0;
    if rc_rate > RATE_CURVE_SUPER_EXPO_THRESHOLD {
        rc_rate += RATE_CURVE_RC_RATE_INCREMENTAL * (rc_rate - RATE_CURVE_SUPER_EXPO_THRESHOLD);
    }
    let mut angle_rate = 200.0 * rc_rate * rc_commandf;

    if super_rate != 0.0 {
        let denom =
            (1.0 - (FULL_STICK * (super_rate / 100.0))).clamp(RATE_CURVE_SUPER_FACTOR_MIN, 1.0);
        angle_rate /= denom;
    }

    angle_rate
}

/// Mirrors `applyRaceFlightRates()`. Same full-stick expo cancellation as Betaflight: at
/// `rcCommandf = 1.0`, `(1 + 0.01 * rcExpo * (rcCommandf^2 - 1)) * rcCommandf` collapses to 1.0.
fn apply_raceflight_rate(rc_rate_raw: f64, rc_expo: f64, super_rate: f64) -> f64 {
    let rc_commandf = (1.0 + 0.01 * rc_expo * (FULL_STICK * FULL_STICK - 1.0)) * FULL_STICK;
    let angle_rate = 10.0 * rc_rate_raw * rc_commandf;
    angle_rate * (1.0 + FULL_STICK * super_rate * 0.01)
}

/// Mirrors `applyKissRates()`, including its internal clamp to
/// `RATE_CURVE_SETPOINT_LIMIT_DPS` before the caller's own `rate_limit` clamp.
fn apply_kiss_rate(rc_rate_raw: f64, rc_expo: f64, super_rate: f64) -> f64 {
    let rc_curvef = rc_expo / 100.0;
    let denom = (1.0 - (FULL_STICK * (super_rate / 100.0))).clamp(RATE_CURVE_SUPER_FACTOR_MIN, 1.0);
    let kiss_rpy_use_rates = 1.0 / denom;
    let kiss_rc_commandf =
        (FULL_STICK.powi(3) * rc_curvef + FULL_STICK * (1.0 - rc_curvef)) * (rc_rate_raw / 1000.0);
    (2000.0 * kiss_rpy_use_rates * kiss_rc_commandf).clamp(
        -RATE_CURVE_SETPOINT_LIMIT_DPS,
        RATE_CURVE_SETPOINT_LIMIT_DPS,
    )
}

/// Mirrors `applyActualRates()`. Same full-stick expo cancellation as the others above:
/// `rcCommandfAbs * (rcCommandf^5 * expof + rcCommandf * (1 - expof))` collapses to
/// `rcCommandfAbs` at full stick regardless of `expof`, so `rc_expo` is intentionally not
/// threaded through here.
fn apply_actual_rate(rc_rate_raw: f64, super_rate: f64) -> f64 {
    let expof_term = FULL_STICK;
    let center_sensitivity = rc_rate_raw * 10.0;
    let stick_movement = (super_rate * 10.0 - center_sensitivity).max(0.0);
    FULL_STICK * center_sensitivity + stick_movement * expof_term
}

/// Mirrors `applyQuickRates()`. Both `quickRatesRcExpo` branches converge to the same value at
/// full stick (the curve term collapses to 1.0 either way), so that header toggle — not
/// otherwise exposed here — doesn't affect the result.
fn apply_quick_rate(rc_rate_raw: f64, super_rate: f64) -> f64 {
    let rc_rate = rc_rate_raw * 2.0;
    let max_dps = (super_rate * 10.0).max(rc_rate);
    let super_factor_config = (max_dps / rc_rate - 1.0) / (max_dps / rc_rate);
    let denom = (1.0 - super_factor_config).clamp(RATE_CURVE_SUPER_FACTOR_MIN, 1.0);
    let super_factor = 1.0 / denom;
    (rc_rate * super_factor).clamp(
        -RATE_CURVE_SETPOINT_LIMIT_DPS,
        RATE_CURVE_SETPOINT_LIMIT_DPS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn betaflight_default_rates_match_known_community_value() {
        // rcRate=100 (1.0), Super Rate=70, Expo=0 — a widely-known Betaflight default profile,
        // documented to yield ~666-670 deg/s max rate.
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "\"100,100,100\""),
            ("rc_expo", "\"0,0,0\""),
            ("rates", "\"70,70,70\""),
            ("rate_limits", "\"1998,1998,1998\""),
        ]))
        .unwrap();
        assert_eq!(config.rates_type, RatesType::Betaflight);
        let max_rate = configured_max_rate(&config, 0).unwrap();
        assert!(
            (max_rate - 666.7).abs() < 1.0,
            "expected ~666.7 deg/s, got {max_rate}"
        );
    }

    #[test]
    fn rates_type_defaults_to_betaflight_when_header_absent() {
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,100,100"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
        ]))
        .unwrap();
        assert_eq!(config.rates_type, RatesType::Betaflight);
    }

    #[test]
    fn missing_rc_rates_returns_none() {
        let config =
            parse_rate_curve_config(&headers(&[("rc_expo", "0,0,0"), ("rates", "70,70,70")]));
        assert!(config.is_none());
    }

    #[test]
    fn rate_limit_defaults_when_header_absent() {
        // EmuFlight never logs "rate_limits" — must default, not silently produce an
        // unbounded/unclamped result.
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,100,100"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
        ]))
        .unwrap();
        assert_eq!(config.rate_limit, [RATE_CURVE_DEFAULT_RATE_LIMIT_DPS; 3]);
    }

    #[test]
    fn rate_limit_clamps_a_high_configured_rate() {
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "\"200,200,200\""),
            ("rc_expo", "\"0,0,0\""),
            ("rates", "\"100,100,100\""),
            ("rate_limits", "\"500,500,500\""),
        ]))
        .unwrap();
        let max_rate = configured_max_rate(&config, 0).unwrap();
        assert_eq!(max_rate, 500.0);
    }

    #[test]
    fn quick_rates_zero_rc_rate_is_none_not_nan() {
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "\"0,0,0\""),
            ("rc_expo", "\"0,0,0\""),
            ("rates", "\"70,70,70\""),
            ("rates_type", "4"),
        ]))
        .unwrap();
        assert_eq!(config.rates_type, RatesType::Quick);
        assert_eq!(configured_max_rate(&config, 0), None);
    }

    #[test]
    fn out_of_range_axis_is_none() {
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,100,100"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
        ]))
        .unwrap();
        assert_eq!(configured_max_rate(&config, 3), None);
    }

    #[test]
    fn all_five_rates_types_parse_from_header_value() {
        for (value, expected) in [
            ("0", RatesType::Betaflight),
            ("1", RatesType::RaceFlight),
            ("2", RatesType::Kiss),
            ("3", RatesType::Actual),
            ("4", RatesType::Quick),
        ] {
            let config = parse_rate_curve_config(&headers(&[
                ("rc_rates", "100,100,100"),
                ("rc_expo", "0,0,0"),
                ("rates", "70,70,70"),
                ("rates_type", value),
            ]))
            .unwrap();
            assert_eq!(config.rates_type, expected);
        }
    }

    #[test]
    fn unrecognized_rates_type_value_falls_back_to_betaflight() {
        // Neither firmware currently defines 5+, so treat it the same as an absent header
        // rather than silently misreporting a rate curve that was never actually applied.
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,100,100"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
            ("rates_type", "9"),
        ]))
        .unwrap();
        assert_eq!(config.rates_type, RatesType::Betaflight);
    }

    #[test]
    fn malformed_middle_token_does_not_misalign_remaining_values() {
        // A `filter_map`-style parse would drop "bad" and shift 90/80 left by one, silently
        // assigning yaw's value to pitch. Must reject the whole triple instead.
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,bad,90,80"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
        ]));
        assert!(config.is_none());
    }

    #[test]
    fn present_but_malformed_rate_limits_is_rejected_not_defaulted() {
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "100,100,100"),
            ("rc_expo", "0,0,0"),
            ("rates", "70,70,70"),
            ("rate_limits", "abc,def,ghi"),
        ]));
        assert!(config.is_none());
    }

    #[test]
    fn negative_rate_limit_returns_none_instead_of_panicking() {
        // f64::clamp panics when min > max; a negative rate_limit would make
        // clamp(-rate_limit, rate_limit) exactly that. Must guard before calling it.
        let config = parse_rate_curve_config(&headers(&[
            ("rc_rates", "\"100,100,100\""),
            ("rc_expo", "\"0,0,0\""),
            ("rates", "\"70,70,70\""),
            ("rate_limits", "\"-100,-100,-100\""),
        ]))
        .unwrap();
        assert_eq!(configured_max_rate(&config, 0), None);
    }
}
