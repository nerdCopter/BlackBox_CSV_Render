// src/data_analysis/optimal_p_estimation.rs
//
// Optimal P Estimation Module
//
// Provides physics-aware P gain recommendations based on:
// - Frame-class-specific Td (time to 50%) targets
// - High-frequency noise level analysis
// - Response consistency metrics
//
// Implements theory from BrianWhite (PIDtoolbox) that optimal response timing
// is aircraft-specific, determined by power-to-rotational-inertia ratio.

use crate::constants::*;
use std::f64;

/// Frame class for Td target selection (prop size in inches)
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameClass {
    OneInch,
    TwoInch,
    ThreeInch,
    FourInch,
    FiveInch,
    SixInch,
    SevenInch,
    EightInch,
    NineInch,
    TenInch,
    ElevenInch,
    TwelveInch,
    ThirteenInch,
}

impl FrameClass {
    /// Get Td target and tolerance for this frame class
    pub fn td_target(&self) -> (f64, f64) {
        match self {
            FrameClass::OneInch => (TD_TARGET_1INCH, TD_TARGET_1INCH_TOLERANCE),
            FrameClass::TwoInch => (TD_TARGET_2INCH, TD_TARGET_2INCH_TOLERANCE),
            FrameClass::ThreeInch => (TD_TARGET_3INCH, TD_TARGET_3INCH_TOLERANCE),
            FrameClass::FourInch => (TD_TARGET_4INCH, TD_TARGET_4INCH_TOLERANCE),
            FrameClass::FiveInch => (TD_TARGET_5INCH, TD_TARGET_5INCH_TOLERANCE),
            FrameClass::SixInch => (TD_TARGET_6INCH, TD_TARGET_6INCH_TOLERANCE),
            FrameClass::SevenInch => (TD_TARGET_7INCH, TD_TARGET_7INCH_TOLERANCE),
            FrameClass::EightInch => (TD_TARGET_8INCH, TD_TARGET_8INCH_TOLERANCE),
            FrameClass::NineInch => (TD_TARGET_9INCH, TD_TARGET_9INCH_TOLERANCE),
            FrameClass::TenInch => (TD_TARGET_10INCH, TD_TARGET_10INCH_TOLERANCE),
            FrameClass::ElevenInch => (TD_TARGET_11INCH, TD_TARGET_11INCH_TOLERANCE),
            FrameClass::TwelveInch => (TD_TARGET_12INCH, TD_TARGET_12INCH_TOLERANCE),
            FrameClass::ThirteenInch => (TD_TARGET_13INCH, TD_TARGET_13INCH_TOLERANCE),
        }
    }

    /// Get name for display
    pub fn name(&self) -> &str {
        match self {
            FrameClass::OneInch => "1\"",
            FrameClass::TwoInch => "2\"",
            FrameClass::ThreeInch => "3\"",
            FrameClass::FourInch => "4\"",
            FrameClass::FiveInch => "5\"",
            FrameClass::SixInch => "6\"",
            FrameClass::SevenInch => "7\"",
            FrameClass::EightInch => "8\"",
            FrameClass::NineInch => "9\"",
            FrameClass::TenInch => "10\"",
            FrameClass::ElevenInch => "11\"",
            FrameClass::TwelveInch => "12\"",
            FrameClass::ThirteenInch => "13\"",
        }
    }
}

/// Noise level classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseLevel {
    Low,      // < 10% HF energy
    Moderate, // 10-15% HF energy
    High,     // > 15% HF energy
    Unknown,  // Cannot determine
}

impl NoiseLevel {
    pub fn name(&self) -> &str {
        match self {
            NoiseLevel::Low => "LOW",
            NoiseLevel::Moderate => "MODERATE",
            NoiseLevel::High => "HIGH",
            NoiseLevel::Unknown => "UNKNOWN",
        }
    }

    pub fn assessment(&self) -> &str {
        match self {
            NoiseLevel::Low => "Noise levels are acceptable, P has headroom",
            NoiseLevel::Moderate => "Approaching noise limits",
            NoiseLevel::High => "At or exceeding recommended noise limits",
            NoiseLevel::Unknown => "Cannot assess noise levels (D-term data unavailable)",
        }
    }
}

/// Td deviation classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TdDeviation {
    SignificantlySlower, // > 30% slower than target
    ModeratelySlower,    // 15-30% slower
    WithinTarget,        // ±15% of target
    SignificantlyFaster, // < -15% faster than target
}

impl TdDeviation {
    pub fn name(&self) -> &str {
        match self {
            TdDeviation::SignificantlySlower => "SIGNIFICANTLY SLOWER",
            TdDeviation::ModeratelySlower => "MODERATELY SLOWER",
            TdDeviation::WithinTarget => "WITHIN TARGET",
            TdDeviation::SignificantlyFaster => "FASTER",
        }
    }
}

/// P optimization recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum PRecommendation {
    Optimal {
        reasoning: String,
    },
    Increase {
        conservative_p: u32,
        moderate_p: u32,
        reasoning: String,
    },
    Decrease {
        recommended_p: u32,
        reasoning: String,
    },
    Investigate {
        issue: String,
    },
}

/// Statistics for Td measurements across multiple step responses
#[derive(Debug, Clone)]
pub struct TdStatistics {
    pub mean_ms: f64,
    pub std_dev_ms: f64,
    pub coefficient_of_variation: f64,
    pub num_samples: usize,
    pub consistency: f64, // Fraction of samples within ±1 std dev
}

impl TdStatistics {
    /// Calculate statistics from array of Td values (in milliseconds)
    pub fn from_samples(td_samples_ms: &[f64]) -> Option<Self> {
        if td_samples_ms.is_empty() {
            return None;
        }

        let n = td_samples_ms.len() as f64;
        let mean = td_samples_ms.iter().sum::<f64>() / n;

        if mean == 0.0 {
            return None;
        }

        let variance = td_samples_ms
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        // Calculate consistency: fraction within ±1 std dev
        let within_range = td_samples_ms
            .iter()
            .filter(|&&x| (x - mean).abs() <= std_dev)
            .count();
        let consistency = within_range as f64 / n;

        Some(TdStatistics {
            mean_ms: mean,
            std_dev_ms: std_dev,
            coefficient_of_variation,
            num_samples: td_samples_ms.len(),
            consistency,
        })
    }

    /// Check if measurements are consistent enough for reliable analysis
    pub fn is_consistent(&self) -> bool {
        self.consistency >= TD_CONSISTENCY_MIN_THRESHOLD
            && self.coefficient_of_variation <= TD_COEFFICIENT_OF_VARIATION_MAX
    }
}

/// Complete optimal P analysis result
#[derive(Debug, Clone)]
pub struct OptimalPAnalysis {
    pub frame_class: FrameClass,
    pub current_p: u32,
    pub td_stats: TdStatistics,
    pub td_deviation: TdDeviation,
    pub td_deviation_percent: f64,
    pub noise_level: NoiseLevel,
    pub hf_energy_percent: Option<f64>,
    pub recommendation: PRecommendation,
}

impl OptimalPAnalysis {
    /// Analyze optimal P for a given axis
    ///
    /// # Arguments
    /// * `td_samples_ms` - Array of Td measurements from multiple step responses (milliseconds)
    /// * `current_p` - Current P gain
    /// * `frame_class` - Aircraft frame class
    /// * `hf_energy_ratio` - Optional: ratio of D-term energy above DTERM_HF_CUTOFF_HZ (0.0-1.0)
    pub fn analyze(
        td_samples_ms: &[f64],
        current_p: u32,
        frame_class: FrameClass,
        hf_energy_ratio: Option<f64>,
    ) -> Option<Self> {
        // Calculate Td statistics
        let td_stats = TdStatistics::from_samples(td_samples_ms)?;

        // Get target Td for frame class
        let (td_target_ms, _td_tolerance_ms) = frame_class.td_target();

        // Calculate deviation from target
        let td_deviation_percent = ((td_stats.mean_ms - td_target_ms) / td_target_ms) * 100.0;

        // Classify deviation
        let td_deviation = if td_deviation_percent > 30.0 {
            TdDeviation::SignificantlySlower
        } else if td_deviation_percent > 15.0 {
            TdDeviation::ModeratelySlower
        } else if td_deviation_percent < -15.0 {
            TdDeviation::SignificantlyFaster
        } else {
            TdDeviation::WithinTarget
        };

        // Classify noise level
        let (noise_level, hf_energy_percent) = if let Some(hf_ratio) = hf_energy_ratio {
            let hf_percent = hf_ratio * 100.0;
            let level = if hf_ratio < DTERM_HF_ENERGY_MODERATE {
                NoiseLevel::Low
            } else if hf_ratio < DTERM_HF_ENERGY_THRESHOLD {
                NoiseLevel::Moderate
            } else {
                NoiseLevel::High
            };
            (level, Some(hf_percent))
        } else {
            (NoiseLevel::Unknown, None)
        };

        // Generate recommendation based on Td deviation and noise level
        let recommendation = Self::generate_recommendation(
            current_p,
            &td_deviation,
            td_deviation_percent,
            &noise_level,
            &td_stats,
        );

        Some(OptimalPAnalysis {
            frame_class,
            current_p,
            td_stats,
            td_deviation,
            td_deviation_percent,
            noise_level,
            hf_energy_percent,
            recommendation,
        })
    }

    /// Generate P recommendation based on analysis
    fn generate_recommendation(
        current_p: u32,
        td_deviation: &TdDeviation,
        td_deviation_percent: f64,
        noise_level: &NoiseLevel,
        td_stats: &TdStatistics,
    ) -> PRecommendation {
        match (td_deviation, noise_level) {
            // Case 1: Td significantly slower + low noise = clear headroom to increase P
            (TdDeviation::SignificantlySlower, NoiseLevel::Low) => {
                let conservative = ((current_p as f64) * P_HEADROOM_MODERATE_MULTIPLIER) as u32;
                let moderate = ((current_p as f64) * P_HEADROOM_AGGRESSIVE_MULTIPLIER) as u32;
                PRecommendation::Increase {
                    conservative_p: conservative,
                    moderate_p: moderate,
                    reasoning: format!(
                        "Response is {:.1}% slower than target with low noise levels. \
                         P can be increased significantly for faster response.",
                        td_deviation_percent
                    ),
                }
            }

            // Case 2: Td moderately slower + low/moderate noise = modest headroom
            (TdDeviation::ModeratelySlower, NoiseLevel::Low | NoiseLevel::Moderate) => {
                let conservative = ((current_p as f64) * P_HEADROOM_CONSERVATIVE_MULTIPLIER) as u32;
                let moderate = ((current_p as f64) * P_HEADROOM_MODERATE_MULTIPLIER) as u32;
                PRecommendation::Increase {
                    conservative_p: conservative,
                    moderate_p: moderate,
                    reasoning: format!(
                        "Response is {:.1}% slower than target. Modest P increase recommended.",
                        td_deviation_percent
                    ),
                }
            }

            // Case 2b: Td moderately slower + high noise = investigate
            (TdDeviation::ModeratelySlower, NoiseLevel::High) => PRecommendation::Investigate {
                issue: format!(
                    "Response is {:.1}% slower than target despite high noise levels. \
                     This suggests mechanical issues (damaged props, loose hardware, motor problems) \
                     or incorrect frame class. Inspect aircraft.",
                    td_deviation_percent
                ),
            },

            // Case 3: Td within target + low noise = slight headroom available
            (TdDeviation::WithinTarget, NoiseLevel::Low) => {
                let conservative = ((current_p as f64) * P_HEADROOM_CONSERVATIVE_MULTIPLIER) as u32;
                let moderate = ((current_p as f64) * P_HEADROOM_MODERATE_MULTIPLIER) as u32;
                if conservative > current_p {
                    PRecommendation::Increase {
                        conservative_p: conservative,
                        moderate_p: moderate,
                        reasoning: "Response time is in target range with low noise. \
                             Minor P increase possible if seeking faster response.".to_string(),
                    }
                } else {
                    PRecommendation::Optimal {
                        reasoning: format!(
                            "Response time is in target range ({:.1}ms) with low noise levels. \
                             Current P ({}) appears optimal.",
                            td_stats.mean_ms, current_p
                        ),
                    }
                }
            }

            // Case 4: Td within target + moderate noise = likely optimal
            (TdDeviation::WithinTarget, NoiseLevel::Moderate) => PRecommendation::Optimal {
                reasoning: format!(
                    "Response time is in target range ({:.1}ms) with moderate noise levels. \
                     Current P ({}) appears optimal for this aircraft.",
                    td_stats.mean_ms, current_p
                ),
            },

            // Case 5: Td within target + high noise = optimal but monitor
            (TdDeviation::WithinTarget, NoiseLevel::High) => PRecommendation::Optimal {
                reasoning: format!(
                    "Response time is in target range but noise levels are high. \
                     Current P ({}) is at or near physical limits. Monitor motor temperatures.",
                    current_p
                ),
            },

            // Case 6: Td faster than target + high noise = at limit, consider reduction
            (TdDeviation::SignificantlyFaster, NoiseLevel::High) => {
                let recommended = ((current_p as f64) * P_REDUCTION_MODERATE_MULTIPLIER) as u32;
                PRecommendation::Decrease {
                    recommended_p: recommended,
                    reasoning: format!(
                        "Response is {:.1}% faster than target with high noise levels. \
                         P may be exceeding physical limits. Consider reduction if motors overheat.",
                        td_deviation_percent
                    ),
                }
            }

            // Case 7: Td faster than target + moderate noise = at optimal limit
            (TdDeviation::SignificantlyFaster, NoiseLevel::Moderate) => PRecommendation::Optimal {
                reasoning: format!(
                    "Response is {:.1}% faster than target with moderate noise. \
                     Current P ({}) is at or near optimal limit for this aircraft.",
                    td_deviation_percent, current_p
                ),
            },

            // Case 8: Td faster than target + low noise = unusual, may indicate issue
            (TdDeviation::SignificantlyFaster, NoiseLevel::Low) => PRecommendation::Investigate {
                issue: format!(
                    "Response is {:.1}% faster than typical for frame class, \
                     but noise is low. This may indicate incorrect frame class selection \
                     or unusual power-to-inertia ratio. Verify frame class and check build specs.",
                    td_deviation_percent
                ),
            },

            // Case 9: Td significantly slower + moderate/high noise = investigate
            (TdDeviation::SignificantlySlower, NoiseLevel::Moderate | NoiseLevel::High) => {
                PRecommendation::Investigate {
                    issue: format!(
                        "Response is {:.1}% slower than target despite moderate/high noise. \
                         This suggests mechanical issues (damaged props, loose hardware, \
                         motor problems) or incorrect frame class. Inspect aircraft.",
                        td_deviation_percent
                    ),
                }
            }

            // Case 10: Unknown noise level - provide Td-only recommendation
            (_, NoiseLevel::Unknown) => match td_deviation {
                TdDeviation::SignificantlySlower | TdDeviation::ModeratelySlower => {
                    let conservative =
                        ((current_p as f64) * P_HEADROOM_CONSERVATIVE_MULTIPLIER) as u32;
                    let moderate = ((current_p as f64) * P_HEADROOM_MODERATE_MULTIPLIER) as u32;
                    PRecommendation::Increase {
                        conservative_p: conservative,
                        moderate_p: moderate,
                        reasoning: format!(
                            "Response is {:.1}% slower than target. P increase recommended, \
                             but monitor motor temperatures (D-term data unavailable for noise analysis).",
                            td_deviation_percent
                        ),
                    }
                }
                TdDeviation::WithinTarget => PRecommendation::Optimal {
                    reasoning: format!(
                        "Response time ({:.1}ms) is in target range. Current P ({}) appears appropriate, \
                         but monitor motor temperatures (D-term data unavailable for noise analysis).",
                        td_stats.mean_ms, current_p
                    ),
                },
                TdDeviation::SignificantlyFaster => PRecommendation::Optimal {
                    reasoning: format!(
                        "Response is {:.1}% faster than target. Current P ({}) may be at limits, \
                         monitor motor temperatures (D-term data unavailable for noise analysis).",
                        td_deviation_percent, current_p
                    ),
                },
            },
        }
    }

    /// Format analysis as human-readable console output
    pub fn format_console_output(&self, axis_name: &str) -> String {
        let (td_target, td_tolerance) = self.frame_class.td_target();

        let mut output = String::new();
        output.push_str(&format!("\n{}\n", "=".repeat(70)));
        output.push_str(&format!("OPTIMAL P ESTIMATION ({} Axis)\n", axis_name));
        output.push_str(&format!("{}\n", "=".repeat(70)));

        // Current configuration
        output.push_str("Current Configuration:\n");
        output.push_str(&format!("  P Gain: {}\n", self.current_p));

        // Step response analysis
        output.push_str("\nStep Response Analysis:\n");
        output.push_str(&format!(
            "  Time to 50% (Td): {:.1}ms (± {:.1}ms, CV: {:.1}%)\n",
            self.td_stats.mean_ms,
            self.td_stats.std_dev_ms,
            self.td_stats.coefficient_of_variation * 100.0
        ));
        output.push_str(&format!(
            "  Response Consistency: {:.0}% ({}/{} valid responses)\n",
            self.td_stats.consistency * 100.0,
            (self.td_stats.consistency * self.td_stats.num_samples as f64) as usize,
            self.td_stats.num_samples
        ));

        if !self.td_stats.is_consistent() {
            output.push_str("  ⚠ WARNING: Low consistency - results may be unreliable\n");
        }

        // Frame class comparison
        output.push_str(&format!(
            "\nFrame Class: {} (Target Td: {:.1}ms ± {:.1}ms)\n",
            self.frame_class.name(),
            td_target,
            td_tolerance
        ));
        output.push_str(&format!(
            "  Td Deviation: {:.1}% ({})\n",
            self.td_deviation_percent,
            self.td_deviation.name()
        ));

        let assessment = if self.td_deviation_percent.abs() <= 15.0 {
            "Response is appropriately fast for frame class"
        } else if self.td_deviation_percent > 0.0 {
            "Response is slower than typical for frame class"
        } else {
            "Response is faster than typical for frame class"
        };
        output.push_str(&format!("  Assessment: {}\n", assessment));

        // Noise analysis
        output.push_str("\nNoise Analysis:\n");
        if let Some(hf_percent) = self.hf_energy_percent {
            output.push_str(&format!(
                "  D-term HF Energy (>{}Hz): {:.1}% of total\n",
                DTERM_HF_CUTOFF_HZ, hf_percent
            ));
        }
        output.push_str(&format!("  Noise Level: {}\n", self.noise_level.name()));
        output.push_str(&format!(
            "  Assessment: {}\n",
            self.noise_level.assessment()
        ));

        // Physical limit indicators
        output.push_str("\nPhysical Limit Indicators:\n");
        let response_indicator = match self.td_deviation {
            TdDeviation::WithinTarget => "GOOD (within target range)",
            TdDeviation::ModeratelySlower => "IMPROVABLE (slower than target)",
            TdDeviation::SignificantlySlower => "SUBOPTIMAL (significantly slower)",
            TdDeviation::SignificantlyFaster => "VERY FAST (faster than typical)",
        };
        output.push_str(&format!("  ├─ Response speed: {}\n", response_indicator));

        let noise_indicator = match self.noise_level {
            NoiseLevel::Low => "GOOD (low noise)",
            NoiseLevel::Moderate => "ACCEPTABLE (moderate noise)",
            NoiseLevel::High => "AT LIMIT (high noise)",
            NoiseLevel::Unknown => "UNKNOWN (no D-term data)",
        };
        output.push_str(&format!("  ├─ Noise level: {}\n", noise_indicator));

        let consistency_indicator = if self.td_stats.is_consistent() {
            "GOOD (low variation)"
        } else {
            "POOR (high variation)"
        };
        output.push_str(&format!("  └─ Consistency: {}\n", consistency_indicator));

        // Recommendation
        output.push_str(&format!("\n{}\n", "=".repeat(70)));
        output.push_str("P OPTIMIZATION RECOMMENDATION\n");
        output.push_str(&format!("{}\n", "=".repeat(70)));

        match &self.recommendation {
            PRecommendation::Optimal { reasoning } => {
                output.push_str(&format!(
                    "Current P ({}) appears OPTIMAL for this aircraft.\n\n",
                    self.current_p
                ));
                output.push_str(&format!("{}\n", reasoning));
            }
            PRecommendation::Increase {
                conservative_p,
                moderate_p,
                reasoning,
            } => {
                output.push_str("P increase recommended:\n\n");
                let conservative_pct = if self.current_p == 0 {
                    "N/A".to_string()
                } else {
                    format!(
                        "+{:.0}%",
                        (((*conservative_p as f64) / (self.current_p as f64)) - 1.0) * 100.0
                    )
                };
                output.push_str(&format!(
                    "  • Conservative: P = {} ({})\n",
                    conservative_p, conservative_pct
                ));
                let moderate_pct = if self.current_p == 0 {
                    "N/A".to_string()
                } else {
                    format!(
                        "+{:.0}%",
                        (((*moderate_p as f64) / (self.current_p as f64)) - 1.0) * 100.0
                    )
                };
                output.push_str(&format!(
                    "  • Moderate: P = {} ({})\n\n",
                    moderate_p, moderate_pct
                ));
                output.push_str(&format!("{}\n\n", reasoning));
                output.push_str("⚠ Always test incrementally and monitor motor temperatures.\n");
            }
            PRecommendation::Decrease {
                recommended_p,
                reasoning,
            } => {
                output.push_str("P reduction recommended:\n\n");
                let decrease_pct = if self.current_p == 0 {
                    "N/A".to_string()
                } else {
                    format!(
                        "{:.0}%",
                        (((*recommended_p as f64) / (self.current_p as f64)) - 1.0) * 100.0
                    )
                };
                output.push_str(&format!(
                    "  • Recommended: P = {} ({})\n\n",
                    recommended_p, decrease_pct
                ));
                output.push_str(&format!("{}\n", reasoning));
            }
            PRecommendation::Investigate { issue } => {
                output.push_str("⚠ INVESTIGATION RECOMMENDED\n\n");
                output.push_str(&format!("{}\n", issue));
            }
        }

        output.push_str(&format!("{}\n", "=".repeat(70)));
        output
    }
}
