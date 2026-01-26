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
    FourteenInch,
    FifteenInch,
}

impl FrameClass {
    /// Get Td target and tolerance for this frame class
    pub fn td_target(&self) -> (f64, f64) {
        // Convert to 1-based frame size (inches) for the helper method
        let frame_size = self.array_index() + 1;
        // Fail-fast if TdTargetSpec is missing (invariant violation)
        let spec = crate::constants::TdTargetSpec::for_frame_inches(frame_size)
            .expect("TdTargetSpec missing for valid FrameClass - this should never happen");
        (spec.target_ms, spec.tolerance_ms)
    }

    /// Get array index for this frame class (0-14)
    pub fn array_index(&self) -> usize {
        match self {
            FrameClass::OneInch => 0,
            FrameClass::TwoInch => 1,
            FrameClass::ThreeInch => 2,
            FrameClass::FourInch => 3,
            FrameClass::FiveInch => 4,
            FrameClass::SixInch => 5,
            FrameClass::SevenInch => 6,
            FrameClass::EightInch => 7,
            FrameClass::NineInch => 8,
            FrameClass::TenInch => 9,
            FrameClass::ElevenInch => 10,
            FrameClass::TwelveInch => 11,
            FrameClass::ThirteenInch => 12,
            FrameClass::FourteenInch => 13,
            FrameClass::FifteenInch => 14,
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
            FrameClass::FourteenInch => "14\"",
            FrameClass::FifteenInch => "15\"",
        }
    }

    /// Create a FrameClass from prop size in inches (1-15)
    pub fn from_inches(size: u8) -> Option<Self> {
        match size {
            1 => Some(FrameClass::OneInch),
            2 => Some(FrameClass::TwoInch),
            3 => Some(FrameClass::ThreeInch),
            4 => Some(FrameClass::FourInch),
            5 => Some(FrameClass::FiveInch),
            6 => Some(FrameClass::SixInch),
            7 => Some(FrameClass::SevenInch),
            8 => Some(FrameClass::EightInch),
            9 => Some(FrameClass::NineInch),
            10 => Some(FrameClass::TenInch),
            11 => Some(FrameClass::ElevenInch),
            12 => Some(FrameClass::TwelveInch),
            13 => Some(FrameClass::ThirteenInch),
            14 => Some(FrameClass::FourteenInch),
            15 => Some(FrameClass::FifteenInch),
            _ => None,
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

        // Use epsilon-based comparison to avoid division by near-zero values
        if mean.abs() <= TD_MEAN_EPSILON {
            return None;
        }

        // Calculate sample variance with Bessel's correction (divide by n-1)
        // For small samples, set std_dev to 0.0 to avoid division by zero
        let (std_dev, coefficient_of_variation) = if td_samples_ms.len() < TD_SAMPLES_MIN_FOR_STDDEV
        {
            (0.0, 0.0)
        } else {
            let sum_sq_dev = td_samples_ms
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>();
            let variance = sum_sq_dev / (n - 1.0);
            let std_dev = variance.sqrt();
            let coefficient_of_variation = std_dev / mean;
            (std_dev, coefficient_of_variation)
        };

        // Calculate consistency: fraction within ±1 std dev
        // When std_dev == 0.0 (identical samples), consistency is perfect (1.0)
        // Otherwise, tolerance = std_dev and calculate fraction within range
        let consistency = if std_dev == 0.0 {
            // All samples identical → perfect consistency
            1.0
        } else {
            let tolerance = std_dev;
            let within_range = td_samples_ms
                .iter()
                .filter(|&&x| (x - mean).abs() <= tolerance)
                .count();
            within_range as f64 / n
        };

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
        // Need at least 2 samples for meaningful consistency check
        self.num_samples >= TD_SAMPLES_MIN_FOR_STDDEV
            && self.consistency >= TD_CONSISTENCY_MIN_THRESHOLD
            && self.coefficient_of_variation <= TD_COEFFICIENT_OF_VARIATION_MAX
    }
}

/// Complete optimal P analysis result
#[derive(Debug, Clone)]
pub struct OptimalPAnalysis {
    pub frame_class: FrameClass,
    pub current_p: u32,
    pub current_d: Option<u32>,
    pub recommended_pd_conservative: Option<f64>,
    pub td_stats: TdStatistics,
    pub td_deviation: TdDeviation,
    pub td_deviation_percent: f64,
    pub noise_level: NoiseLevel,
    pub recommendation: PRecommendation,
}

impl OptimalPAnalysis {
    /// Analyze optimal P for a given axis
    ///
    /// # Arguments
    /// * `td_samples_ms` - Array of Td measurements from multiple step responses (milliseconds)
    /// * `current_p` - Current P gain
    /// * `current_d` - Current D gain (optional)
    /// * `frame_class` - Aircraft frame class
    /// * `hf_energy_ratio` - Optional: ratio of D-term energy above DTERM_HF_CUTOFF_HZ (0.0-1.0)
    /// * `recommended_pd_conservative` - Optional: recommended P:D ratio from step response (conservative)
    pub fn analyze(
        td_samples_ms: &[f64],
        current_p: u32,
        current_d: Option<u32>,
        frame_class: FrameClass,
        hf_energy_ratio: Option<f64>,
        recommended_pd_conservative: Option<f64>,
        physics_td_target_ms: Option<(f64, f64)>, // Optional (td_target, tolerance) from physics
    ) -> Option<Self> {
        // Calculate Td statistics
        let td_stats = TdStatistics::from_samples(td_samples_ms)?;

        // Get target Td - use physics-based if available, otherwise frame class
        let (td_target_ms, _td_tolerance_ms) =
            if let Some((phys_target, _phys_tol)) = physics_td_target_ms {
                (phys_target, _phys_tol)
            } else {
                frame_class.td_target()
            };

        // Calculate deviation from target
        let td_deviation_percent = ((td_stats.mean_ms - td_target_ms) / td_target_ms) * 100.0;

        // Classify deviation
        let td_deviation = if td_deviation_percent > TD_DEVIATION_SIGNIFICANTLY_SLOWER_THRESHOLD {
            TdDeviation::SignificantlySlower
        } else if td_deviation_percent > TD_DEVIATION_MODERATELY_SLOWER_THRESHOLD {
            TdDeviation::ModeratelySlower
        } else if td_deviation_percent < TD_DEVIATION_SIGNIFICANTLY_FASTER_THRESHOLD {
            TdDeviation::SignificantlyFaster
        } else {
            TdDeviation::WithinTarget
        };

        // Classify noise level
        let noise_level = if let Some(hf_ratio) = hf_energy_ratio {
            if hf_ratio < DTERM_HF_ENERGY_MODERATE {
                NoiseLevel::Low
            } else if hf_ratio < DTERM_HF_ENERGY_THRESHOLD {
                NoiseLevel::Moderate
            } else {
                NoiseLevel::High
            }
        } else {
            NoiseLevel::Unknown
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
            current_d,
            recommended_pd_conservative,
            td_stats,
            td_deviation,
            td_deviation_percent,
            noise_level,
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
                PRecommendation::Increase {
                    conservative_p: conservative,
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
                PRecommendation::Increase {
                    conservative_p: conservative,
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
                if conservative > current_p {
                    PRecommendation::Increase {
                        conservative_p: conservative,
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

            // Case 8: Td faster than target + low noise = headroom available for P increase
            // This is GOOD - faster response + low noise means the prop size might be 
            // slightly smaller than specified, or the build is exceptionally clean.
            // Either way, there's headroom to push P higher if desired.
            (TdDeviation::SignificantlyFaster, NoiseLevel::Low) => {
                let conservative = ((current_p as f64) * P_HEADROOM_CONSERVATIVE_MULTIPLIER) as u32;
                if conservative > current_p {
                    PRecommendation::Increase {
                        conservative_p: conservative,
                        reasoning: format!(
                            "Response is {:.1}% faster than target with low noise levels. \
                             This indicates excellent build quality with headroom for P increase. \
                             Note: Verify prop size is correct (may be smaller than specified).",
                            td_deviation_percent
                        ),
                    }
                } else {
                    PRecommendation::Optimal {
                        reasoning: format!(
                            "Response is {:.1}% faster than target with low noise. \
                             Current P ({}) is optimal. Excellent build quality or prop size may differ from spec.",
                            td_deviation_percent, current_p
                        ),
                    }
                }
            }

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
                    PRecommendation::Increase {
                        conservative_p: conservative,
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

        // Compact header - axis name and basic info
        output.push_str(&format!(
            "{}: Td={:.1}ms (target {:.1}±{:.1}ms, {:+.0}% dev), Noise={}, Consistency={:.0}%\n",
            axis_name,
            self.td_stats.mean_ms,
            td_target,
            td_tolerance,
            self.td_deviation_percent,
            self.noise_level.name(),
            self.td_stats.consistency * 100.0
        ));

        // Warning for low consistency (inline)
        if !self.td_stats.is_consistent() {
            output.push_str(&format!(
                "  ⚠ WARNING: Low consistency (CV={:.1}%, {}/{} responses) - results may be unreliable\n",
                self.td_stats.coefficient_of_variation * 100.0,
                (self.td_stats.consistency * self.td_stats.num_samples as f64).round() as usize,
                self.td_stats.num_samples
            ));
        }

        // Compact recommendation
        output.push_str(&format!("  Current P={}\n", self.current_p));

        match &self.recommendation {
            PRecommendation::Optimal { reasoning } => {
                output.push_str("    → Optimal (no change recommended)\n");
                output.push_str(&format!("    {}\n", reasoning));
            }
            PRecommendation::Increase {
                conservative_p,
                reasoning,
            } => {
                output.push_str("    → Increase recommended:\n");

                // Calculate P delta
                let conservative_delta = *conservative_p as i32 - self.current_p as i32;

                // Show P recommendation (conservative only for simplicity)
                output.push_str(&format!(
                    "      Conservative: P≈{} ({:+})",
                    conservative_p, conservative_delta
                ));

                // Add D recommendation using recommended P:D ratio (not current ratio!)
                if let (Some(current_d), Some(rec_pd)) =
                    (self.current_d, self.recommended_pd_conservative)
                {
                    if rec_pd > 0.0 && current_d > 0 {
                        let conservative_d = ((*conservative_p as f64) / rec_pd).round() as u32;
                        let conservative_d_delta = conservative_d as i32 - current_d as i32;
                        output.push_str(&format!(
                            ", D≈{} ({:+})",
                            conservative_d, conservative_d_delta
                        ));
                    }
                }
                output.push('\n');

                output.push_str(&format!("    {}\n", reasoning));
            }
            PRecommendation::Decrease {
                recommended_p,
                reasoning,
            } => {
                output.push_str("    → Decrease recommended:\n");
                let decrease_delta = *recommended_p as i32 - self.current_p as i32;
                output.push_str(&format!("      P≈{} ({:+})", recommended_p, decrease_delta));

                // Add D recommendation using recommended P:D ratio (not current ratio!)
                // For decrease, use conservative ratio (safer)
                if let (Some(current_d), Some(rec_pd)) =
                    (self.current_d, self.recommended_pd_conservative)
                {
                    if rec_pd > 0.0 && current_d > 0 {
                        let recommended_d = ((*recommended_p as f64) / rec_pd).round() as u32;
                        let d_delta = recommended_d as i32 - current_d as i32;
                        output.push_str(&format!(", D≈{} ({:+})", recommended_d, d_delta));
                    }
                }
                output.push('\n');

                output.push_str(&format!("    {}\n", reasoning));
            }
            PRecommendation::Investigate { issue } => {
                output.push_str("    → ⚠ INVESTIGATION RECOMMENDED\n");
                output.push_str(&format!("    {}\n", issue));
            }
        }

        output
    }
}
