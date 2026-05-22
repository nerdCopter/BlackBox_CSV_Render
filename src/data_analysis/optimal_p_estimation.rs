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

/// Error type for optimal P analysis
#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// No Td target available for the given frame class
    MissingTdTarget { message: String },
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::MissingTdTarget { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Safe conversion from scaled f64 to u32 with saturation.
///
/// Computes (base * multiplier) and returns a saturated `u32` result:
/// - If `multiplier` is NaN, returns 0 (deterministic handling of invalid multiplier).
/// - If the scaled value is >= `u32::MAX`, returns `u32::MAX`.
/// - If the scaled value is <= 0.0, returns 0.
/// - Otherwise returns the truncated `u32` value.
fn safe_scaled_p(base: u32, multiplier: f64) -> u32 {
    // Explicitly handle NaN deterministically rather than relying on cast behavior
    if multiplier.is_nan() {
        return 0;
    }

    let scaled = (base as f64) * multiplier;
    if scaled.is_infinite() {
        if scaled.is_sign_positive() {
            return u32::MAX;
        } else {
            return 0;
        }
    }

    if scaled >= (u32::MAX as f64) {
        u32::MAX
    } else if scaled <= 0.0 {
        0
    } else {
        scaled as u32
    }
}

/// Noise level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub coefficient_of_variation: Option<f64>,
    pub num_samples: usize,
    pub consistency: f64, // Fraction of samples within ±1 std dev
}

impl TdStatistics {
    /// Calculate statistics from array of Td values (in milliseconds)
    pub fn from_samples(td_samples_ms: &[f64]) -> Option<Self> {
        let td_samples_ms: Vec<f64> = td_samples_ms
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

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
        // For small samples, set coefficient_of_variation to None to indicate insufficient data
        let coefficient_of_variation = if td_samples_ms.len() < TD_SAMPLES_MIN_FOR_STDDEV {
            None
        } else {
            let sum_sq_dev = td_samples_ms
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>();
            let variance = sum_sq_dev / (n - 1.0);
            let std_dev = variance.sqrt();
            Some(std_dev / mean)
        };

        // Calculate consistency: fraction within ±1 std dev
        // When coefficient_of_variation is None (too few samples), consistency is perfect (1.0)
        // Otherwise, tolerance = cv * mean and calculate fraction within range
        let consistency = coefficient_of_variation.map_or(1.0, |cv| {
            let tolerance = cv * mean;
            let within_range = td_samples_ms
                .iter()
                .filter(|&&x| (x - mean).abs() <= tolerance)
                .count();
            within_range as f64 / n
        });

        Some(TdStatistics {
            mean_ms: mean,
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
            && self
                .coefficient_of_variation
                .map_or(true, |cv| cv <= TD_COEFFICIENT_OF_VARIATION_MAX)
    }
}

/// Complete optimal P analysis result
#[derive(Debug, Clone)]
pub struct OptimalPAnalysis {
    pub current_p: u32,
    pub current_d: Option<u32>,
    pub recommended_pd_conservative: Option<f64>,
    pub td_stats: TdStatistics,
    pub td_deviation: TdDeviation,
    pub td_deviation_percent: f64,
    pub noise_level: NoiseLevel,
    pub recommendation: PRecommendation,
    /// Actual Td target (in ms) used during analysis (from physics)
    pub td_target_ms: f64,
    /// Actual Td tolerance (in ms) used during analysis (from physics)
    pub td_tolerance_ms: f64,
    /// Number of throttle-punch events used to derive the physics Td target.
    pub source_events: usize,
    /// Number of flight files that contributed to the physics Td target.
    pub source_files: usize,
}

impl OptimalPAnalysis {
    /// Analyze optimal P for a given axis
    ///
    /// # Arguments
    /// * `td_samples_ms` - Array of Td measurements from multiple step responses (milliseconds)
    /// * `current_p` - Current P gain
    /// * `current_d` - Current D gain (optional)
    /// * `hf_energy_ratio` - Optional: ratio of D-term energy above DTERM_HF_CUTOFF_HZ (0.0-1.0)
    /// * `recommended_pd_conservative` - Optional: recommended P:D ratio from step response (conservative)
    /// * `physics_td_target_ms` - Physics-derived (td_target, tolerance) from torque_inertia_profiler
    pub fn analyze(
        td_samples_ms: &[f64],
        current_p: u32,
        current_d: Option<u32>,
        hf_energy_ratio: Option<f64>,
        recommended_pd_conservative: Option<f64>,
        physics_td_target_ms: Option<(f64, f64)>,
    ) -> Result<Self, AnalysisError> {
        // Calculate Td statistics
        let td_stats = TdStatistics::from_samples(td_samples_ms).ok_or_else(|| {
            AnalysisError::MissingTdTarget {
                message: "Failed to calculate Td statistics from samples.".to_string(),
            }
        })?;

        // Get target Td — use physics-based value or error
        let (td_target_ms, td_tolerance_ms) =
            if let Some((phys_target, phys_tol)) = physics_td_target_ms {
                (phys_target, phys_tol)
            } else {
                return Err(AnalysisError::MissingTdTarget {
                    message: "No physics-derived Td target available. \
                              Ensure throttle-punch events were detected (need \
                              \u{2265}5 events)."
                        .to_string(),
                });
            };

        // Defensive check: td_target_ms must be above domain minimum to be physically meaningful
        if td_target_ms <= MIN_TD_MS {
            return Err(AnalysisError::MissingTdTarget {
                message: format!(
                    "Invalid Td target ({:.3}ms, minimum {:.3}ms) for optimal P analysis. Skipping.",
                    td_target_ms, MIN_TD_MS
                ),
            });
        }

        // Calculate deviation from target (safe: td_target_ms validated above)
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

        Ok(OptimalPAnalysis {
            current_p,
            current_d,
            recommended_pd_conservative,
            td_stats,
            td_deviation,
            td_deviation_percent,
            noise_level,
            recommendation,
            td_target_ms,
            td_tolerance_ms,
            source_events: 0,
            source_files: 0,
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
                let conservative = safe_scaled_p(current_p, P_HEADROOM_MODERATE_MULTIPLIER);
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
                let conservative = safe_scaled_p(current_p, P_HEADROOM_CONSERVATIVE_MULTIPLIER);
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
                let conservative = safe_scaled_p(current_p, P_HEADROOM_CONSERVATIVE_MULTIPLIER);
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
                let recommended = safe_scaled_p(current_p, P_REDUCTION_MODERATE_MULTIPLIER);
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
                let conservative = safe_scaled_p(current_p, P_HEADROOM_CONSERVATIVE_MULTIPLIER);
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
                        safe_scaled_p(current_p, P_HEADROOM_CONSERVATIVE_MULTIPLIER);
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
        let mut output = String::new();

        // Header: axis name and Td measurement
        output.push_str(&format!(
            "{}: Td={:.1}ms (target: {:.1}±{:.1}ms, windows={})\n",
            axis_name,
            self.td_stats.mean_ms,
            self.td_target_ms,
            self.td_tolerance_ms,
            self.td_stats.num_samples,
        ));
        // Td source: group/single, flights, punches
        let source_label = if self.source_files > 1 {
            "File Group"
        } else {
            "Single File"
        };
        output.push_str(&format!(
            "  Td source: {} — {} flights, {} throttle-punches\n",
            source_label, self.source_files, self.source_events,
        ));
        output.push_str(&format!("  Noise: {}\n", self.noise_level.name()));

        // Deviation
        let deviation_sign = if self.td_deviation_percent > 0.0 {
            "+"
        } else {
            ""
        };
        output.push_str(&format!(
            "  Deviation: {}{:.1}% ({})\n",
            deviation_sign,
            self.td_deviation_percent,
            self.td_deviation.name(),
        ));

        // Current P
        output.push_str(&format!("  Current P={}\n", self.current_p));

        // Recommendation — shared D-suffix helper
        let effective_pd = self.recommended_pd_conservative.or_else(|| {
            self.current_d
                .filter(|&d| d > 0)
                .map(|d| self.current_p as f64 / d as f64)
        });
        let d_suffix = |recommended_p: u32| -> String {
            if let (Some(current_d), Some(rec_pd)) = (self.current_d, effective_pd) {
                if rec_pd > 0.0 && current_d > 0 {
                    let rec_d = (recommended_p as f64 / rec_pd).round() as u32;
                    let d_delta = rec_d as i32 - current_d as i32;
                    return format!(", D≈{} ({:+})", rec_d, d_delta);
                }
            }
            String::new()
        };

        match &self.recommendation {
            PRecommendation::Optimal { .. } => {
                output.push_str(&format!(
                    "  Recommendation: Current P is optimal (P={})\n",
                    self.current_p
                ));
            }
            PRecommendation::Increase { conservative_p, .. } => {
                let p_delta = *conservative_p as i32 - self.current_p as i32;
                output.push_str(&format!(
                    "  Recommendation (Conservative): P≈{} ({:+}){}\n",
                    conservative_p,
                    p_delta,
                    d_suffix(*conservative_p),
                ));
            }
            PRecommendation::Decrease { recommended_p, .. } => {
                let p_delta = *recommended_p as i32 - self.current_p as i32;
                output.push_str(&format!(
                    "  Recommendation (Decrease): P≈{} ({:+}){}\n",
                    recommended_p,
                    p_delta,
                    d_suffix(*recommended_p),
                ));
            }
            PRecommendation::Investigate { issue } => {
                output.push_str(&format!("  Recommendation: Investigate — {}\n", issue));
            }
        }

        // Reliability — always shown, after recommendation
        let cv_str = self.td_stats.coefficient_of_variation.map_or_else(
            || "CV=N/A".to_string(),
            |cv| {
                format!(
                    "CV={:.1}% (⊢≤{:.0}%)",
                    cv * 100.0,
                    TD_COEFFICIENT_OF_VARIATION_MAX * 100.0,
                )
            },
        );
        let cons_str = format!(
            "Consistency={:.0}% (⊢≥{:.0}%)",
            self.td_stats.consistency * 100.0,
            TD_CONSISTENCY_MIN_THRESHOLD * 100.0,
        );
        if self.td_stats.is_consistent() {
            output.push_str(&format!("  Reliable: {cons_str}, {cv_str}\n"));
        } else {
            output.push_str(&format!("  Unreliable: {cons_str}, {cv_str}\n"));
        }

        output
    }
}
