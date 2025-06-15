// src/data_analysis/filter_delay.rs

use crate::constants::{
    MIN_CORRELATION_THRESHOLD, FALLBACK_CORRELATION_THRESHOLD, MAX_DELAY_FRACTION, MAX_DELAY_SAMPLES, MIN_SAMPLES_FOR_DELAY
};
use ndarray::Array1;
use std::fmt;

/// Error type for delay calculation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are used in Display implementation
pub enum DelayCalculationError {
    InsufficientData { samples: usize, minimum: usize },
    LowCorrelation { correlation: f32, threshold: f32 },
    SignalMismatch,
}

impl fmt::Display for DelayCalculationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DelayCalculationError::InsufficientData { samples, minimum } => {
                write!(f, "Insufficient data: {} samples available, minimum {} required", samples, minimum)
            }
            DelayCalculationError::LowCorrelation { correlation, threshold } => {
                write!(f, "Low correlation: {:.3} below threshold {:.3}", correlation, threshold)
            }
            DelayCalculationError::SignalMismatch => {
                write!(f, "Signal mismatch: filtered and unfiltered signals have different lengths")
            }
        }
    }
}

/// Calculate filtering delay using cross-correlation between filtered and unfiltered signals
/// Returns delay in milliseconds, or a detailed error
pub fn calculate_filtering_delay(
    filtered: &Array1<f32>,
    unfiltered: &Array1<f32>,
    sample_rate: f64
) -> Result<f32, DelayCalculationError> {
    if sample_rate <= 0.0 {
        return Err(DelayCalculationError::InsufficientData {
            samples: 0,
            minimum: 1, // sentinel for invalid sample rate
        });
    }
    if filtered.len() != unfiltered.len() {
        return Err(DelayCalculationError::SignalMismatch);
    }
    if filtered.len() < MIN_SAMPLES_FOR_DELAY {
        return Err(DelayCalculationError::InsufficientData { samples: filtered.len(), minimum: MIN_SAMPLES_FOR_DELAY });
    }
    let n = filtered.len();
    let max_delay_samples = (n / MAX_DELAY_FRACTION).min(MAX_DELAY_SAMPLES);
    let mut best_correlation = f64::NEG_INFINITY;
    let mut best_delay = 0;
    for delay in 1..max_delay_samples {
        if delay >= n { break; }
        let len = n - delay;
        if len < MIN_SAMPLES_FOR_DELAY { break; }
        
        // Additional bounds check for safety
        let safe_len = len.min(filtered.len().saturating_sub(delay)).min(unfiltered.len());
        if safe_len < MIN_SAMPLES_FOR_DELAY { break; }
        
        let mut sum_xy = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for i in 0..safe_len {
            // Explicit bounds checking (should always pass due to safe_len calculation)
            if i + delay >= filtered.len() || i >= unfiltered.len() {
                break;
            }
            let x = filtered[i + delay] as f64;
            let y = unfiltered[i] as f64;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_x += x;
            sum_y += y;
        }
        let n_f = safe_len as f64;
        let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
        if denominator > 1e-10 {
            let correlation = (n_f * sum_xy - sum_x * sum_y) / denominator;
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }
    }
    if best_correlation > MIN_CORRELATION_THRESHOLD as f64 && best_delay > 0 {
        Ok((best_delay as f32 / sample_rate as f32) * 1000.0)
    } else {
        // Fallback
        let mut fallback_delay = 0;
        let mut fallback_correlation = 0.0f32;
        for delay in 1..((n/20).min(50)) {
            if delay >= n { break; }
            let len = n - delay;
            if len < 50 { break; }
            let mut correlation_sum = 0.0f32;
            let mut filtered_norm = 0.0f32;
            let mut unfiltered_norm = 0.0f32;
            for i in 0..len {
                let f_val = filtered[i + delay];
                let u_val = unfiltered[i];
                correlation_sum += f_val * u_val;
                filtered_norm += f_val * f_val;
                unfiltered_norm += u_val * u_val;
            }
            if filtered_norm > 1e-10 && unfiltered_norm > 1e-10 {
                let normalized_corr = correlation_sum / (filtered_norm.sqrt() * unfiltered_norm.sqrt());
                if normalized_corr > fallback_correlation {
                    fallback_correlation = normalized_corr;
                    fallback_delay = delay;
                }
            }
        }
        if fallback_correlation > FALLBACK_CORRELATION_THRESHOLD && fallback_delay > 0 {
            Ok((fallback_delay as f32 / sample_rate as f32) * 1000.0)
        } else {
            Err(DelayCalculationError::LowCorrelation { correlation: best_correlation as f32, threshold: MIN_CORRELATION_THRESHOLD })
        }
    }
}

/// Calculate average filtering delay across multiple axes
pub fn calculate_average_filtering_delay(
    log_data: &[crate::data_input::log_data::LogRowData], 
    sample_rate: f64
) -> Option<f32> {
    let axis_names = ["Roll", "Pitch", "Yaw"];
    let mut delays = Vec::new();
    
    for axis in 0..3 {
        // Extract filtered and unfiltered data for this axis
        let mut filtered_data = Vec::new();
        let mut unfiltered_data = Vec::new();
        
        for row in log_data {
            if let (Some(filtered), Some(unfiltered)) = (row.gyro[axis], row.gyro_unfilt[axis]) {
                filtered_data.push(filtered as f32);
                unfiltered_data.push(unfiltered as f32);
            }
        }
        
        if !filtered_data.is_empty() && filtered_data.len() == unfiltered_data.len() {
            let filtered_array = Array1::from(filtered_data);
            let unfiltered_array = Array1::from(unfiltered_data);
            
            match calculate_filtering_delay(&filtered_array, &unfiltered_array, sample_rate) {
                Ok(delay_ms) => {
                    println!("Gyro Filtering delay for {} axis: {:.2} ms", axis_names[axis], delay_ms);
                    delays.push(delay_ms);
                }
                Err(err) => {
                    println!("Gyro Filtering delay for {} axis: Unable to calculate - {}", axis_names[axis], err);
                }
            }
        }
    }
    
    if !delays.is_empty() {
        let average_delay = delays.iter().sum::<f32>() / delays.len() as f32;
        println!("Average Gyro Filtering delay across all axes: {:.2} ms", average_delay);
        Some(average_delay)
    } else {
        println!("Average Gyro Filtering delay: Unable to calculate for any axis");
        None
    }
}

/// Calculate average filtering delay across multiple axes using dual-method comparison
/// Returns None for average_delay (deprecated) and Vec<DelayResult> with separate method results
pub fn calculate_average_filtering_delay_comparison(
    log_data: &[crate::data_input::log_data::LogRowData],
    sample_rate: f64
) -> Option<(Option<f32>, Vec<DelayResult>)> {
    let axis_names = ["Roll", "Pitch", "Yaw"];
    let mut all_results: Vec<DelayResult> = Vec::new();
    
    // First, diagnose data availability
    println!("=== Gyro Data Availability Diagnostic ===");
    for axis in 0..3 {
        let mut gyro_count = 0;
        let mut gyro_unfilt_count = 0;
        let mut both_available = 0;
        let mut longest_continuous = 0;
        let mut current_continuous = 0;
        
        for row in log_data {
            if row.gyro[axis].is_some() { gyro_count += 1; }
            if row.gyro_unfilt[axis].is_some() { gyro_unfilt_count += 1; }
            if row.gyro[axis].is_some() && row.gyro_unfilt[axis].is_some() { 
                both_available += 1; 
                current_continuous += 1;
                longest_continuous = longest_continuous.max(current_continuous);
            } else {
                current_continuous = 0;
            }
        }
        
        println!("  {} axis: gyro={}, unfilt={}, both={}, longest_continuous={}", 
                axis_names[axis], gyro_count, gyro_unfilt_count, both_available, longest_continuous);
    }
    
    for axis in 0..3 {
        // Extract filtered and unfiltered data for this axis
        let mut filtered_data = Vec::new();
        let mut unfiltered_data = Vec::new();
        
        for row in log_data {
            if let (Some(filtered), Some(unfiltered)) = (row.gyro[axis], row.gyro_unfilt[axis]) {
                filtered_data.push(filtered as f32);
                unfiltered_data.push(unfiltered as f32);
            }
        }
        
        println!("  Axis {} ({}) extracted: {} samples", axis, axis_names[axis], filtered_data.len());
        
        if !filtered_data.is_empty() && filtered_data.len() == unfiltered_data.len() {
            let filtered_array = Array1::from(filtered_data);
            let unfiltered_array = Array1::from(unfiltered_data);
            
            let axis_results = calculate_filtering_delay_comparison(&filtered_array, &unfiltered_array, sample_rate);
            
            println!("Gyro Filtering delay analysis for {} axis:", axis_names[axis]);
            for result in &axis_results {
                if let Some(freq) = result.frequency_hz {
                    println!("  {}: {:.2} ms (confidence: {:.0}%, freq: {:.1} Hz)", 
                        result.method, result.delay_ms, result.confidence * 100.0, freq);
                } else {
                    println!("  {}: {:.2} ms (confidence: {:.0}%)", 
                        result.method, result.delay_ms, result.confidence * 100.0);
                }
            }
            
            all_results.extend(axis_results);
        }
    }
    
    if !all_results.is_empty() {
        let mut method_summaries = Vec::new();
        let enhanced_results: Vec<&DelayResult> = all_results.iter()
            .filter(|r| r.method == "Enhanced Cross-Correlation")
            .collect();
        if !enhanced_results.is_empty() {
            let avg_delay = enhanced_results.iter()
                .map(|r| r.delay_ms)
                .sum::<f32>() / enhanced_results.len() as f32;
            let avg_confidence = enhanced_results.iter()
                .map(|r| r.confidence)
                .sum::<f32>() / enhanced_results.len() as f32;
            method_summaries.push(DelayResult {
                method: "Enhanced Cross-Correlation".to_string(),
                delay_ms: avg_delay,
                confidence: avg_confidence,
                frequency_hz: None,
            });
            Some((Some(avg_delay), method_summaries))
        } else {
            Some((None, method_summaries))
        }
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct DelayResult {
    pub method: String,
    pub delay_ms: f32,
    pub confidence: f32,
    pub frequency_hz: Option<f32>, // Preserved for compatibility, typically None for cross-correlation
}

/// Calculate filtering delay using enhanced cross-correlation method only
fn calculate_filtering_delay_comparison(
    filtered: &Array1<f32>, 
    unfiltered: &Array1<f32>, 
    sample_rate: f64
) -> Vec<DelayResult> {
    let mut results = Vec::new();
    
    // Enhanced Cross-correlation method with sub-sample precision (only method used)
    if let Some(enhanced_result) = calculate_filtering_delay_enhanced_xcorr(filtered, unfiltered, sample_rate) {
        results.push(enhanced_result);
    } else {
        println!("  Enhanced Cross-correlation method failed for {} samples", filtered.len());
    }
    
    results
}

/// Enhanced cross-correlation with parabolic peak interpolation for sub-sample accuracy
/// This addresses the expert feedback about achieving better than sample-rate precision
fn calculate_filtering_delay_enhanced_xcorr(
    filtered: &Array1<f32>, 
    unfiltered: &Array1<f32>, 
    sample_rate: f64
) -> Option<DelayResult> {
    if filtered.len() != unfiltered.len() || filtered.len() < MIN_SAMPLES_FOR_DELAY {
        return None;
    }
    let n = filtered.len();
    let max_delay_samples = (n / MAX_DELAY_FRACTION).min(MAX_DELAY_SAMPLES);
    let mut correlations = Vec::with_capacity(max_delay_samples);
    let mut best_correlation = f64::NEG_INFINITY;
    let mut best_delay = 0;
    for delay in 1..max_delay_samples {
        if delay >= n { break; }
        let len = n - delay;
        if len < MIN_SAMPLES_FOR_DELAY { break; }
        
        // Additional bounds check for safety
        let safe_len = len.min(filtered.len().saturating_sub(delay)).min(unfiltered.len());
        if safe_len < MIN_SAMPLES_FOR_DELAY { break; }
        
        let mut sum_xy = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for i in 0..safe_len {
            // Explicit bounds checking (should always pass due to safe_len calculation)
            if i + delay >= filtered.len() || i >= unfiltered.len() {
                break;
            }
            let x = filtered[i + delay] as f64;
            let y = unfiltered[i] as f64;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_x += x;
            sum_y += y;
        }
        let n_f = safe_len as f64;
        let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
        if denominator > 1e-10 {
            let correlation = (n_f * sum_xy - sum_x * sum_y) / denominator;
            correlations.push(correlation as f32);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        } else {
            correlations.push(0.0);
        }
    }
    if best_correlation < FALLBACK_CORRELATION_THRESHOLD as f64 || best_delay == 0 {
        return None;
    }
    // Parabolic interpolation bounds check fix
    let idx = best_delay - 1; // map delay→index (delay 1 → index 0)
    if idx > 0 && idx < correlations.len() - 1 {
        let y1 = correlations[idx - 1];
        let y2 = correlations[idx];
        let y3 = correlations[idx + 1];
        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        let b = (y3 - y1) / 2.0;
        if a.abs() > 1e-10 {
            let sub_sample_offset = -b / (2.0 * a);
            let clamped_offset = sub_sample_offset.clamp(-0.5, 0.5);
            let refined_delay = best_delay as f32 + clamped_offset;
            return Some(DelayResult {
                method: "Enhanced Cross-Correlation".to_string(),
                delay_ms: (refined_delay / sample_rate as f32) * 1000.0,
                confidence: ((best_correlation + 1.0) / 2.0) as f32,
                frequency_hz: None,
            });
        }
    }
    Some(DelayResult {
        method: "Enhanced Cross-Correlation".to_string(),
        delay_ms: (best_delay as f32 / sample_rate as f32) * 1000.0,
        confidence: ((best_correlation + 1.0) / 2.0) as f32,
        frequency_hz: None,
    })
}
