// src/data_analysis/derivative.rs

/// Minimum number of data points required for derivative calculation
const MIN_DERIVATIVE_SAMPLES: usize = 2;

/// Central difference coefficient for improved accuracy
const CENTRAL_DIFF_COEFFICIENT: f32 = 0.5;

/// Calculates discrete derivative of a time series using finite differences.
///
/// This function computes the rate of change of the input signal for D-term analysis,
/// representing how quickly the gyro signal is changing over time.
///
/// # Method
/// - Forward difference for first sample: `(data[1] - data[0]) * fs`
/// - Central difference for middle samples: `(data[i+1] - data[i-1]) * (0.5 * fs)`
/// - Backward difference for last sample: `(data[n] - data[n-1]) * fs`
///
/// # Arguments
/// * `data` - Input time series data to differentiate
/// * `sample_rate` - Sampling frequency in Hz (must be positive and finite)
///
/// # Returns
/// Vector containing derivative values, same length as input data.
/// Returns empty vector if input validation fails.
///
/// # Error Conditions
/// - Empty result if data length < 2 (need at least 2 points for derivative)
/// - Empty result if sample_rate is not finite or <= 0
pub fn calculate_derivative(data: &[f32], sample_rate: f64) -> Vec<f32> {
    if data.len() < MIN_DERIVATIVE_SAMPLES {
        return Vec::new();
    }

    // Validate sample_rate to prevent silent failures with invalid values
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Vec::new();
    }

    // Convert to samples per second (frequency) for multiplication instead of division
    let fs: f32 = sample_rate as f32;
    let mut derivative = Vec::with_capacity(data.len());

    // Use forward difference for first point
    derivative.push((data[1] - data[0]) * fs);

    // Use central difference for middle points
    for i in 1..data.len() - 1 {
        derivative.push((data[i + 1] - data[i - 1]) * (CENTRAL_DIFF_COEFFICIENT * fs));
    }

    // Use backward difference for last point
    let n = data.len() - 1;
    derivative.push((data[n] - data[n - 1]) * fs);

    derivative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let result = calculate_derivative(&[], 1000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_sample() {
        let result = calculate_derivative(&[1.0], 1000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let data = [1.0, 2.0, 3.0];
        assert!(calculate_derivative(&data, 0.0).is_empty());
        assert!(calculate_derivative(&data, -1.0).is_empty());
        assert!(calculate_derivative(&data, f64::NAN).is_empty());
        assert!(calculate_derivative(&data, f64::INFINITY).is_empty());
    }

    #[test]
    fn test_basic_derivative() {
        let data = [0.0, 1.0, 2.0, 3.0];
        let sample_rate = 1.0;
        let result = calculate_derivative(&data, sample_rate);

        assert_eq!(result.len(), 4);
        // Forward difference: (1-0)*1 = 1
        assert_eq!(result[0], 1.0);
        // Central difference: (2-0)*0.5*1 = 1
        assert_eq!(result[1], 1.0);
        // Central difference: (3-1)*0.5*1 = 1
        assert_eq!(result[2], 1.0);
        // Backward difference: (3-2)*1 = 1
        assert_eq!(result[3], 1.0);
    }
}
