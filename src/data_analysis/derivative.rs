// src/data_analysis/derivative.rs

/// Calculates discrete derivative of a time series
/// For D-term analysis, this represents the rate of change of gyro signal
pub fn calculate_derivative(data: &[f32], sample_rate: f64) -> Vec<f32> {
    if data.len() < 2 {
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
        derivative.push((data[i + 1] - data[i - 1]) * (0.5 * fs));
    }

    // Use backward difference for last point
    let n = data.len() - 1;
    derivative.push((data[n] - data[n - 1]) * fs);

    derivative
}
