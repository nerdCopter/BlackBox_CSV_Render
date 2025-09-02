// src/data_analysis/derivative.rs

/// Calculates discrete derivative of a time series
/// For D-term analysis, this represents the rate of change of gyro signal
pub fn calculate_derivative(data: &[f32], sample_rate: f64) -> Vec<f32> {
    if data.len() < 2 {
        return Vec::new();
    }

    let dt = 1.0 / sample_rate;
    let mut derivative = Vec::with_capacity(data.len());

    // Use forward difference for first point
    derivative.push((data[1] - data[0]) / dt as f32);

    // Use central difference for middle points
    for i in 1..data.len() - 1 {
        derivative.push((data[i + 1] - data[i - 1]) / (2.0 * dt as f32));
    }

    // Use backward difference for last point
    let n = data.len() - 1;
    derivative.push((data[n] - data[n - 1]) / dt as f32);

    derivative
}
