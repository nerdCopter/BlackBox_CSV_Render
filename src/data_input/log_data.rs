// src/log_data.rs

/// Structure to hold data parsed from a single row of the CSV log.
/// Uses `Option<f64>` to handle potentially missing or unparseable values.
#[derive(Debug, Default, Clone)]
pub struct LogRowData {
    pub time_sec: Option<f64>,        // Timestamp (in seconds).
    pub p_term: [Option<f64>; 3],     // Proportional term [Roll, Pitch, Yaw].
    pub i_term: [Option<f64>; 3],     // Integral term [Roll, Pitch, Yaw].
    pub d_term: [Option<f64>; 3],     // Derivative term [Roll, Pitch, Yaw].
    pub f_term: [Option<f64>; 3],     // Feed-Forward  [Roll, Pitch, Yaw].
    pub setpoint: [Option<f64>; 4],   // Target setpoint value [Roll, Pitch, Yaw, Throttle].
    pub gyro: [Option<f64>; 3],       // Gyroscope readings (filtered) [Roll, Pitch, Yaw].
    pub gyro_unfilt: [Option<f64>; 3], // Unfiltered Gyroscope readings [Roll, Pitch, Yaw]. Fallback: debug[0..2].
    pub debug: [Option<f64>; 4],      // Debug values [0..3].
}

// src/log_data.rs
