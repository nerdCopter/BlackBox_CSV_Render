// src/data_analysis/filter_delay.rs

use ndarray::Array1;

/// Calculate filtering delay using cross-correlation between filtered and unfiltered signals
/// Returns delay in milliseconds, or None if calculation fails or correlation is too weak
pub fn calculate_filtering_delay(filtered: &Array1<f32>, unfiltered: &Array1<f32>, sample_rate: f64) -> Option<f32> {
    if filtered.len() != unfiltered.len() || filtered.len() < 100 {
        return None;
    }
    
    let n = filtered.len();
    let max_delay_samples = (n / 10).min(200); // Increase search range and use more reasonable fraction
    let mut best_correlation = f32::NEG_INFINITY;
    let mut best_delay = 0;
    
    // Calculate cross-correlation for different delays
    // Start from delay=1 since delay=0 means no filtering delay
    for delay in 1..max_delay_samples {
        if delay >= n { break; }
        
        let len = n - delay;
        if len < 100 { break; } // Need minimum samples for reliable correlation
        
        // Calculate normalized cross-correlation
        let mut sum_xy = 0.0f32;
        let mut sum_x2 = 0.0f32;
        let mut sum_y2 = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        
        for i in 0..len {
            let x = filtered[i + delay];
            let y = unfiltered[i];
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_x += x;
            sum_y += y;
        }
        
        let n_f = len as f32;
        let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator > 1e-10 {
            let correlation = (n_f * sum_xy - sum_x * sum_y) / denominator;
            
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }
    }
    
    // Convert delay from samples to milliseconds
    // If no significant delay found (best_delay is still 0), try to find peak correlation above delay=1
    if best_correlation > 0.3 && best_delay > 0 { // Lowered threshold and ensure delay > 0
        Some((best_delay as f32 / sample_rate as f32) * 1000.0)
    } else {
        // If correlation is too low or delay is 0, try alternative method
        // Look for peak in cross-correlation that's not at delay=0
        let mut fallback_delay = 0;
        let mut fallback_correlation = 0.0f32;
        
        for delay in 1..((n/20).min(50)) { // Smaller range for fallback
            if delay >= n { break; }
            let len = n - delay;
            if len < 50 { break; }
            
            // Simplified correlation calculation
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
        
        if fallback_correlation > 0.2 && fallback_delay > 0 {
            Some((fallback_delay as f32 / sample_rate as f32) * 1000.0)
        } else {
            None
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
            
            if let Some(delay_ms) = calculate_filtering_delay(&filtered_array, &unfiltered_array, sample_rate) {
                println!("Gyro Filtering delay for {} axis: {:.2} ms", axis_names[axis], delay_ms);
                delays.push(delay_ms);
            } else {
                println!("Gyro Filtering delay for {} axis: Unable to calculate (correlation too low)", axis_names[axis]);
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
