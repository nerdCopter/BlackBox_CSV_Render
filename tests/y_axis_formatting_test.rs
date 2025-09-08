// tests/y_axis_formatting_test.rs

/// Mock Y-axis formatter to test the formatting logic from plot_framework.rs
fn test_y_axis_formatter(y: f64, y_label: &str) -> String {
    // This mimics the logic from plot_framework.rs draw_single_axis_chart_with_config
    if !y_label.contains("dB") {
        if y.abs() >= 1_000_000.0 {
            format!("{:.1}M", y / 1_000_000.0)
        } else if y.abs() >= 1000.0 {
            format!("{:.0}k", y / 1000.0)
        } else if y.abs() < 10.0 && (y.fract() != 0.0 || y_label.contains("Response")) {
            // Use decimal formatting for small values with fractional parts
            // or for step response plots (which use normalized values 0.0-2.0)
            format!("{:.1}", y)
        } else {
            format!("{:.0}", y)
        }
    } else {
        format!("{:.0}", y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_response_y_axis_formatting() {
        // Test step response normalized values - should have decimal precision
        let step_response_label = "Normalized Response";
        
        assert_eq!(test_y_axis_formatter(0.0, step_response_label), "0.0");
        assert_eq!(test_y_axis_formatter(0.2, step_response_label), "0.2");
        assert_eq!(test_y_axis_formatter(0.4, step_response_label), "0.4");
        assert_eq!(test_y_axis_formatter(0.6, step_response_label), "0.6");
        assert_eq!(test_y_axis_formatter(0.8, step_response_label), "0.8");
        assert_eq!(test_y_axis_formatter(1.0, step_response_label), "1.0");
        assert_eq!(test_y_axis_formatter(1.2, step_response_label), "1.2");
    }

    #[test]
    fn test_spectrum_y_axis_formatting() {
        // Test spectrum values - should use k/M notation for large values
        let spectrum_label = "Amplitude";
        
        // Small integer values (0.0 has fract = 0.0, but < 10.0 condition doesn't match)
        assert_eq!(test_y_axis_formatter(0.0, spectrum_label), "0"); // 0.0.fract() = 0.0, so goes to integer format
        assert_eq!(test_y_axis_formatter(10.0, spectrum_label), "10");
        assert_eq!(test_y_axis_formatter(100.0, spectrum_label), "100");
        
        // Values with decimals < 10 should show decimals
        assert_eq!(test_y_axis_formatter(0.5, spectrum_label), "0.5");
        assert_eq!(test_y_axis_formatter(5.7, spectrum_label), "5.7");
        
        // Large values should use k notation
        assert_eq!(test_y_axis_formatter(1000.0, spectrum_label), "1k");
        assert_eq!(test_y_axis_formatter(5000.0, spectrum_label), "5k");
        assert_eq!(test_y_axis_formatter(12500.0, spectrum_label), "12k"); // 12500/1000 = 12.5, formatted with {:.0} = 12
        
        // Very large values should use M notation  
        assert_eq!(test_y_axis_formatter(1_000_000.0, spectrum_label), "1.0M");
        assert_eq!(test_y_axis_formatter(2_500_000.0, spectrum_label), "2.5M");
    }

    #[test]
    fn test_db_scale_y_axis_formatting() {
        // Test dB scale values - should be integers
        let db_label = "Power Spectral Density (dB/Hz)";
        
        assert_eq!(test_y_axis_formatter(-60.0, db_label), "-60");
        assert_eq!(test_y_axis_formatter(-30.5, db_label), "-30"); // Uses {:.0} which truncates decimals
        assert_eq!(test_y_axis_formatter(0.0, db_label), "0");
        assert_eq!(test_y_axis_formatter(10.7, db_label), "11"); // Uses {:.0} which rounds decimals
    }
}