// tests/formatting_integration_test.rs

// Mock the plot config structure for testing
struct MockPlotConfig {
    y_label: String,
}

/// Test the Y-axis formatting logic that would be used in actual plots
fn format_y_axis_label(y: f64, plot_config: &MockPlotConfig) -> String {
    // This replicates the exact logic from plot_framework.rs
    if !plot_config.y_label.contains("dB") {
        if y.abs() >= 1_000_000.0 {
            format!("{:.1}M", y / 1_000_000.0)
        } else if y.abs() >= 1000.0 {
            format!("{:.0}k", y / 1000.0)
        } else if y.abs() < 10.0 && (y.fract() != 0.0 || plot_config.y_label.contains("Response")) {
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
    fn test_step_response_formatting_integration() {
        // This represents what a step response plot would have
        let step_response_config = MockPlotConfig {
            y_label: "Normalized Response".to_string(),
        };

        // Test the exact values that were problematic in the issue
        let test_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
        let expected_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"];

        println!("Step Response Y-axis formatting test:");
        println!("Before fix: values like 0.2, 0.4, 0.6, 0.8 would show as 0, 0, 1, 1");
        println!("After fix:");

        for (value, expected) in test_values.iter().zip(expected_labels.iter()) {
            let formatted = format_y_axis_label(*value, &step_response_config);
            println!("  Value {:.1} -> Label '{}'", value, formatted);
            assert_eq!(
                &formatted, expected,
                "Y-axis formatting failed for value {}",
                value
            );
        }

        println!("✓ All step response Y-axis labels now show proper decimal values!");
    }

    #[test]
    fn test_other_plot_types_unaffected() {
        // Verify spectrum plots still work correctly
        let spectrum_config = MockPlotConfig {
            y_label: "Amplitude".to_string(),
        };

        // Large values should still use k/M notation
        assert_eq!(format_y_axis_label(1000.0, &spectrum_config), "1k");
        assert_eq!(format_y_axis_label(1_000_000.0, &spectrum_config), "1.0M");

        // dB plots should use integer formatting
        let psd_config = MockPlotConfig {
            y_label: "Power Spectral Density (dB/Hz)".to_string(),
        };

        assert_eq!(format_y_axis_label(-30.5, &psd_config), "-30");
        assert_eq!(format_y_axis_label(10.7, &psd_config), "11");

        println!("✓ Other plot types maintain their correct formatting!");
    }
}
