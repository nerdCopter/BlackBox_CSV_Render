// src/plot_functions/plot_motor_spectrums.rs

use ndarray::Array1;
use std::error::Error;

use crate::constants::{
    LINE_WIDTH_PLOT, MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD, MOTOR_OSCILLATION_FREQ_MAX_HZ,
    MOTOR_OSCILLATION_FREQ_MIN_HZ, MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER, TUKEY_ALPHA,
};
use crate::data_analysis::calc_step_response; // For tukeywin
use crate::data_analysis::fft_utils; // For fft_forward
use crate::data_input::log_data::LogRowData;
use plotters::prelude::*;
use plotters::style::colors::full_palette::{AMBER, BLUE, GREEN, ORANGE, PURPLE, RED};
use plotters::style::RGBColor;

/// Motor colors for consistent visualization (rotating palette for up to 8 motors)
const MOTOR_COLORS: [RGBColor; 8] = [
    BLUE,                                   // Motor 0
    GREEN,                                  // Motor 1
    ORANGE,                                 // Motor 2
    PURPLE,                                 // Motor 3
    RED,                                    // Motor 4
    AMBER,                                  // Motor 5
    plotters::style::RGBColor(0, 255, 255), // Cyan - Motor 6
    plotters::style::RGBColor(255, 0, 255), // Magenta - Motor 7
];

/// Type alias for motor spectrum data: (frequencies, amplitudes, max_amplitude)
type MotorSpectrumData = (Vec<f64>, Vec<f64>, f64);

/// Generates stacked motor spectrum plots showing frequency content of each motor output.
/// Useful for identifying motor oscillations, ESC noise, and saturation issues.
pub fn plot_motor_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let output_file = format!("{root_name}_Motor_Spectrums_stacked.png");

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Motor Spectrum Plot: Sample rate could not be determined.");
        return Ok(());
    };

    // Determine motor count from first row
    let motor_count = log_data.first().map(|row| row.motors.len()).unwrap_or(0);

    if motor_count == 0 {
        println!("\nINFO: Skipping Motor Spectrum Plot: No motor data available.");
        return Ok(());
    }

    println!(
        "\n--- Generating Motor Spectrum Analysis ({} motor{}) ---",
        motor_count,
        if motor_count == 1 { "" } else { "s" }
    );

    let mut global_max_amplitude = 0.0f64;

    // Extract motor data for each motor
    let mut motor_samples: Vec<Vec<f32>> = vec![Vec::new(); motor_count];

    for row in log_data {
        for (motor_idx, motor_val) in row.motors.iter().enumerate() {
            // Defensive: skip any motor entries that exceed the expected motor count
            // (shouldn't happen if headers are consistent, but protects against malformed logs)
            if motor_idx >= motor_samples.len() {
                continue; // Skip if motor count varies unexpectedly
            }
            if let Some(val) = motor_val {
                motor_samples[motor_idx].push(*val as f32);
            }
        }
    }

    // Process FFT for each motor
    let mut motor_spectrums: Vec<Option<MotorSpectrumData>> = Vec::new();

    for (motor_idx, samples) in motor_samples.iter().enumerate() {
        if samples.is_empty() {
            println!("  Motor {}: No data available", motor_idx);
            motor_spectrums.push(None);
            continue;
        }

        let n_samples = samples.len();

        // Apply Tukey window
        let tukey_window = calc_step_response::tukeywin(n_samples, TUKEY_ALPHA);
        let windowed_data: Array1<f32> = Array1::from_vec(samples.clone()) * &tukey_window;

        // Compute FFT
        let fft_output = fft_utils::fft_forward(&windowed_data);
        let num_freqs = fft_output.len();

        // Generate frequency axis
        let freq_spacing = sr_value / n_samples as f64;
        let frequencies: Vec<f64> = (0..num_freqs).map(|i| i as f64 * freq_spacing).collect();

        // Compute magnitude spectrum (use 2/N scaling for better visibility)
        let amplitudes: Vec<f64> = fft_output
            .iter()
            .map(|c| {
                let magnitude = (c.re.powi(2) + c.im.powi(2)).sqrt();
                // Use 2/N scaling to get peak amplitude (standard for real signals)
                (2.0 * magnitude / n_samples as f32) as f64
            })
            .collect();

        // Find max amplitude for this motor (skip DC at index 0)
        let motor_max = if amplitudes.len() > 1 {
            amplitudes[1..].iter().copied().fold(0.0f64, f64::max)
        } else {
            amplitudes.first().copied().unwrap_or(0.0)
        };

        if motor_max > global_max_amplitude {
            global_max_amplitude = motor_max;
        }

        motor_spectrums.push(Some((frequencies, amplitudes, motor_max)));
        println!("  Motor {}: Max amplitude = {:.2}", motor_idx, motor_max);
    }

    // Check for oscillation issues (peaks > 3× average in 50-200 Hz range)
    for (motor_idx, spectrum_data) in motor_spectrums.iter().enumerate() {
        if let Some((frequencies, amplitudes, _)) = spectrum_data {
            let freq_range_50_200: Vec<(f64, f64)> = frequencies
                .iter()
                .zip(amplitudes.iter())
                .filter(|(f, _)| {
                    **f >= MOTOR_OSCILLATION_FREQ_MIN_HZ && **f <= MOTOR_OSCILLATION_FREQ_MAX_HZ
                })
                .map(|(f, a)| (*f, *a))
                .collect();

            if !freq_range_50_200.is_empty() {
                let avg_amplitude: f64 = freq_range_50_200.iter().map(|(_, a)| a).sum::<f64>()
                    / freq_range_50_200.len() as f64;
                let max_in_range = freq_range_50_200
                    .iter()
                    .map(|(_, a)| *a)
                    .fold(0.0f64, f64::max);

                if max_in_range > MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER * avg_amplitude
                    && max_in_range > MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD
                {
                    println!(
                        "  ⚠ Motor {}: Potential oscillation detected in {:.0}-{:.0} Hz range (peak {:.1} >> avg {:.1})",
                        motor_idx, MOTOR_OSCILLATION_FREQ_MIN_HZ, MOTOR_OSCILLATION_FREQ_MAX_HZ, max_in_range, avg_amplitude
                    );
                }
            }
        }
    }

    // Generate stacked plots with dynamic row count for motors
    let root_area = BitMapBackend::new(
        &output_file,
        (crate::constants::PLOT_WIDTH, crate::constants::PLOT_HEIGHT),
    )
    .into_drawing_area();
    root_area.fill(&WHITE)?;
    root_area.draw(&Text::new(
        root_name,
        (10, 10),
        crate::font_config::FONT_TUPLE_MAIN_TITLE
            .into_font()
            .color(&BLACK),
    ))?;

    let margined_root_area = root_area.margin(50, 5, 5, 5);
    let sub_plot_areas = margined_root_area.split_evenly((motor_count, 1));
    let mut any_motor_plotted = false;

    for motor_idx in 0..motor_count {
        let area = &sub_plot_areas[motor_idx];

        if let Some(spectrum_data) = motor_spectrums.get(motor_idx).and_then(|opt| opt.as_ref()) {
            let (frequencies, amplitudes, _) = spectrum_data;

            // Use full frequency range starting from 0 Hz with static Y-cap
            // This shows throttle-dominated low frequencies (0-10 Hz) and motor diagnostics (10+Hz)
            let nyquist_freq = sr_value / 2.0;

            let filtered_data: Vec<(f64, f64)> = frequencies
                .iter()
                .zip(amplitudes.iter())
                .take_while(|(f, _)| **f <= nyquist_freq)
                .map(|(f, a)| (*f, *a))
                .collect();

            if !filtered_data.is_empty() {
                // Use static Y-axis range for standardized comparison across copters.
                // MOTOR_SPECTRUM_Y_AXIS_MAX provides consistent visual scaling and future-proofs against outliers.
                let y_range = 0.0..crate::constants::MOTOR_SPECTRUM_Y_AXIS_MAX;

                // X-axis: show full range from 0 Hz (includes throttle and motor data)
                let x_max = filtered_data
                    .last()
                    .map(|(f, _)| *f)
                    .unwrap_or(nyquist_freq);
                let x_range = 0.0..x_max;

                // Create chart
                let motor_color = MOTOR_COLORS[motor_idx % MOTOR_COLORS.len()];
                let chart_title = format!("Motor {} Spectrum", motor_idx);

                let mut chart = ChartBuilder::on(area)
                    .caption(&chart_title, crate::font_config::FONT_TUPLE_CHART_TITLE)
                    .margin(5)
                    .x_label_area_size(50)
                    .y_label_area_size(50)
                    .build_cartesian_2d(x_range, y_range)?;

                chart
                    .configure_mesh()
                    .x_desc("Frequency (Hz)")
                    .y_desc("Amplitude")
                    .x_labels(20)
                    .y_labels(10)
                    .x_label_formatter(&|x| format!("{:.0}", x))
                    .y_label_formatter(&|y| format!("{:.0}", y))
                    .light_line_style(WHITE.mix(0.7))
                    .label_style(crate::font_config::FONT_TUPLE_AXIS_LABEL)
                    .draw()?;

                chart
                    .draw_series(LineSeries::new(
                        filtered_data.iter().copied(),
                        ShapeStyle::from(motor_color).stroke_width(LINE_WIDTH_PLOT),
                    ))?
                    .label(format!("Motor {}", motor_idx))
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)],
                            ShapeStyle::from(&motor_color).stroke_width(LINE_WIDTH_PLOT),
                        )
                    });

                chart
                    .configure_series_labels()
                    .background_style(WHITE.mix(0.8))
                    .border_style(BLACK)
                    .draw()?;

                any_motor_plotted = true;
            }
        }
    }

    if any_motor_plotted {
        root_area.present()?;
        println!("  Stacked plot saved as '{}'.", output_file);
    } else {
        println!("  Skipping '{}': No motor data to plot.", output_file);
    }

    Ok(())
}
