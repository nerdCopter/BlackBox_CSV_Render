// src/plot_functions/plot_motor_spectrums.rs

use ndarray::Array1;
use std::error::Error;

use crate::constants::{
    LINE_WIDTH_PLOT, MIN_FFT_SAMPLES, MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD,
    MOTOR_OSCILLATION_FREQ_MAX_HZ, MOTOR_OSCILLATION_FREQ_MIN_HZ, MOTOR_OSCILLATION_HOP_FRACTION,
    MOTOR_OSCILLATION_MAX_GAP_TOLERANCE, MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER,
    MOTOR_OSCILLATION_WINDOW_S, MOTOR_SPECTRUM_AXIS_ORIGIN,
    MOTOR_SPECTRUM_Y_LABEL_PRECISION_THRESHOLD, NYQUIST_DIVISOR, TUKEY_ALPHA,
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

/// Per-motor result from oscillation analysis in the MOTOR_OSCILLATION_FREQ range.
/// `peak_in_range`/`avg_in_range`/`event_time_s` describe the single worst sliding window
/// found (see `detect_windowed_oscillation`), not a whole-log average — a brief oscillation
/// burst is the failure mode this looks for, and averaging over the whole flight is exactly
/// what hides one.
pub struct MotorOscillationResult {
    pub motor_idx: usize,
    pub max_amplitude: Option<f64>,
    pub oscillation_detected: bool,
    pub peak_in_range: Option<f64>,
    pub avg_in_range: Option<f64>,
    pub event_time_s: Option<f64>,
}

/// Scans `samples` in overlapping time-based windows and returns the worst (highest-peak)
/// window whose in-band peak clears the oscillation criteria (peak > N× the window's own
/// band average, and above an absolute floor) — the same criteria the whole-log spectrum
/// used before, just applied per-window instead of once over the entire flight. Returns
/// `(event_time_s, peak, avg)` for the worst window found, or `None` if no window ever
/// crosses the bar.
fn detect_windowed_oscillation(
    samples: &[f32],
    times: &[f64],
    sr_value: f64,
) -> Option<(f64, f64, f64)> {
    let win_samples =
        ((MOTOR_OSCILLATION_WINDOW_S * sr_value).round() as usize).max(MIN_FFT_SAMPLES);
    if samples.len() < win_samples {
        return None;
    }
    let hop = ((win_samples as f64 * MOTOR_OSCILLATION_HOP_FRACTION).round() as usize).max(1);
    let window = calc_step_response::tukeywin(win_samples, TUKEY_ALPHA);
    let freq_spacing = sr_value / win_samples as f64;

    // Expected span of a full window if samples are truly contiguous at sr_value — used
    // below to reject a window whose actual timestamp span is far wider, which means it
    // silently joined samples across a dropped-frame gap rather than a real oscillation.
    let expected_span_s = (win_samples - 1) as f64 / sr_value;

    // Checks one window starting at `start`, returning its (time, peak, avg) if it clears the
    // oscillation bar. `start` is always a valid window start (caller guarantees
    // start + win_samples <= samples.len()), so `times[start]` never panics.
    let check_window = |start: usize| -> Option<(f64, f64, f64)> {
        let actual_span_s = times[start + win_samples - 1] - times[start];
        if actual_span_s > expected_span_s * MOTOR_OSCILLATION_MAX_GAP_TOLERANCE {
            // A dropped-frame gap sits inside this window — its samples aren't uniformly
            // spaced at sr_value, so the FFT below would misread the gap as content.
            return None;
        }

        let seg: Array1<f32> =
            Array1::from_vec(samples[start..start + win_samples].to_vec()) * &window;
        let fft_output = fft_utils::fft_forward(&seg);

        let mut band_sum = 0.0f64;
        let mut band_peak = 0.0f64;
        let mut band_count = 0usize;
        for (i, c) in fft_output.iter().enumerate() {
            let freq = i as f64 * freq_spacing;
            if freq < MOTOR_OSCILLATION_FREQ_MIN_HZ {
                continue;
            }
            if freq > MOTOR_OSCILLATION_FREQ_MAX_HZ {
                break;
            }
            let magnitude = (c.re.powi(2) + c.im.powi(2)).sqrt();
            let amp = (2.0 * magnitude / win_samples as f32) as f64;
            band_sum += amp;
            band_count += 1;
            if amp > band_peak {
                band_peak = amp;
            }
        }

        if band_count == 0 {
            return None;
        }
        let avg = band_sum / band_count as f64;
        let is_oscillating = band_peak > MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER * avg
            && band_peak > MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD;
        is_oscillating.then_some((times[start], band_peak, avg))
    };

    let mut worst: Option<(f64, f64, f64)> = None;
    let consider = |start: usize, worst: &mut Option<(f64, f64, f64)>| {
        if let Some((t, peak, avg)) = check_window(start) {
            if worst.map_or(true, |(_, wp, _)| peak > wp) {
                *worst = Some((t, peak, avg));
            }
        }
    };

    let mut start = 0usize;
    while start + win_samples <= samples.len() {
        consider(start, &mut worst);
        start += hop;
    }

    // The stride above can land short of the log's final window (e.g. hop doesn't evenly
    // divide the tail), leaving a burst confined to the very end of the flight unchecked.
    // Always test the last possible window explicitly, regardless of stride alignment.
    let last_start = samples.len() - win_samples;
    if last_start % hop != 0 {
        consider(last_start, &mut worst);
    }

    worst
}

/// Generates stacked motor spectrum plots showing frequency content of each motor output.
/// Useful for identifying motor oscillations, ESC noise, and saturation issues.
pub fn plot_motor_spectrums(
    log_data: &[LogRowData],
    root_name: &str,
    sample_rate: Option<f64>,
) -> Result<Vec<MotorOscillationResult>, Box<dyn Error>> {
    let output_file = format!("{root_name}_Motor_Spectrums_stacked.png");

    let sr_value = if let Some(sr) = sample_rate {
        sr
    } else {
        println!("\nINFO: Skipping Motor Spectrum Plot: Sample rate could not be determined.");
        return Ok(vec![]);
    };

    // Determine motor count from first row
    let motor_count = log_data.first().map(|row| row.motors.len()).unwrap_or(0);

    if motor_count == 0 {
        println!("\nINFO: Skipping Motor Spectrum Plot: No motor data available.");
        return Ok(vec![]);
    }

    println!(
        "\n--- Generating Motor Spectrum Analysis ({} motor{}) ---",
        motor_count,
        if motor_count == 1 { "" } else { "s" }
    );

    // Extract motor data for each motor, alongside each sample's own row timestamp — the
    // windowed oscillation scan below needs real time to report when an episode occurred,
    // not just a sample index (motor_samples can have gaps relative to row count if any
    // row is missing a motor value).
    let mut motor_samples: Vec<Vec<f32>> = vec![Vec::new(); motor_count];
    let mut motor_times: Vec<Vec<f64>> = vec![Vec::new(); motor_count];

    for row in log_data {
        for (motor_idx, motor_val) in row.motors.iter().enumerate() {
            // Defensive: skip any motor entries that exceed the expected motor count
            // (shouldn't happen if headers are consistent, but protects against malformed logs)
            if motor_idx >= motor_samples.len() {
                continue; // Skip if motor count varies unexpectedly
            }
            // Only keep a sample when both its motor value and its row timestamp exist —
            // a fabricated 0.0 timestamp would misreport an event time and corrupt the
            // windowed scan's gap-detection below (see detect_windowed_oscillation).
            if let (Some(val), Some(t)) = (motor_val, row.time_sec) {
                motor_samples[motor_idx].push(*val as f32);
                motor_times[motor_idx].push(t);
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

        // Minimum samples needed for meaningful FFT
        if samples.len() < MIN_FFT_SAMPLES {
            println!(
                "  Motor {}: Insufficient data ({} samples, need >= {})",
                motor_idx,
                samples.len(),
                MIN_FFT_SAMPLES
            );
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

        motor_spectrums.push(Some((frequencies, amplitudes, motor_max)));
        println!("  Motor {}: Max amplitude = {:.2}", motor_idx, motor_max);
    }

    // Check for oscillation issues via a sliding window, not the single whole-log spectrum
    // computed above (that spectrum still drives the plot below, unchanged). A whole-log FFT
    // spreads a brief oscillation burst's energy across the entire flight's duration, diluting
    // it under the 3x-avg/absolute-amplitude bar long before a multi-minute flight is done;
    // confirmed against two real-world desync logs during development — one showed 45
    // sliding-window episodes this catches that the whole-log FFT reported as "None".
    let mut motor_osc_results: Vec<MotorOscillationResult> = Vec::new();
    for (motor_idx, samples) in motor_samples.iter().enumerate() {
        let max_amplitude = motor_spectrums
            .get(motor_idx)
            .and_then(|s| s.as_ref())
            .map(|(_, _, max)| *max);

        let worst = detect_windowed_oscillation(samples, &motor_times[motor_idx], sr_value);
        let oscillation_detected = worst.is_some();
        let (event_time_s, peak_in_range, avg_in_range) = match worst {
            Some((t, peak, avg)) => {
                println!(
                    "  ⚠ Motor {}: Potential oscillation detected in {:.0}-{:.0} Hz range at t={:.2}s (peak {:.1} >> avg {:.1})",
                    motor_idx, MOTOR_OSCILLATION_FREQ_MIN_HZ, MOTOR_OSCILLATION_FREQ_MAX_HZ, t, peak, avg
                );
                (Some(t), Some(peak), Some(avg))
            }
            None => (None, None, None),
        };

        motor_osc_results.push(MotorOscillationResult {
            motor_idx,
            max_amplitude,
            oscillation_detected,
            peak_in_range,
            avg_in_range,
            event_time_s,
        });
    }

    // Use full frequency range starting from 0 Hz with static Y-cap.
    // This shows throttle-dominated low frequencies (0-10 Hz) and motor diagnostics (10+Hz).
    let nyquist_freq = sr_value / NYQUIST_DIVISOR;

    // Pre-filter each motor's spectrum to the plotted frequency range before deciding whether
    // there's anything to plot at all — "skip" means the file is never written, not written
    // then deleted. The draw loop below reuses this same cached, filtered data.
    let motor_plot_data: Vec<Option<Vec<(f64, f64)>>> = motor_spectrums
        .iter()
        .map(|spectrum_data| {
            spectrum_data
                .as_ref()
                .and_then(|(frequencies, amplitudes, _)| {
                    let filtered: Vec<(f64, f64)> = frequencies
                        .iter()
                        .zip(amplitudes.iter())
                        .take_while(|(f, _)| **f <= nyquist_freq)
                        .map(|(f, a)| (*f, *a))
                        .collect();
                    if filtered.is_empty() {
                        None
                    } else {
                        Some(filtered)
                    }
                })
        })
        .collect();

    if motor_plot_data.iter().all(Option::is_none) {
        println!("  ⚠️  Skipping Motor Spectrums: no motor data to plot.");
        return Ok(motor_osc_results);
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

    for (motor_idx, filtered_data) in motor_plot_data.iter().enumerate().take(motor_count) {
        let area = &sub_plot_areas[motor_idx];

        if let Some(filtered_data) = filtered_data {
            // Use static Y-axis range for standardized comparison across copters.
            // MOTOR_SPECTRUM_Y_AXIS_MAX provides consistent visual scaling and future-proofs against outliers.
            let y_range = MOTOR_SPECTRUM_AXIS_ORIGIN..crate::constants::MOTOR_SPECTRUM_Y_AXIS_MAX;

            // X-axis: show full range from 0 Hz (includes throttle and motor data)
            let x_max = filtered_data
                .last()
                .map(|(f, _)| *f)
                .unwrap_or(nyquist_freq);
            let x_range = MOTOR_SPECTRUM_AXIS_ORIGIN..x_max;

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
                .y_label_formatter(&|y| {
                    // Show tenths for Y-axis if max is < 5, otherwise show integers
                    if crate::constants::MOTOR_SPECTRUM_Y_AXIS_MAX
                        < MOTOR_SPECTRUM_Y_LABEL_PRECISION_THRESHOLD
                    {
                        format!("{:.1}", y)
                    } else {
                        format!("{:.0}", y)
                    }
                })
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
        }
    }

    root_area.present()?;
    println!("  Stacked plot saved as '{}'.", output_file);

    Ok(motor_osc_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SAMPLE_RATE: f64 = 2000.0;
    const TEST_BURST_HZ: f64 = 100.0; // mid-band, well inside 50-200 Hz
    const TEST_BURST_AMPLITUDE: f32 = 100.0; // raw motor-command units

    fn constant_series(n: usize) -> Vec<f32> {
        vec![1500.0; n]
    }

    fn times_for(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / TEST_SAMPLE_RATE).collect()
    }

    #[test]
    fn quiet_series_reports_no_oscillation() {
        // 10 seconds of a perfectly flat motor command — no window should ever cross the bar.
        let n = (10.0 * TEST_SAMPLE_RATE) as usize;
        let samples = constant_series(n);
        let times = times_for(n);

        let result = detect_windowed_oscillation(&samples, &times, TEST_SAMPLE_RATE);
        assert!(result.is_none());
    }

    #[test]
    fn brief_burst_amid_long_quiet_flight_is_detected() {
        // A ~0.3s, 100 Hz burst embedded in 30 seconds of otherwise flat command — under 1% of
        // total flight duration. A single whole-log FFT dilutes this well below the detection
        // bar (confirmed against two real-world desync logs during development); the sliding
        // window must still catch it.
        let total_s = 30.0;
        let n = (total_s * TEST_SAMPLE_RATE) as usize;
        let mut samples = constant_series(n);
        let times = times_for(n);

        let burst_start = n / 2;
        let burst_len = (0.3 * TEST_SAMPLE_RATE) as usize;
        for (i, sample) in samples
            .iter_mut()
            .enumerate()
            .skip(burst_start)
            .take(burst_len)
        {
            let t = i as f64 / TEST_SAMPLE_RATE;
            *sample += TEST_BURST_AMPLITUDE
                * (2.0 * std::f64::consts::PI * TEST_BURST_HZ * t).sin() as f32;
        }

        let result = detect_windowed_oscillation(&samples, &times, TEST_SAMPLE_RATE);
        let (event_time, peak, avg) = result.expect("burst must be detected");
        assert!(peak > MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD);
        assert!(peak > MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER * avg);
        // Event time must land inside the burst window, not anywhere in the 30s flight.
        let burst_start_s = burst_start as f64 / TEST_SAMPLE_RATE;
        let burst_end_s = (burst_start + burst_len) as f64 / TEST_SAMPLE_RATE;
        assert!(event_time >= burst_start_s - MOTOR_OSCILLATION_WINDOW_S);
        assert!(event_time <= burst_end_s);
    }

    #[test]
    fn burst_confined_to_stride_misaligned_tail_is_still_detected() {
        // Regression case: the sliding window's stride (half the window length) doesn't
        // necessarily land exactly on the series' true final window when the series length
        // isn't an exact multiple of the stride — that final window must still be checked
        // explicitly. `last_start` is deliberately chosen not a multiple of `hop`, so this
        // only passes if the tail-window fix's explicit final check actually runs.
        let win_samples =
            ((MOTOR_OSCILLATION_WINDOW_S * TEST_SAMPLE_RATE).round() as usize).max(MIN_FFT_SAMPLES);
        let hop = ((win_samples as f64 * MOTOR_OSCILLATION_HOP_FRACTION).round() as usize).max(1);
        // Any last_start not a multiple of hop works; +125 keeps it stride-misaligned (125 %
        // 250 != 0) while adding only a small, arbitrary remainder past a whole-second mark.
        let last_start = (10.0 * TEST_SAMPLE_RATE) as usize + 125;
        assert_ne!(
            last_start % hop,
            0,
            "test setup must be stride-misaligned to be meaningful"
        );
        let n = last_start + win_samples;

        let mut samples = constant_series(n);
        let times = times_for(n);

        // Burst fills the entire final window (matching the full-window-coverage pattern the
        // FFT needs for a clean peak/avg ratio — a burst narrower than the window it's
        // analyzed in leaks energy across bins and can fail the ratio check regardless of
        // amplitude, which is a signal-processing property, not a bug in the tail-window fix).
        for (i, sample) in samples.iter_mut().enumerate().skip(last_start) {
            let t = i as f64 / TEST_SAMPLE_RATE;
            *sample += TEST_BURST_AMPLITUDE
                * (2.0 * std::f64::consts::PI * TEST_BURST_HZ * t).sin() as f32;
        }

        let result = detect_windowed_oscillation(&samples, &times, TEST_SAMPLE_RATE);
        let (event_time, peak, avg) = result.expect("tail burst must still be detected");
        assert!(peak > MOTOR_OSCILLATION_ABSOLUTE_THRESHOLD);
        assert!(peak > MOTOR_OSCILLATION_THRESHOLD_MULTIPLIER * avg);
        assert!((event_time - last_start as f64 / TEST_SAMPLE_RATE).abs() < f64::EPSILON);
    }

    #[test]
    fn series_shorter_than_one_window_reports_none() {
        let n = 10; // far below MIN_FFT_SAMPLES and one window's worth of samples
        let samples = constant_series(n);
        let times = times_for(n);

        let result = detect_windowed_oscillation(&samples, &times, TEST_SAMPLE_RATE);
        assert!(result.is_none());
    }
}
