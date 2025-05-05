// src/fft_utils.rs

use ndarray::Array1;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
/// Returns the complex frequency spectrum. Handles empty input.
/// Input is Array1<f32>, output is Array1<Complex32>.
pub fn fft_forward(data: &Array1<f32>) -> Array1<Complex32> {
    let n = data.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    // RealFFT requires input buffer size to be a power of 2 for optimal performance,
    // but it can handle any size. Padding might be done internally or required by plan.
    // The planner should handle the buffer size correctly.

    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(n); // Plan for the actual data length

    let mut input = data.to_vec();
    let mut output = plan.make_output_vec();

    if plan.process(&mut input, &mut output).is_err() {
         eprintln!("Warning: FFT forward processing failed.");
         // Return zeros with the expected output length for consistency
         let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
         return Array1::zeros(expected_complex_len);
    }
    Array1::from(output)
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of a complex spectrum.
/// Returns the reconstructed real-valued signal. Requires the original signal length N.
/// Normalizes the output. Handles empty input or length mismatches.
/// Input is Array1<Complex32>, output is Array1<f32>.
pub fn fft_inverse(data: &Array1<Complex32>, original_length_n: usize) -> Array1<f32> {
    let complex_len = data.len();
     if complex_len == 0 || original_length_n == 0 {
        return Array1::zeros(original_length_n);
    }

    let expected_complex_len = if original_length_n % 2 == 0 {
        original_length_n / 2 + 1
    } else {
        (original_length_n + 1) / 2
    };

    if complex_len != expected_complex_len {
        eprintln!(
            "Warning: FFT inverse length mismatch. Expected complex length {}, got {} for original length {}. Returning zeros.",
            expected_complex_len,
            complex_len,
            original_length_n
        );
        return Array1::zeros(original_length_n);
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_inverse(original_length_n); // Plan for the target real length

    let mut input = data.to_vec();
    let mut output = plan.make_output_vec();

    if plan.process(&mut input, &mut output).is_ok() {
        // realfft's inverse does not normalize, so we divide by N
        let scale = 1.0 / original_length_n as f32;
        let mut output_arr = Array1::from(output);
        output_arr.mapv_inplace(|x| x * scale);
        output_arr
    } else {
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        Array1::zeros(original_length_n)
    }
}

/// Calculates the frequencies for the real FFT output.
/// Given N samples and sample period d (1/sample_rate), returns frequencies up to Nyquist.
/// Output is Array1<f64> for consistency with time.
#[allow(dead_code)] // Keep available even if not currently used directly by spectrograph
pub fn fft_rfftfreq(n: usize, d: f64) -> Array1<f64> {
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let nyquist = 0.5 / d;
    Array1::linspace(0.0, nyquist, num_freqs)
}

// src/fft_utils.rs
