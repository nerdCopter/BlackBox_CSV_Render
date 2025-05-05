// src/fft_utils.rs

use ndarray::Array1;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
/// Returns the complex frequency spectrum. Handles empty input.
pub fn fft_forward(data: &Array1<f32>) -> Array1<Complex32> {
    if data.is_empty() {
        return Array1::zeros(0);
    }
    let n = data.len();
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut output = planner.make_output_vec();
    if planner.process(&mut input, &mut output).is_err() {
         eprintln!("Warning: FFT forward processing failed.");
         let expected_complex_len = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
         return Array1::zeros(expected_complex_len);
    }
    Array1::from(output)
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of a complex spectrum.
/// Returns the reconstructed real-valued signal. Requires the original signal length N.
/// Normalizes the output. Handles empty input or length mismatches.
pub fn fft_inverse(data: &Array1<Complex32>, original_length_n: usize) -> Array1<f32> {
    if data.is_empty() || original_length_n == 0 {
        return Array1::zeros(original_length_n);
    }
    let mut input = data.to_vec();
    let planner = RealFftPlanner::<f32>::new().plan_fft_inverse(original_length_n);
    let mut output = planner.make_output_vec();

    let expected_complex_len = if original_length_n % 2 == 0 {
        original_length_n / 2 + 1
    } else {
        (original_length_n + 1) / 2
    };

    if input.len() != expected_complex_len {
        eprintln!(
            "Warning: FFT inverse length mismatch. Expected complex length {}, got {}. Returning zeros.",
            expected_complex_len,
            input.len()
        );
        return Array1::zeros(original_length_n);
    }

    if planner.process(&mut input, &mut output).is_ok() {
        let scale = 1.0 / original_length_n as f32;
        let mut output_arr = Array1::from(output);
        output_arr.mapv_inplace(|x| x * scale);
        output_arr
    } else {
        eprintln!("Warning: FFT inverse processing failed. Returning zeros.");
        Array1::zeros(original_length_n)
    }
}

/// Calculates the frequencies for the real FFT output. (Currently unused in deconvolution logic)
#[allow(dead_code)]
pub fn fft_rfftfreq(n: usize, d: f32) -> Array1<f32> {
    if n == 0 || d <= 0.0 {
        return Array1::zeros(0);
    }
    let num_freqs = if n % 2 == 0 { n / 2 + 1 } else { (n + 1) / 2 };
    let mut freqs = Array1::<f32>::zeros(num_freqs);
    let nyquist = 0.5 / d;
    for i in 0..num_freqs {
        freqs[i] = i as f32 * nyquist / (num_freqs - 1) as f32;
    }
    freqs
}