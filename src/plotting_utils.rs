// src/plotting_utils.rs

use plotters::prelude::*;
use std::error::Error;

/// Calculate plot range with padding.
/// Adds 15% padding, or a fixed padding for very small ranges.
pub fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

/// Draw a "Data Unavailable" message on a plot area.
pub fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
) -> Result<(), Box<dyn Error>> {
    let message = format!("Axis {} {} Data Unavailable", axis_index, plot_type);
    area.draw(&Text::new(
        message,
        (50, 50),
        ("sans-serif", 20).into_font().color(&RED),
    ))?;
    Ok(())
}