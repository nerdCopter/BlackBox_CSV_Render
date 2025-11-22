// src/font_config.rs

// Global font style constants for plot rendering.
// All font styles are centralized here to ensure consistency across the entire project
// and make it easy to update the font configuration in the future.

use crate::constants::{
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_CHART_TITLE, FONT_SIZE_LEGEND, FONT_SIZE_MAIN_TITLE,
    FONT_SIZE_MESSAGE, FONT_SIZE_PEAK_LABEL,
};

/// Embedded monospace font at compile time
pub static BUNDLED_FONT_BYTES: &[u8] = include_bytes!("../fonts/DejaVuSansMono.ttf");

/// Font family name for the bundled font
pub const FONT_FAMILY_BUNDLED: &str = "DejaVu Sans Mono";

/// Font family name for default system fonts (used by plotters for backwards compatibility)
/// When plotters renders with "sans-serif", it uses system fonts
pub const FONT_FAMILY_SYSTEM: &str = "sans-serif";

/// Represents a font style (family + size) for consistent usage throughout the application
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct FontStyle {
    pub family: &'static str,
    pub size: i32,
}

// Global font style constants for all text elements in the plots
#[allow(dead_code)]
pub const FONT_MAIN_TITLE: FontStyle = FontStyle {
    family: FONT_FAMILY_SYSTEM,
    size: FONT_SIZE_MAIN_TITLE,
};

#[allow(dead_code)]
pub const FONT_CHART_TITLE: FontStyle = FontStyle {
    family: FONT_FAMILY_SYSTEM,
    size: FONT_SIZE_CHART_TITLE,
};

#[allow(dead_code)]
pub const FONT_AXIS_LABEL: FontStyle = FontStyle {
    family: FONT_FAMILY_SYSTEM,
    size: FONT_SIZE_AXIS_LABEL,
};

#[allow(dead_code)]
pub const FONT_LEGEND: FontStyle = FontStyle {
    family: FONT_FAMILY_SYSTEM,
    size: FONT_SIZE_LEGEND,
};

#[allow(dead_code)]
pub const FONT_PEAK_LABEL: FontStyle = FontStyle {
    family: FONT_FAMILY_BUNDLED,
    size: FONT_SIZE_PEAK_LABEL,
};

#[allow(dead_code)]
pub const FONT_MESSAGE: FontStyle = FontStyle {
    family: FONT_FAMILY_SYSTEM,
    size: FONT_SIZE_MESSAGE,
};

// Tuple representations for use with plotters' IntoFont trait
// These are convenient for direct use with plotters methods like `.caption()` and `.label_style()`
pub const FONT_TUPLE_MAIN_TITLE: (&str, i32) = (FONT_FAMILY_SYSTEM, FONT_SIZE_MAIN_TITLE);
pub const FONT_TUPLE_CHART_TITLE: (&str, i32) = (FONT_FAMILY_SYSTEM, FONT_SIZE_CHART_TITLE);
pub const FONT_TUPLE_AXIS_LABEL: (&str, i32) = (FONT_FAMILY_SYSTEM, FONT_SIZE_AXIS_LABEL);
pub const FONT_TUPLE_LEGEND: (&str, i32) = (FONT_FAMILY_SYSTEM, FONT_SIZE_LEGEND);
pub const FONT_TUPLE_PEAK_LABEL: (&str, i32) = (FONT_FAMILY_BUNDLED, FONT_SIZE_PEAK_LABEL);
pub const FONT_TUPLE_MESSAGE: (&str, i32) = (FONT_FAMILY_SYSTEM, FONT_SIZE_MESSAGE);
