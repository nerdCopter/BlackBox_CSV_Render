/// Centralized axis naming utilities
///
/// Provides consistent axis names across all plot functions and data analysis modules.
/// Get the standard axis name for a given index
///
/// # Arguments
/// * `index` - Axis index (0=Roll, 1=Pitch, 2=Yaw)
///
/// # Returns
/// Static string slice with the axis name
///
/// # Panics
/// Panics if index is greater than 2
#[allow(dead_code)]
pub fn axis_name(index: usize) -> &'static str {
    match index {
        0 => "Roll",
        1 => "Pitch",
        2 => "Yaw",
        _ => panic!(
            "Invalid axis index: {}. Expected 0 (Roll), 1 (Pitch), or 2 (Yaw)",
            index
        ),
    }
}

/// Get all axis names as a static array
pub const AXIS_NAMES: [&str; 3] = ["Roll", "Pitch", "Yaw"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_name() {
        assert_eq!(axis_name(0), "Roll");
        assert_eq!(axis_name(1), "Pitch");
        assert_eq!(axis_name(2), "Yaw");
    }

    #[test]
    #[should_panic(expected = "Invalid axis index")]
    fn test_axis_name_panic() {
        axis_name(3);
    }

    #[test]
    fn test_axis_names_constant() {
        assert_eq!(AXIS_NAMES[0], "Roll");
        assert_eq!(AXIS_NAMES[1], "Pitch");
        assert_eq!(AXIS_NAMES[2], "Yaw");
    }
}
