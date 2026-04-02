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
    AXIS_NAMES.get(index).copied().unwrap_or_else(|| {
        panic!(
            "Invalid axis index: {}. Expected 0 (Roll), 1 (Pitch), or 2 (Yaw)",
            index
        )
    })
}

/// Number of axes (Roll, Pitch, Yaw)
pub const AXIS_COUNT: usize = 3;

/// Number of primary control axes (Roll, Pitch only - excludes Yaw)
pub const ROLL_PITCH_AXIS_COUNT: usize = 2;

/// Get all axis names as a static array
pub const AXIS_NAMES: [&str; AXIS_COUNT] = ["Roll", "Pitch", "Yaw"];

// Compile-time assertions to prevent invariant drift
const _: [(); AXIS_COUNT] = [(); AXIS_NAMES.len()];
const _: () = assert!(
    ROLL_PITCH_AXIS_COUNT > 0 && ROLL_PITCH_AXIS_COUNT < AXIS_COUNT,
    "ROLL_PITCH_AXIS_COUNT must be > 0 and < AXIS_COUNT"
);

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
