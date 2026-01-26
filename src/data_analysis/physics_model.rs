// src/data_analysis/physics_model.rs
//
// Physics-based quadcopter modeling for Td target calculation
// Calculates rotational inertia from physical parameters and predicts optimal response time

#![allow(dead_code)] // Module not yet integrated into main - will be used in Phase 2

use std::f64::consts::PI;

/// Motor specification with stator dimensions and KV rating
#[derive(Debug, Clone, Copy)]
pub struct MotorSpec {
    pub stator_diameter_mm: u8,
    pub stator_height_mm: f32, // Can be fractional (e.g., 6.5)
    pub kv: u16,
}

impl MotorSpec {
    /// Parse motor size from string (e.g., "2207", "2306.5")
    pub fn from_string(s: &str) -> Result<Self, String> {
        if s.len() < 4 {
            return Err("Motor size must be at least 4 characters".to_string());
        }

        // Validate ASCII-only input to avoid UTF-8 multi-byte issues
        if !s.is_ascii() {
            return Err("Motor size must contain only ASCII characters".to_string());
        }

        // Extract first two characters for diameter (char-safe)
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < 4 {
            return Err("Motor size must be at least 4 characters".to_string());
        }

        let diameter_str: String = chars[0..2].iter().collect();
        let diameter: u8 = diameter_str
            .parse()
            .map_err(|_| "Invalid stator diameter (first 2 digits)")?;

        // Extract remaining characters for height (char-safe)
        let height_chars: String = chars[2..].iter().collect();

        let height: f32 = if height_chars.contains('.') {
            // Format: "2306.5"
            height_chars
                .parse()
                .map_err(|_| "Invalid stator height (fractional)")?
        } else if height_chars.len() == 2 {
            // Format: "2207"
            height_chars
                .parse::<u8>()
                .map(|h| h as f32)
                .map_err(|_| "Invalid stator height (2 digits)")?
        } else if height_chars.len() == 3 {
            // Format: "23065" → interpret as 2306.5
            let whole: String = chars[2..4].iter().collect();
            let frac: String = chars[4..5].iter().collect();
            format!("{}.{}", whole, frac)
                .parse()
                .map_err(|_| "Invalid stator height (3 digits)")?
        } else {
            return Err("Invalid motor size format (expected XXYY or XXYY.Z)".to_string());
        };

        Ok(Self {
            stator_diameter_mm: diameter,
            stator_height_mm: height,
            kv: 0, // Must be set separately
        })
    }

    /// Estimate motor mass from stator dimensions
    /// Uses empirical formula: mass ≈ k × diameter² × height
    pub fn estimated_mass_g(&self) -> f64 {
        let volume = (self.stator_diameter_mm as f64).powi(2) * (self.stator_height_mm as f64);
        // Empirical constant calibrated from actual motor masses
        // Accounts for copper windings, magnets, bell, shaft, bearings
        0.0088 * volume
    }

    /// Calculate motor torque constant (Kt) in N·m/A
    /// Kt = 60/(2π·KV)
    ///
    /// Returns None if kv is zero (from_string may leave kv uninitialized at 0)
    #[allow(dead_code)]
    pub fn torque_constant(&self) -> Option<f64> {
        if self.kv == 0 {
            None
        } else {
            Some(60.0 / (2.0 * PI * self.kv as f64))
        }
    }
}

/// Frame geometry with motor positions
/// Motor layout (looking down on quad):
///       Front
///   M2(FR) - M4(FL)
///      \  FC  /
///   M1(RR) - M3(RL)
///       Rear
///
/// Supports asymmetric frames (stretched-X, squashed-X)
#[derive(Debug, Clone, Copy)]
pub struct FrameGeometry {
    /// Distance from FC to motor along M1→M4 diagonal (mm)
    /// This is the true diagonal: rear-right to front-left
    pub arm_length_diagonal_mm: f64,

    /// Distance from FC to motor along M1→M3 width (mm)
    /// This is side-to-side (rear-right to rear-left), NOT diagonal
    pub arm_length_width_mm: f64,
}

impl FrameGeometry {
    /// Create from motor-to-motor measurements
    /// diagonal_mm: M1→M4 (rear-right to front-left, true diagonal)
    /// width_mm: M1→M3 (rear-right to rear-left, side-to-side)
    /// Pilot measures motor shaft to motor shaft, tool divides by 2
    pub fn from_motor_measurements(diagonal_mm: f64, width_mm: f64) -> Self {
        Self {
            arm_length_diagonal_mm: diagonal_mm / 2.0,
            arm_length_width_mm: width_mm / 2.0,
        }
    }

    /// Get arm length for specific axis
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    pub fn arm_length_for_axis(&self, axis: usize) -> f64 {
        match axis {
            0 => self.arm_length_width_mm,    // Roll: Side-to-side (width)
            1 => self.arm_length_diagonal_mm, // Pitch: Diagonal (front-to-back)
            2 => {
                // Yaw uses average (all motors contribute equally)
                (self.arm_length_diagonal_mm + self.arm_length_width_mm) / 2.0
            }
            _ => self.arm_length_diagonal_mm, // Default to pitch
        }
    }

    /// Check if frame is symmetric (true-X)
    pub fn is_symmetric(&self) -> bool {
        (self.arm_length_diagonal_mm - self.arm_length_width_mm).abs() < 2.0 // Within 2mm
    }

    /// Get asymmetry ratio (diagonal / width)
    /// Returns None if width is near zero (would cause division by zero)
    pub fn asymmetry_ratio(&self) -> Option<f64> {
        if self.arm_length_width_mm.abs() <= f64::EPSILON {
            None
        } else {
            Some(self.arm_length_diagonal_mm / self.arm_length_width_mm)
        }
    }
}

/// Complete quadcopter physical model
#[derive(Debug, Clone)]
pub struct QuadcopterPhysics {
    pub geometry: FrameGeometry,
    pub motor_spec: MotorSpec,
    pub prop_diameter_inch: f32,
    pub prop_pitch_inch: f32, // Propeller pitch (affects aerodynamic loading)
    pub total_mass_g: f64,    // All-up weight (everything that flies)
}

impl QuadcopterPhysics {
    /// Calculate rotational inertia for specific axis using mass distribution
    /// I = Σ(mᵢ × rᵢ²) for each component
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    ///
    /// NOTE: Pilot provides total weight via --weight (scale reading in grams).
    /// In physics: weight = force (mass × gravity), but scale readings represent mass.
    /// We distribute total_mass_g across components using constants from src/constants.rs
    pub fn calculate_rotational_inertia(&self, axis: usize) -> f64 {
        use crate::constants::*;

        let arm_length_m = self.geometry.arm_length_for_axis(axis) / 1000.0;
        let total_mass_kg = self.total_mass_g / 1000.0;

        // Distribute total mass across components
        let motors_mass_kg = total_mass_kg * MASS_FRACTION_MOTORS;
        let props_mass_kg = total_mass_kg * MASS_FRACTION_PROPS;
        let frame_mass_kg = total_mass_kg * MASS_FRACTION_FRAME;
        let battery_mass_kg = total_mass_kg * MASS_FRACTION_BATTERY;
        // Central and misc components at r≈0 contribute negligibly to I

        // 4 motors at arm tips: I = 4 × (m_motor/4) × r² = m_motors × r²
        let i_motors = motors_mass_kg * arm_length_m.powi(2);

        // 4 props at arm tips: I = 4 × (m_props/4) × r² = m_props × r²
        let i_props = props_mass_kg * arm_length_m.powi(2);

        // Frame arms (4 uniform rods from center to tip): I = 4 × (1/3) × (m_frame/4) × r²
        // For rod rotating about end: I = (1/3)×m×L²
        let i_frame_arms = (1.0 / 3.0) * frame_mass_kg * arm_length_m.powi(2);

        // Central components (FC, ESC, VTX, camera) at rotation center: r ≈ 0, so I ≈ 0

        // Battery (rear-mounted for COG balance)
        let battery_offset_m = BATTERY_OFFSET_FROM_CENTER_MM / 1000.0;
        let i_battery = battery_mass_kg * battery_offset_m.powi(2);

        i_motors + i_props + i_frame_arms + i_battery
    }

    /// Calculate expected Td (time to 50%) for given P gain
    ///
    /// Formula: Td = (π/2) × √(I/P) / pitch_factor
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    ///
    /// **Preconditions:**
    /// - current_p_gain must be positive (> 0); returns 0.0 if invalid
    /// - inertia must be positive; returns 0.0 if invalid (prevents division by zero)
    ///
    /// Pitch loading factor: Higher pitch = more aerodynamic drag = slower actual response
    /// We DIVIDE by pitch_factor so that:
    /// - Low pitch (3.0"): expect FASTER measured Td (factor ~0.606 → Td_expected SMALLER)
    /// - Medium pitch (4.5"): baseline (factor = 1.0)
    /// - High pitch (6.0"): expect SLOWER measured Td (factor ~1.23 → Td_expected LARGER)
    ///
    /// Normalized to 4.5" pitch baseline (typical freestyle props)
    pub fn calculate_expected_td_ms(&self, current_p_gain: f64, axis: usize) -> f64 {
        if current_p_gain <= 0.0 {
            return 0.0;
        }
        let inertia = self.calculate_rotational_inertia(axis);

        // Defensive check: inertia must be positive to avoid division by zero
        if inertia <= f64::EPSILON {
            return 0.0;
        }
        debug_assert!(
            inertia > 0.0,
            "Rotational inertia must be positive, got {}",
            inertia
        );

        let omega_n = (current_p_gain / inertia).sqrt();
        let td_seconds = PI / (2.0 * omega_n);

        // Pitch loading factor: empirically tuned exponent 1.3
        // Low pitch (3.0"): faster response → larger expected Td target (factor ~0.606, divide → 1/0.606 = 1.65×)
        // Medium pitch (4.5"): baseline (factor = 1.0)
        // High pitch (6.0"): slower response → smaller expected Td target (factor ~1.23, divide → 1/1.23 = 0.81×)
        let pitch_factor = (self.prop_pitch_inch as f64 / 4.5).powf(1.3);

        // DIVIDE by pitch_factor so low-pitch props get HIGHER target (expect faster actual Td)
        (td_seconds / pitch_factor) * 1000.0 // Convert to milliseconds
    }

    /// Calculate optimal P gain for target Td
    /// P = I × (π / (2 × Td))²
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    ///
    /// Returns f64::INFINITY for non-positive target_td_ms (invalid input)
    pub fn calculate_optimal_p_for_target_td(&self, target_td_ms: f64, axis: usize) -> f64 {
        if target_td_ms <= 0.0 {
            return f64::INFINITY;
        }
        let inertia = self.calculate_rotational_inertia(axis);
        let target_td_s = target_td_ms / 1000.0;
        let omega_n_target = PI / (2.0 * target_td_s);
        inertia * omega_n_target.powi(2)
    }
}

/// Builder for quadcopter physics with sensible defaults
pub struct QuadcopterPhysicsBuilder {
    geometry: Option<FrameGeometry>,
    motor_spec: Option<MotorSpec>,
    prop_diameter_inch: Option<f32>,
    prop_pitch_inch: Option<f32>,
    total_mass_g: Option<f64>, // All-up weight (total mass including everything flown)
}

impl QuadcopterPhysicsBuilder {
    pub fn new() -> Self {
        Self {
            geometry: None,
            motor_spec: None,
            prop_diameter_inch: None,
            prop_pitch_inch: None,
            total_mass_g: None,
        }
    }

    pub fn geometry(mut self, geometry: FrameGeometry) -> Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn motor_spec(mut self, motor_spec: MotorSpec) -> Self {
        self.motor_spec = Some(motor_spec);
        self
    }

    pub fn prop_diameter_inch(mut self, diameter: f32) -> Self {
        self.prop_diameter_inch = Some(diameter);
        self
    }

    pub fn prop_pitch_inch(mut self, pitch: f32) -> Self {
        self.prop_pitch_inch = Some(pitch);
        self
    }

    pub fn total_mass_g(mut self, mass: f64) -> Self {
        self.total_mass_g = Some(mass);
        self
    }

    pub fn build(self) -> Result<QuadcopterPhysics, String> {
        // Extract all required fields
        let geometry = self
            .geometry
            .ok_or("Frame geometry is required (arm lengths)")?;
        let motor_spec = self.motor_spec.ok_or("Motor specification is required")?;
        let prop_diameter_inch = self.prop_diameter_inch.ok_or("Prop diameter is required")?;
        let total_mass_g = self
            .total_mass_g
            .ok_or("Total mass (all-up weight) is required")?;

        // Default pitch to 4.5" if not specified (typical freestyle props)
        let prop_pitch_inch = self.prop_pitch_inch.unwrap_or(4.5);

        // Validate numeric ranges
        if total_mass_g <= 0.0 {
            return Err(format!(
                "Total mass must be positive, got {:.0}g",
                total_mass_g
            ));
        }

        if prop_diameter_inch <= 0.0 {
            return Err(format!(
                "Prop diameter must be positive, got {:.1}\"",
                prop_diameter_inch
            ));
        }

        if prop_pitch_inch <= 0.0 {
            return Err(format!(
                "Prop pitch must be positive, got {:.1}\"",
                prop_pitch_inch
            ));
        }

        // Validate geometry (arm lengths)
        if geometry.arm_length_diagonal_mm <= 0.0 {
            return Err(format!(
                "Frame diagonal arm length must be positive, got {:.1}mm",
                geometry.arm_length_diagonal_mm
            ));
        }

        if geometry.arm_length_width_mm <= 0.0 {
            return Err(format!(
                "Frame width arm length must be positive, got {:.1}mm",
                geometry.arm_length_width_mm
            ));
        }

        // Validate motor spec
        if motor_spec.stator_diameter_mm == 0 {
            return Err("Motor stator diameter must be non-zero".to_string());
        }

        if motor_spec.stator_height_mm <= 0.0 {
            return Err(format!(
                "Motor stator height must be positive, got {:.1}mm",
                motor_spec.stator_height_mm
            ));
        }

        // All validations passed, construct the model
        Ok(QuadcopterPhysics {
            geometry,
            motor_spec,
            prop_diameter_inch,
            prop_pitch_inch,
            total_mass_g,
        })
    }
}

impl Default for QuadcopterPhysicsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_spec_parsing() {
        // Standard format
        let m1 = MotorSpec::from_string("2207").unwrap();
        assert_eq!(m1.stator_diameter_mm, 22);
        assert_eq!(m1.stator_height_mm, 7.0);

        // Fractional format
        let m2 = MotorSpec::from_string("2306.5").unwrap();
        assert_eq!(m2.stator_diameter_mm, 23);
        assert_eq!(m2.stator_height_mm, 6.5);

        // Alternative fractional format
        let m3 = MotorSpec::from_string("23065").unwrap();
        assert_eq!(m3.stator_diameter_mm, 23);
        assert_eq!(m3.stator_height_mm, 6.5);
    }

    #[test]
    fn test_motor_mass_estimation() {
        let mut motor = MotorSpec::from_string("2207").unwrap();
        motor.kv = 1900;
        let mass = motor.estimated_mass_g();
        println!("2207 estimated mass: {:.1}g", mass);
        assert!(
            (mass - 30.0).abs() < 8.0,
            "2207 should be ~30g, got {:.1}g",
            mass
        );

        let mut motor2 = MotorSpec::from_string("2407").unwrap();
        motor2.kv = 1700;
        let mass2 = motor2.estimated_mass_g();
        println!("2407 estimated mass: {:.1}g", mass2);
        assert!(mass2 > mass, "2407 should be heavier than 2207");
    }

    #[test]
    fn test_frame_geometry_symmetric() {
        let geom = FrameGeometry::from_motor_measurements(450.0, 450.0);
        assert!(geom.is_symmetric());
        assert_eq!(geom.arm_length_diagonal_mm, 225.0);
        assert_eq!(geom.arm_length_width_mm, 225.0);
    }

    #[test]
    fn test_frame_geometry_asymmetric() {
        // HELIOV1: M1→M4 diagonal ≈ 452mm, M1→M3 width ≈ 346mm (squashed-X)
        let geom = FrameGeometry::from_motor_measurements(452.0, 346.0);
        assert!(!geom.is_symmetric());
        assert_eq!(geom.arm_length_diagonal_mm, 226.0);
        assert_eq!(geom.arm_length_width_mm, 173.0);
        assert!((geom.asymmetry_ratio().unwrap() - 1.31).abs() < 0.01);
    }

    #[test]
    fn test_physics_builder() {
        let geom = FrameGeometry::from_motor_measurements(450.0, 450.0);
        let mut motor = MotorSpec::from_string("2207").unwrap();
        motor.kv = 2400;

        let physics = QuadcopterPhysicsBuilder::new()
            .geometry(geom)
            .motor_spec(motor)
            .prop_diameter_inch(5.0)
            .prop_pitch_inch(4.5)
            .total_mass_g(650.0) // Typical 5" 4S quad
            .build()
            .unwrap();

        assert_eq!(physics.total_mass_g, 650.0);
        assert_eq!(physics.prop_pitch_inch, 4.5);
        println!(
            "Built physics model with total mass: {}g, pitch: {}\"",
            physics.total_mass_g, physics.prop_pitch_inch
        );
    }

    #[test]
    fn test_td_calculation() {
        let geom = FrameGeometry::from_motor_measurements(450.0, 450.0);
        let mut motor = MotorSpec::from_string("2207").unwrap();
        motor.kv = 2400;

        let physics = QuadcopterPhysicsBuilder::new()
            .geometry(geom)
            .motor_spec(motor)
            .prop_diameter_inch(5.0)
            .total_mass_g(650.0) // Typical 5" 4S quad
            .build()
            .unwrap();

        let td_roll = physics.calculate_expected_td_ms(65.0, 0); // Roll
        let td_pitch = physics.calculate_expected_td_ms(65.0, 1); // Pitch

        // For symmetric frame, Roll and Pitch should be equal
        assert!((td_roll - td_pitch).abs() < 0.1);

        // Should be in reasonable range for 5" quad at P=65
        assert!(td_roll > 10.0 && td_roll < 30.0);
    }

    #[test]
    fn test_asymmetric_td_difference() {
        let geom = FrameGeometry::from_motor_measurements(452.0, 346.0);
        let mut motor = MotorSpec::from_string("2207").unwrap();
        motor.kv = 1900;

        let physics = QuadcopterPhysicsBuilder::new()
            .geometry(geom)
            .motor_spec(motor)
            .prop_diameter_inch(5.0)
            .total_mass_g(741.0) // HELIO typical weight
            .build()
            .unwrap();

        let td_roll = physics.calculate_expected_td_ms(65.0, 0); // Roll
        let td_pitch = physics.calculate_expected_td_ms(65.0, 1); // Pitch

        // Pitch should be slower due to longer arm length
        assert!(td_pitch > td_roll);

        // Difference should be ~20-30% based on geometry
        let ratio = td_pitch / td_roll;
        assert!(ratio > 1.15 && ratio < 1.40);
    }
}
