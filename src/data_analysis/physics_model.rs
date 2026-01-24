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

        let diameter: u8 = s[0..2]
            .parse()
            .map_err(|_| "Invalid stator diameter (first 2 digits)")?;

        let height_str = &s[2..];
        let height: f32 = if height_str.contains('.') {
            // Format: "2306.5"
            height_str
                .parse()
                .map_err(|_| "Invalid stator height (fractional)")?
        } else if height_str.len() == 2 {
            // Format: "2207"
            height_str
                .parse::<u8>()
                .map(|h| h as f32)
                .map_err(|_| "Invalid stator height (2 digits)")?
        } else if height_str.len() == 3 {
            // Format: "23065" → interpret as 2306.5
            let whole = &height_str[0..2];
            let frac = &height_str[2..3];
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
    #[allow(dead_code)]
    pub fn torque_constant(&self) -> f64 {
        60.0 / (2.0 * PI * self.kv as f64)
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
    pub fn asymmetry_ratio(&self) -> f64 {
        self.arm_length_diagonal_mm / self.arm_length_width_mm
    }
}

/// Complete quadcopter physical model
#[derive(Debug, Clone)]
pub struct QuadcopterPhysics {
    pub geometry: FrameGeometry,
    pub motor_spec: MotorSpec,
    pub prop_diameter_inch: f32, // Supports decimal sizes (5.1", 6.5", etc.)
    pub battery_mass_g: f64,
    pub frame_mass_g: f64,
    pub central_components_mass_g: f64,
}

impl QuadcopterPhysics {
    /// Calculate rotational inertia for specific axis
    /// I = Σ(mᵢ × rᵢ²) for each component
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    pub fn calculate_rotational_inertia(&self, axis: usize) -> f64 {
        let arm_length_m = self.geometry.arm_length_for_axis(axis) / 1000.0;

        // 4 motors at arm tips: I = 4 × m_motor × r²
        let motor_mass_kg = self.motor_spec.estimated_mass_g() / 1000.0;
        let i_motors = 4.0 * motor_mass_kg * arm_length_m.powi(2);

        // 4 props at arm tips: I = 4 × m_prop × r²
        // Prop mass scales with area (diameter²)
        let prop_mass_kg = ((self.prop_diameter_inch as f64 / 5.0).powi(2) * 5.0) / 1000.0;
        let i_props = 4.0 * prop_mass_kg * arm_length_m.powi(2);

        // Frame arms (4 uniform rods from center to tip): I = 4 × (1/3) × m_arm × r²
        // Assume frame mass distributed equally across 4 arms
        let arm_mass_kg = (self.frame_mass_g / 4.0) / 1000.0;
        let i_frame_arms = 4.0 * (1.0 / 3.0) * arm_mass_kg * arm_length_m.powi(2);

        // Central components (FC, ESC, VTX, camera) at rotation center
        // Negligible contribution to rotational inertia (r ≈ 0)
        let _i_central = 0.0;

        // Battery (rear-mounted, typically 30mm behind center for COG balance)
        let battery_offset_m: f64 = 0.030;
        let battery_mass_kg = self.battery_mass_g / 1000.0;
        let i_battery = battery_mass_kg * battery_offset_m.powi(2);

        i_motors + i_props + i_frame_arms + i_battery
    }

    /// Calculate expected Td (time to 50%) for given P gain
    /// Td = (π/2) × √(I/P)
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    pub fn calculate_expected_td_ms(&self, current_p_gain: f64, axis: usize) -> f64 {
        let inertia = self.calculate_rotational_inertia(axis);
        let omega_n = (current_p_gain / inertia).sqrt();
        let td_seconds = PI / (2.0 * omega_n);
        td_seconds * 1000.0 // Convert to milliseconds
    }

    /// Calculate optimal P gain for target Td
    /// P = I × (π / (2 × Td))²
    /// axis: 0=Roll, 1=Pitch, 2=Yaw
    pub fn calculate_optimal_p_for_target_td(&self, target_td_ms: f64, axis: usize) -> f64 {
        let inertia = self.calculate_rotational_inertia(axis);
        let target_td_s = target_td_ms / 1000.0;
        let omega_n_target = PI / (2.0 * target_td_s);
        inertia * omega_n_target.powi(2)
    }

    /// Estimate total mass from components
    pub fn estimated_total_mass_g(&self) -> f64 {
        let motors_total = self.motor_spec.estimated_mass_g() * 4.0;
        let props_total = ((self.prop_diameter_inch as f64 / 5.0).powi(2) * 5.0) * 4.0;

        motors_total
            + props_total
            + self.frame_mass_g
            + self.battery_mass_g
            + self.central_components_mass_g
    }
}

/// Builder for quadcopter physics with sensible defaults
pub struct QuadcopterPhysicsBuilder {
    geometry: Option<FrameGeometry>,
    motor_spec: Option<MotorSpec>,
    prop_diameter_inch: Option<f32>,
    lipo_cells: Option<u8>,
    battery_mass_g: Option<f64>,
    frame_mass_g: Option<f64>,
    central_components_mass_g: Option<f64>,
}

impl QuadcopterPhysicsBuilder {
    pub fn new() -> Self {
        Self {
            geometry: None,
            motor_spec: None,
            prop_diameter_inch: None,
            lipo_cells: None,
            battery_mass_g: None,
            frame_mass_g: None,
            central_components_mass_g: None,
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

    pub fn lipo_cells(mut self, cells: u8) -> Self {
        self.lipo_cells = Some(cells);
        self
    }

    pub fn battery_mass_g(mut self, mass: f64) -> Self {
        self.battery_mass_g = Some(mass);
        self
    }

    pub fn frame_mass_g(mut self, mass: f64) -> Self {
        self.frame_mass_g = Some(mass);
        self
    }

    pub fn central_components_mass_g(mut self, mass: f64) -> Self {
        self.central_components_mass_g = Some(mass);
        self
    }

    /// Build with automatic estimation of missing parameters
    pub fn build(self) -> Result<QuadcopterPhysics, String> {
        let geometry = self
            .geometry
            .ok_or("Frame geometry is required (arm lengths)")?;
        let motor_spec = self.motor_spec.ok_or("Motor specification is required")?;
        let prop_diameter_inch = self.prop_diameter_inch.ok_or("Prop diameter is required")?;

        // Estimate battery mass if not provided
        let battery_mass_g = if let Some(mass) = self.battery_mass_g {
            mass
        } else if let Some(cells) = self.lipo_cells {
            estimate_battery_mass(cells, prop_diameter_inch)
        } else {
            return Err("Either battery mass or LiPo cell count is required".to_string());
        };

        // Estimate frame mass if not provided
        let frame_mass_g = self.frame_mass_g.unwrap_or_else(|| {
            estimate_frame_mass(
                geometry.arm_length_diagonal_mm,
                geometry.arm_length_width_mm,
            )
        });

        // Estimate central components mass if not provided
        let central_components_mass_g = self
            .central_components_mass_g
            .unwrap_or_else(|| estimate_central_components_mass(prop_diameter_inch));

        Ok(QuadcopterPhysics {
            geometry,
            motor_spec,
            prop_diameter_inch,
            battery_mass_g,
            frame_mass_g,
            central_components_mass_g,
        })
    }
}

impl Default for QuadcopterPhysicsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate battery mass from cell count and prop size
/// Uses empirical data from common battery configurations
fn estimate_battery_mass(cells: u8, prop_size_inch: f32) -> f64 {
    let prop_size = prop_size_inch.round() as u8;
    // Typical capacity by cell count and application
    let typical_capacity_mah = match (cells, prop_size) {
        (1, _) => 400.0,       // Tiny whoop
        (2, _) => 450.0,       // 2" micro
        (3, _) => 550.0,       // 3" toothpick
        (4, 1..=3) => 650.0,   // Small 4S
        (4, 4..=5) => 1500.0,  // 5" 4S freestyle
        (4, 6..=7) => 1800.0,  // 7" 4S long-range
        (6, 1..=4) => 850.0,   // Small 6S
        (6, 5) => 1300.0,      // 5" 6S racing/freestyle
        (6, 6..=7) => 1800.0,  // 7" 6S long-range
        (6, 8..=10) => 3500.0, // 10" 6S cinelifter
        (12, _) => 5000.0,     // 12S heavy-lift
        _ => 1500.0,           // Default
    };

    // Mass estimation: ~0.12 g per mAh (typical LiPo energy density ~140 Wh/kg)
    typical_capacity_mah * 0.12
}

/// Estimate frame mass from arm lengths
/// Larger frames with longer arms are heavier
fn estimate_frame_mass(arm_length_diagonal_mm: f64, arm_length_width_mm: f64) -> f64 {
    let avg_arm_length = (arm_length_diagonal_mm + arm_length_width_mm) / 2.0;
    // Empirical formula: frame mass ≈ 15g base + 0.35g per mm of arm length
    15.0 + (avg_arm_length * 0.35)
}

/// Estimate central component mass (FC, ESC, VTX, camera, RX, wiring)
/// Scales slightly with quad size
fn estimate_central_components_mass(prop_size_inch: f32) -> f64 {
    let prop_size = prop_size_inch.round() as u8;
    match prop_size {
        1..=2 => 10.0,   // Tiny AIO boards
        3..=4 => 40.0,   // Lightweight stack
        5 => 70.0,       // Standard 5" stack
        6..=7 => 90.0,   // Long-range with GPS
        8..=10 => 120.0, // Cinelifter with HD system
        _ => 150.0,      // Heavy-lift
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
        assert!((geom.asymmetry_ratio() - 1.31).abs() < 0.01);
    }

    #[test]
    fn test_physics_builder_with_estimates() {
        let geom = FrameGeometry::from_motor_measurements(450.0, 450.0);
        let mut motor = MotorSpec::from_string("2207").unwrap();
        motor.kv = 2400;

        let physics = QuadcopterPhysicsBuilder::new()
            .geometry(geom)
            .motor_spec(motor)
            .prop_diameter_inch(5.0)
            .lipo_cells(4)
            .build()
            .unwrap();

        // Check that estimates were applied
        assert!(physics.battery_mass_g > 100.0); // 4S battery
        assert!(physics.frame_mass_g > 50.0); // Frame
        assert!(physics.central_components_mass_g > 30.0); // Electronics

        let total = physics.estimated_total_mass_g();
        println!("Total estimated mass: {:.1}g", total);
        assert!(
            total > 300.0 && total < 800.0,
            "5\" 4S should be 300-800g, got {:.1}g",
            total
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
            .lipo_cells(4)
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
            .lipo_cells(6)
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
