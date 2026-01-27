#[cfg(test)]
mod tests {
    use crate::data_analysis::optimal_p_estimation::{FrameClass, TdTargetSpec};

    #[test]
    fn td_target_spec_out_of_range_returns_none() {
        assert!(TdTargetSpec::for_frame_inches(0).is_none());
        assert!(TdTargetSpec::for_frame_inches(16).is_none());
    }

    #[test]
    fn frame_class_td_target_is_some_for_valid_classes() {
        assert!(FrameClass::OneInch.td_target().is_some());
        assert!(FrameClass::FiveInch.td_target().is_some());
        assert!(FrameClass::FifteenInch.td_target().is_some());
    }

    #[test]
    fn td_target_spec_valid_range_returns_some() {
        // Representative in-range values should return Some(TdTargetSpec)
        assert!(TdTargetSpec::for_frame_inches(1).is_some());
        assert!(TdTargetSpec::for_frame_inches(5).is_some());
        assert!(TdTargetSpec::for_frame_inches(15).is_some());
    }

    #[test]
    fn td_target_spec_returns_expected_values() {
        // Check FrameClass td_target matches constants for key sizes
        let (t1, tol1) = FrameClass::OneInch.td_target().unwrap();
        let spec1 = TdTargetSpec::for_frame_inches(1).unwrap();
        assert_eq!(t1, spec1.target_ms);
        assert_eq!(tol1, spec1.tolerance_ms);

        let (t5, tol5) = FrameClass::FiveInch.td_target().unwrap();
        let spec5 = TdTargetSpec::for_frame_inches(5).unwrap();
        assert_eq!(t5, spec5.target_ms);
        assert_eq!(tol5, spec5.tolerance_ms);

        let (t15, tol15) = FrameClass::FifteenInch.td_target().unwrap();
        let spec15 = TdTargetSpec::for_frame_inches(15).unwrap();
        assert_eq!(t15, spec15.target_ms);
        assert_eq!(tol15, spec15.tolerance_ms);
    }
}
