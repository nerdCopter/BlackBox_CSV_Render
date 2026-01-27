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
}
