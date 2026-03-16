use chrono::{DateTime, Utc};

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionMethod {
    MachineLearning { confidence: f64 },
    RandomForest { confidence: f64, n_trees: usize },
    HistoricalAverage,
}

impl PredictionMethod {
    pub fn is_ml(&self) -> bool {
        matches!(
            self,
            PredictionMethod::MachineLearning { .. } | PredictionMethod::RandomForest { .. }
        )
    }

    pub fn confidence(&self) -> f64 {
        match self {
            PredictionMethod::MachineLearning { confidence }
            | PredictionMethod::RandomForest { confidence, .. } => *confidence,
            PredictionMethod::HistoricalAverage => 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionWithConfidence {
    pub timestamp: DateTime<Utc>,
    pub predicted_value: f64,
    pub confidence_low: f64,
    pub confidence_high: f64,
    pub confidence_score: f64,
    pub method: PredictionMethod,
}

impl PredictionWithConfidence {
    pub fn new(
        timestamp: DateTime<Utc>,
        predicted_value: f64,
        confidence_low: f64,
        confidence_high: f64,
        confidence_score: f64,
        method: PredictionMethod,
    ) -> Self {
        let clamped_predicted = predicted_value.clamp(0.0, 100.0);
        let clamped_low = confidence_low.clamp(0.0, 100.0);
        let clamped_high = confidence_high.clamp(0.0, 100.0);
        let (sorted_low, sorted_high) = if clamped_low <= clamped_high {
            (clamped_low, clamped_high)
        } else {
            (clamped_high, clamped_low)
        };
        // Expand the CI to always contain the predicted value so the
        // prediction line is visually inside the confidence band.
        let final_low = sorted_low.min(clamped_predicted);
        let final_high = sorted_high.max(clamped_predicted);
        Self {
            timestamp,
            predicted_value: clamped_predicted,
            confidence_low: final_low,
            confidence_high: final_high,
            confidence_score: confidence_score.clamp(0.0, 1.0),
            method,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.predicted_value >= 0.0
            && self.predicted_value <= 100.0
            && self.confidence_low <= self.predicted_value
            && self.confidence_high >= self.predicted_value
            && self.confidence_score >= 0.0
            && self.confidence_score <= 1.0
    }

    pub fn interval_width(&self) -> f64 {
        self.confidence_high - self.confidence_low
    }

    pub fn to_simple(&self) -> (DateTime<Utc>, f64) {
        (self.timestamp, self.predicted_value)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_prediction_method_is_ml() {
        let ml = PredictionMethod::MachineLearning { confidence: 0.8 };
        let avg = PredictionMethod::HistoricalAverage;

        assert!(ml.is_ml());
        assert!(!avg.is_ml());
    }

    #[test]
    fn test_prediction_method_confidence() {
        let ml = PredictionMethod::MachineLearning { confidence: 0.8 };
        let avg = PredictionMethod::HistoricalAverage;

        assert_relative_eq!(ml.confidence(), 0.8);
        assert_relative_eq!(avg.confidence(), 0.5);
    }

    #[test]
    fn test_prediction_with_confidence_creation() {
        let timestamp = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = PredictionWithConfidence::new(
            timestamp,
            50.0,
            40.0,
            60.0,
            0.8,
            PredictionMethod::MachineLearning { confidence: 0.8 },
        );

        assert_relative_eq!(pred.predicted_value, 50.0);
        assert_relative_eq!(pred.confidence_low, 40.0);
        assert_relative_eq!(pred.confidence_high, 60.0);
        assert!(pred.is_valid());
    }

    #[test]
    fn test_prediction_clamping() {
        let timestamp = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = PredictionWithConfidence::new(
            timestamp,
            150.0,
            -10.0,
            200.0,
            1.5,
            PredictionMethod::HistoricalAverage,
        );

        assert_relative_eq!(pred.predicted_value, 100.0);
        assert_relative_eq!(pred.confidence_low, 0.0);
        assert_relative_eq!(pred.confidence_high, 100.0);
        assert_relative_eq!(pred.confidence_score, 1.0);
    }

    #[test]
    fn test_interval_width() {
        let timestamp = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = PredictionWithConfidence::new(
            timestamp,
            50.0,
            35.0,
            65.0,
            0.7,
            PredictionMethod::HistoricalAverage,
        );

        assert_relative_eq!(pred.interval_width(), 30.0);
    }

    #[test]
    fn test_to_simple() {
        let timestamp = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = PredictionWithConfidence::new(
            timestamp,
            50.0,
            40.0,
            60.0,
            0.8,
            PredictionMethod::HistoricalAverage,
        );

        let (ts, val) = pred.to_simple();
        assert_eq!(ts, timestamp);
        assert_relative_eq!(val, 50.0);
    }

    #[test]
    fn test_is_valid() {
        let timestamp = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();

        let valid = PredictionWithConfidence {
            timestamp,
            predicted_value: 50.0,
            confidence_low: 40.0,
            confidence_high: 60.0,
            confidence_score: 0.8,
            method: PredictionMethod::HistoricalAverage,
        };
        assert!(valid.is_valid());

        let invalid = PredictionWithConfidence {
            timestamp,
            predicted_value: 50.0,
            confidence_low: 60.0,
            confidence_high: 70.0,
            confidence_score: 0.8,
            method: PredictionMethod::HistoricalAverage,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_new_swaps_inverted_interval() {
        let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = PredictionWithConfidence::new(
            ts,
            50.0,
            80.0,
            20.0,
            0.8,
            PredictionMethod::HistoricalAverage,
        );
        assert!(
            pred.confidence_low <= pred.confidence_high,
            "Expected low ({}) <= high ({})",
            pred.confidence_low,
            pred.confidence_high
        );
        assert_relative_eq!(pred.confidence_low, 20.0);
        assert_relative_eq!(pred.confidence_high, 80.0);
    }

    #[test]
    fn test_interval_width_always_non_negative() {
        let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();

        // Case: both in range but inverted
        let pred = PredictionWithConfidence::new(
            ts,
            50.0,
            70.0,
            30.0,
            0.5,
            PredictionMethod::HistoricalAverage,
        );
        assert!(pred.interval_width() >= 0.0);

        // Case: clamping causes inversion (low clamps to 0, high clamps to 0)
        let pred2 = PredictionWithConfidence::new(
            ts,
            50.0,
            -20.0,
            -10.0,
            0.5,
            PredictionMethod::HistoricalAverage,
        );
        assert!(pred2.interval_width() >= 0.0);

        // Case: both clamp to 100
        let pred3 = PredictionWithConfidence::new(
            ts,
            50.0,
            110.0,
            120.0,
            0.5,
            PredictionMethod::HistoricalAverage,
        );
        assert!(pred3.interval_width() >= 0.0);
    }

    #[test]
    fn test_prediction_method_random_forest_is_ml() {
        let rf = PredictionMethod::RandomForest {
            confidence: 0.85,
            n_trees: 100,
        };
        assert!(rf.is_ml());
    }

    #[test]
    fn test_prediction_method_random_forest_confidence() {
        let rf = PredictionMethod::RandomForest {
            confidence: 0.85,
            n_trees: 100,
        };
        assert_relative_eq!(rf.confidence(), 0.85);
    }

    #[test]
    fn test_new_expands_ci_to_contain_predicted_value() {
        let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();

        // Predicted value below both CI bounds (model overpredicts, residuals
        // shift CI upward)
        let pred = PredictionWithConfidence::new(
            ts,
            30.0,
            55.0,
            70.0,
            0.8,
            PredictionMethod::MachineLearning { confidence: 0.8 },
        );
        assert!(
            pred.is_valid(),
            "CI should expand to contain predicted_value: low={}, pred={}, high={}",
            pred.confidence_low,
            pred.predicted_value,
            pred.confidence_high,
        );
        assert_relative_eq!(pred.confidence_low, 30.0);
        assert_relative_eq!(pred.confidence_high, 70.0);

        // Predicted value above both CI bounds (model underpredicts, residuals
        // shift CI downward)
        let pred2 = PredictionWithConfidence::new(
            ts,
            80.0,
            20.0,
            50.0,
            0.8,
            PredictionMethod::MachineLearning { confidence: 0.8 },
        );
        assert!(
            pred2.is_valid(),
            "CI should expand to contain predicted_value: low={}, pred={}, high={}",
            pred2.confidence_low,
            pred2.predicted_value,
            pred2.confidence_high,
        );
        assert_relative_eq!(pred2.confidence_low, 20.0);
        assert_relative_eq!(pred2.confidence_high, 80.0);
    }

    // ── Property-based tests ─────────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_new_clamps_predicted_value(
            predicted in -50.0_f64..200.0,
            low in -50.0_f64..200.0,
            high in -50.0_f64..200.0,
            score in -1.0_f64..2.0,
        ) {
            let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let pred = PredictionWithConfidence::new(
                ts, predicted, low, high, score,
                PredictionMethod::HistoricalAverage,
            );
            prop_assert!(
                pred.predicted_value >= 0.0 && pred.predicted_value <= 100.0,
                "predicted_value out of range: {}", pred.predicted_value
            );
        }

        #[test]
        fn prop_new_clamps_confidence_score(
            score in -1.0_f64..2.0,
        ) {
            let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let pred = PredictionWithConfidence::new(
                ts, 50.0, 40.0, 60.0, score,
                PredictionMethod::HistoricalAverage,
            );
            prop_assert!(
                pred.confidence_score >= 0.0 && pred.confidence_score <= 1.0,
                "confidence_score out of range: {}", pred.confidence_score
            );
        }

        #[test]
        fn prop_new_clamps_intervals(
            low in -50.0_f64..200.0,
            high in -50.0_f64..200.0,
        ) {
            let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let pred = PredictionWithConfidence::new(
                ts, 50.0, low, high, 0.8,
                PredictionMethod::HistoricalAverage,
            );
            prop_assert!(
                pred.confidence_low >= 0.0 && pred.confidence_low <= 100.0,
                "confidence_low out of range: {}", pred.confidence_low
            );
            prop_assert!(
                pred.confidence_high >= 0.0 && pred.confidence_high <= 100.0,
                "confidence_high out of range: {}", pred.confidence_high
            );
        }

        #[test]
        fn prop_new_always_valid(
            predicted in -50.0_f64..200.0,
            low in -50.0_f64..200.0,
            high in -50.0_f64..200.0,
            score in -1.0_f64..2.0,
        ) {
            let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let pred = PredictionWithConfidence::new(
                ts, predicted, low, high, score,
                PredictionMethod::HistoricalAverage,
            );
            prop_assert!(
                pred.is_valid(),
                "new() should always produce valid predictions: \
                 low={}, pred={}, high={}, score={}",
                pred.confidence_low, pred.predicted_value,
                pred.confidence_high, pred.confidence_score,
            );
        }

        #[test]
        fn prop_interval_width_non_negative(
            low in -50.0_f64..=200.0,
            high in -50.0_f64..=200.0,
        ) {
            let ts = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let pred = PredictionWithConfidence::new(
                ts, 50.0, low, high, 0.8,
                PredictionMethod::HistoricalAverage,
            );
            let width = pred.interval_width();
            prop_assert!(
                width >= 0.0,
                "interval_width should be >= 0, got {width}"
            );
        }

        #[test]
        fn prop_method_confidence_range(
            confidence in 0.0_f64..=1.0,
            n_trees in 1_usize..500,
        ) {
            let ml = PredictionMethod::MachineLearning { confidence };
            let result = ml.confidence();
            prop_assert!(
                (0.0..=1.0).contains(&result),
                "ML confidence out of range: {result}"
            );

            let rf = PredictionMethod::RandomForest { confidence, n_trees };
            let result = rf.confidence();
            prop_assert!(
                (0.0..=1.0).contains(&result),
                "RF confidence out of range: {result}"
            );

            let avg = PredictionMethod::HistoricalAverage;
            let result = avg.confidence();
            prop_assert!(
                (0.0..=1.0).contains(&result),
                "Historical confidence out of range: {result}"
            );
        }
    }
}
