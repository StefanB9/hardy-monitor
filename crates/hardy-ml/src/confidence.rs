//! A forecast value with its interval and the method that produced it.

use chrono::{DateTime, Utc};

/// How a forecast value was produced.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionMethod {
    /// A trained model other than a random forest.
    MachineLearning { confidence: f64 },
    /// A trained random forest.
    RandomForest { confidence: f64, n_trees: usize },
    /// Plain slot averages (no usable model).
    HistoricalAverage,
}

impl PredictionMethod {
    /// Whether a trained model produced the value.
    pub fn is_ml(&self) -> bool {
        matches!(
            self,
            PredictionMethod::MachineLearning { .. } | PredictionMethod::RandomForest { .. }
        )
    }
}

/// One forecast value: clamped to 0–100 % with an interval that contains it.
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
    /// Clamps all values to their ranges, orders the interval and widens it
    /// to contain the predicted value.
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
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    /// The invariant `new` guarantees.
    fn is_valid(p: &PredictionWithConfidence) -> bool {
        (0.0..=100.0).contains(&p.predicted_value)
            && p.confidence_low <= p.predicted_value
            && p.confidence_high >= p.predicted_value
            && (0.0..=1.0).contains(&p.confidence_score)
    }

    #[test]
    fn test_prediction_method_is_ml() {
        let ml = PredictionMethod::MachineLearning { confidence: 0.8 };
        let avg = PredictionMethod::HistoricalAverage;

        assert!(ml.is_ml());
        assert!(!avg.is_ml());
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
        assert!(is_valid(&pred));
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
    fn test_prediction_method_random_forest_is_ml() {
        let rf = PredictionMethod::RandomForest {
            confidence: 0.85,
            n_trees: 100,
        };
        assert!(rf.is_ml());
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
            is_valid(&pred),
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
            is_valid(&pred2),
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
                is_valid(&pred),
                "new() should always produce valid predictions: \
                 low={}, pred={}, high={}, score={}",
                pred.confidence_low, pred.predicted_value,
                pred.confidence_high, pred.confidence_score,
            );
        }
    }
}
