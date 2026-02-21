use chrono::{DateTime, Utc};

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionMethod {
    MachineLearning { confidence: f64 },
    HistoricalAverage,
}

impl PredictionMethod {
    pub fn is_ml(&self) -> bool {
        matches!(self, PredictionMethod::MachineLearning { .. })
    }

    pub fn confidence(&self) -> f64 {
        match self {
            PredictionMethod::MachineLearning { confidence } => *confidence,
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
        Self {
            timestamp,
            predicted_value: predicted_value.clamp(0.0, 100.0),
            confidence_low: confidence_low.clamp(0.0, 100.0),
            confidence_high: confidence_high.clamp(0.0, 100.0),
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
    use chrono::TimeZone;

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

        assert_eq!(ml.confidence(), 0.8);
        assert_eq!(avg.confidence(), 0.5);
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

        assert_eq!(pred.predicted_value, 50.0);
        assert_eq!(pred.confidence_low, 40.0);
        assert_eq!(pred.confidence_high, 60.0);
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

        assert_eq!(pred.predicted_value, 100.0);
        assert_eq!(pred.confidence_low, 0.0);
        assert_eq!(pred.confidence_high, 100.0);
        assert_eq!(pred.confidence_score, 1.0);
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

        assert_eq!(pred.interval_width(), 30.0);
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
        assert_eq!(val, 50.0);
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
}
