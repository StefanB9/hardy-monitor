//! Machine Learning module for occupancy prediction
//!
//! This module provides ML-based predictions using Gradient Boosted Decision Trees
//! trained on historical occupancy data.

pub mod confidence;
pub mod features;
pub mod model;
pub mod persistence;
pub mod training;

use std::collections::VecDeque;
use std::path::PathBuf;

use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::db::HourlyAverage;
use crate::schedule::GymSchedule;
use crate::traits::Clock;

pub use confidence::{PredictionMethod, PredictionWithConfidence};
pub use features::{FeatureExtractor, PredictionFeatures};
pub use model::TrainedModel;
pub use persistence::PersistedModel;
pub use training::TrainingResult;

/// Configuration for the ML prediction system
#[derive(Debug, Clone)]
pub struct MlConfig {
    /// Whether ML predictions are enabled
    pub enabled: bool,
    /// Number of days of historical data to use for training
    pub training_window_days: i64,
    /// How often to retrain the model (in hours)
    pub retrain_interval_hours: i64,
    /// How many hours ahead to predict
    pub prediction_horizon_hours: i64,
    /// Minimum number of samples required before training
    pub min_samples_for_training: usize,
    /// Path to save/load the trained model
    pub model_path: Option<PathBuf>,
    /// Whether to fall back to simple averages if ML fails
    pub fallback_on_error: bool,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_window_days: 56,
            retrain_interval_hours: 24,
            prediction_horizon_hours: 6,
            min_samples_for_training: 500,
            model_path: None,
            fallback_on_error: true,
        }
    }
}

/// Main predictor that combines ML model with fallback logic
pub struct OccupancyPredictor {
    /// Trained ML model (if available)
    model: Option<TrainedModel>,
    /// Feature extractor for building prediction inputs
    feature_extractor: FeatureExtractor,
    /// Recent occupancy data for momentum features
    recent_data: VecDeque<(DateTime<Utc>, f64)>,
    /// Timestamp of last model training
    last_training: Option<DateTime<Utc>>,
    /// Configuration
    config: MlConfig,
}

impl OccupancyPredictor {
    /// Create a new predictor with the given configuration
    pub fn new(config: MlConfig) -> Self {
        Self {
            model: None,
            feature_extractor: FeatureExtractor::new(),
            recent_data: VecDeque::with_capacity(180), // 3 hours at 1-min intervals
            last_training: None,
            config,
        }
    }

    /// Check if ML predictions can be used
    pub fn can_use_ml(&self) -> bool {
        self.config.enabled && self.model.is_some()
    }

    /// Check if the model needs retraining
    pub fn needs_retraining(&self, clock: &dyn Clock) -> bool {
        match self.last_training {
            None => true,
            Some(last) => {
                let hours_since = (clock.now_utc() - last).num_hours();
                hours_since >= self.config.retrain_interval_hours
            }
        }
    }

    /// Update the trained model
    pub fn set_model(&mut self, model: TrainedModel, trained_at: DateTime<Utc>) {
        self.model = Some(model);
        self.last_training = Some(trained_at);
    }

    /// Add a recent occupancy observation for momentum features
    pub fn add_observation(&mut self, timestamp: DateTime<Utc>, percentage: f64) {
        // Keep only the last 3 hours of data
        while self.recent_data.len() >= 180 {
            self.recent_data.pop_front();
        }
        self.recent_data.push_back((timestamp, percentage));
    }

    /// Update feature extractor with new baseline data
    pub fn update_baseline(&mut self, baseline: &[HourlyAverage]) {
        self.feature_extractor.update_historical_stats(baseline);
    }

    /// Generate predictions for the next N hours
    pub fn predict(
        &self,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
        clock: &dyn Clock,
    ) -> Vec<PredictionWithConfidence> {
        let now = clock.now_utc();
        let mut predictions = Vec::new();

        for hours_ahead in 1..=self.config.prediction_horizon_hours {
            let target_time = now + chrono::Duration::hours(hours_ahead);

            // Skip if gym is closed at target time
            let local_target = target_time.with_timezone(&chrono::Local);
            if !schedule.is_open(&local_target) {
                continue;
            }

            let prediction = self.predict_single(target_time, hours_ahead, baseline, schedule);
            predictions.push(prediction);
        }

        predictions
    }

    /// Generate a single prediction for a target time
    fn predict_single(
        &self,
        target_time: DateTime<Utc>,
        hours_ahead: i64,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> PredictionWithConfidence {
        // Try ML prediction first if available
        if self.can_use_ml() {
            if let Some(pred) = self.ml_predict(target_time, hours_ahead, baseline, schedule) {
                return pred;
            }
        }

        // Fall back to simple historical average
        self.fallback_predict(target_time, baseline)
    }

    /// ML-based prediction
    fn ml_predict(
        &self,
        target_time: DateTime<Utc>,
        hours_ahead: i64,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> Option<PredictionWithConfidence> {
        let model = self.model.as_ref()?;

        // Extract features for the target time
        let features = self.feature_extractor.extract(
            target_time,
            hours_ahead,
            &self.recent_data,
            baseline,
            schedule,
        );

        // Get prediction from model
        let predicted_value = model.predict(&features)?;

        // Calculate confidence based on historical variance and horizon
        let (confidence_low, confidence_high, confidence_score) =
            self.calculate_confidence(target_time, predicted_value, hours_ahead);

        Some(PredictionWithConfidence {
            timestamp: normalize_timestamp(target_time),
            predicted_value: predicted_value.clamp(0.0, 100.0),
            confidence_low,
            confidence_high,
            confidence_score,
            method: PredictionMethod::MachineLearning {
                confidence: confidence_score,
            },
        })
    }

    /// Fallback prediction using simple historical average
    fn fallback_predict(
        &self,
        target_time: DateTime<Utc>,
        baseline: &[HourlyAverage],
    ) -> PredictionWithConfidence {
        let target_weekday = target_time.weekday().num_days_from_monday();
        let target_hour = target_time.hour();

        let (predicted_value, confidence_low, confidence_high) = baseline
            .iter()
            .find(|avg| avg.weekday == target_weekday && avg.hour == target_hour)
            .map(|avg| {
                let std_dev = self
                    .feature_extractor
                    .get_slot_std(target_weekday, target_hour)
                    .unwrap_or(10.0);
                (
                    avg.avg_percentage,
                    (avg.avg_percentage - std_dev).clamp(0.0, 100.0),
                    (avg.avg_percentage + std_dev).clamp(0.0, 100.0),
                )
            })
            .unwrap_or((50.0, 30.0, 70.0)); // Default if no data

        PredictionWithConfidence {
            timestamp: normalize_timestamp(target_time),
            predicted_value,
            confidence_low,
            confidence_high,
            confidence_score: 0.5, // Lower confidence for fallback
            method: PredictionMethod::HistoricalAverage,
        }
    }

    /// Calculate confidence intervals for a prediction
    fn calculate_confidence(
        &self,
        target_time: DateTime<Utc>,
        predicted_value: f64,
        hours_ahead: i64,
    ) -> (f64, f64, f64) {
        let weekday = target_time.weekday().num_days_from_monday();
        let hour = target_time.hour();

        // Get historical standard deviation for this slot
        let base_std = self
            .feature_extractor
            .get_slot_std(weekday, hour)
            .unwrap_or(15.0);

        // Increase uncertainty with prediction horizon
        let horizon_penalty = 1.0 + (hours_ahead as f64 - 1.0) * 0.15;
        let adjusted_std = base_std * horizon_penalty;

        let confidence_low = (predicted_value - adjusted_std).clamp(0.0, 100.0);
        let confidence_high = (predicted_value + adjusted_std).clamp(0.0, 100.0);

        // Confidence score: higher when std is lower
        let confidence_score = (1.0 / (1.0 + adjusted_std / 20.0)).clamp(0.0, 1.0);

        (confidence_low, confidence_high, confidence_score)
    }

    /// Get the configuration
    pub fn config(&self) -> &MlConfig {
        &self.config
    }

    /// Check if a model is loaded
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Get the last training timestamp
    pub fn last_training(&self) -> Option<DateTime<Utc>> {
        self.last_training
    }
}

/// Normalize a timestamp to the start of the hour
fn normalize_timestamp(dt: DateTime<Utc>) -> DateTime<Utc> {
    dt.with_minute(0)
        .and_then(|d| d.with_second(0))
        .and_then(|d| d.with_nanosecond(0))
        .unwrap_or(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MockClock;
    use chrono::TimeZone;

    #[test]
    fn test_predictor_creation() {
        let config = MlConfig::default();
        let predictor = OccupancyPredictor::new(config);

        assert!(!predictor.can_use_ml());
        assert!(!predictor.has_model());
    }

    #[test]
    fn test_needs_retraining_without_model() {
        let config = MlConfig::default();
        let predictor = OccupancyPredictor::new(config);
        let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());

        assert!(predictor.needs_retraining(&clock));
    }

    #[test]
    fn test_add_observation() {
        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);

        let now = Utc::now();
        predictor.add_observation(now, 50.0);

        assert_eq!(predictor.recent_data.len(), 1);
    }

    #[test]
    fn test_normalize_timestamp() {
        let dt = Utc.with_ymd_and_hms(2024, 6, 17, 10, 30, 45).unwrap();
        let normalized = normalize_timestamp(dt);

        assert_eq!(normalized.minute(), 0);
        assert_eq!(normalized.second(), 0);
        assert_eq!(normalized.hour(), 10);
    }

    #[test]
    fn test_fallback_prediction() {
        let config = MlConfig::default();
        let predictor = OccupancyPredictor::new(config);

        let baseline = vec![HourlyAverage {
            weekday: 0, // Monday
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap(); // Monday
        let pred = predictor.fallback_predict(target, &baseline);

        assert_eq!(pred.predicted_value, 45.0);
        assert!(matches!(pred.method, PredictionMethod::HistoricalAverage));
    }

    #[test]
    fn test_config_defaults() {
        let config = MlConfig::default();

        assert!(config.enabled);
        assert_eq!(config.training_window_days, 56);
        assert_eq!(config.retrain_interval_hours, 24);
        assert_eq!(config.prediction_horizon_hours, 6);
        assert_eq!(config.min_samples_for_training, 500);
        assert!(config.fallback_on_error);
    }
}
