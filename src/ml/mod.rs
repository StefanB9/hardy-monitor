pub mod confidence;
pub mod config;
pub mod evaluation;
pub mod features;
pub mod model;
pub mod persistence;
pub mod residuals;
pub mod training;

use std::collections::VecDeque;

use chrono::{DateTime, Datelike, Timelike, Utc};
pub use confidence::{PredictionMethod, PredictionWithConfidence};
pub use config::MlConfig;
pub use features::{FeatureExtractor, PredictionFeatures};
pub use model::TrainedModel;
pub use persistence::PersistedModel;
pub use residuals::ResidualQuantiles;
pub use training::TrainingResult;

use crate::{db::HourlyAverage, schedule::GymSchedule, traits::Clock};

pub struct OccupancyPredictor {
    model: Option<TrainedModel>,
    feature_extractor: FeatureExtractor,
    residual_quantiles: Option<ResidualQuantiles>,
    recent_data: VecDeque<(DateTime<Utc>, f64)>,
    last_training: Option<DateTime<Utc>>,
    config: MlConfig,
}

impl OccupancyPredictor {
    pub fn new(config: MlConfig) -> Self {
        Self {
            model: None,
            feature_extractor: FeatureExtractor::new(),
            residual_quantiles: None,
            recent_data: VecDeque::with_capacity(360),
            last_training: None,
            config,
        }
    }

    pub fn can_use_ml(&self) -> bool {
        self.config.enabled && self.model.is_some()
    }

    pub fn needs_retraining(&self, clock: &dyn Clock) -> bool {
        match self.last_training {
            None => true,
            Some(last) => {
                let hours_since = (clock.now_utc() - last).num_hours();
                hours_since >= self.config.retrain_interval_hours
            }
        }
    }

    pub fn set_model(&mut self, model: TrainedModel, trained_at: DateTime<Utc>) {
        self.model = Some(model);
        self.last_training = Some(trained_at);
    }

    /// Set or clear the residual-based quantile data for confidence intervals.
    pub fn set_residual_quantiles(&mut self, quantiles: Option<ResidualQuantiles>) {
        self.residual_quantiles = quantiles;
    }

    pub fn add_observation(&mut self, timestamp: DateTime<Utc>, percentage: f64) {
        while self.recent_data.len() >= 360 {
            self.recent_data.pop_front();
        }
        self.recent_data.push_back((timestamp, percentage));
    }

    pub fn update_baseline(&mut self, baseline: &[HourlyAverage]) {
        self.feature_extractor.update_historical_stats(baseline);
    }

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

            let local_target = target_time.with_timezone(&chrono::Local);
            if !schedule.is_open(&local_target) {
                continue;
            }

            let prediction = self.predict_single(target_time, hours_ahead, baseline, schedule);
            predictions.push(prediction);
        }

        predictions
    }

    fn predict_single(
        &self,
        target_time: DateTime<Utc>,
        hours_ahead: i64,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> PredictionWithConfidence {
        if self.can_use_ml()
            && let Some(pred) = self.ml_predict(target_time, hours_ahead, baseline, schedule)
        {
            return pred;
        }

        self.fallback_predict(target_time, baseline)
    }

    fn ml_predict(
        &self,
        target_time: DateTime<Utc>,
        hours_ahead: i64,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> Option<PredictionWithConfidence> {
        let model = self.model.as_ref()?;

        let features = self.feature_extractor.extract(
            target_time,
            hours_ahead,
            &self.recent_data,
            baseline,
            schedule,
        );

        let predicted_value = model.predict(&features)?;

        let (confidence_low, confidence_high, confidence_score) =
            self.calculate_confidence(target_time, predicted_value, hours_ahead);

        let method = if model.is_random_forest() {
            PredictionMethod::RandomForest {
                confidence: confidence_score,
                n_trees: model.n_trees().unwrap_or(0),
            }
        } else {
            PredictionMethod::MachineLearning {
                confidence: confidence_score,
            }
        };

        Some(PredictionWithConfidence {
            timestamp: normalize_timestamp(target_time),
            predicted_value: predicted_value.clamp(0.0, 100.0),
            confidence_low,
            confidence_high,
            confidence_score,
            method,
        })
    }

    fn fallback_predict(
        &self,
        target_time: DateTime<Utc>,
        baseline: &[HourlyAverage],
    ) -> PredictionWithConfidence {
        let target_weekday = target_time.weekday().num_days_from_monday().cast_signed();
        let target_hour = target_time.hour().cast_signed();

        let (predicted_value, confidence_low, confidence_high) = baseline
            .iter()
            .find(|avg| avg.weekday == target_weekday && avg.hour == target_hour)
            .map_or((50.0, 30.0, 70.0), |avg| {
                let std_dev = self
                    .feature_extractor
                    .get_slot_std(target_weekday, target_hour)
                    .unwrap_or(10.0);
                (
                    avg.avg_percentage,
                    (avg.avg_percentage - std_dev).clamp(0.0, 100.0),
                    (avg.avg_percentage + std_dev).clamp(0.0, 100.0),
                )
            });

        PredictionWithConfidence {
            timestamp: normalize_timestamp(target_time),
            predicted_value,
            confidence_low,
            confidence_high,
            confidence_score: 0.5,
            method: PredictionMethod::HistoricalAverage,
        }
    }

    /// Compute confidence interval, delegating to residual quantiles when
    /// available, falling back to heuristic otherwise.
    fn calculate_confidence(
        &self,
        target_time: DateTime<Utc>,
        predicted_value: f64,
        hours_ahead: i64,
    ) -> (f64, f64, f64) {
        if let Some(ref quantiles) = self.residual_quantiles {
            let weekday = target_time.weekday().num_days_from_monday();
            let hour = target_time.hour();
            quantiles.compute_confidence_interval(predicted_value, weekday, hour, hours_ahead)
        } else {
            self.calculate_confidence_heuristic(target_time, predicted_value, hours_ahead)
        }
    }

    /// Heuristic confidence interval based on historical std and horizon
    /// penalty. Used when no residual quantile data is available.
    fn calculate_confidence_heuristic(
        &self,
        target_time: DateTime<Utc>,
        predicted_value: f64,
        hours_ahead: i64,
    ) -> (f64, f64, f64) {
        let weekday = target_time.weekday().num_days_from_monday().cast_signed();
        let hour = target_time.hour().cast_signed();

        let base_std = self
            .feature_extractor
            .get_slot_std(weekday, hour)
            .unwrap_or(15.0);

        #[allow(clippy::cast_precision_loss)]
        let horizon_penalty = 1.0 + (hours_ahead as f64 - 1.0) * 0.15;
        let adjusted_std = base_std * horizon_penalty;

        let confidence_low = (predicted_value - adjusted_std).clamp(0.0, 100.0);
        let confidence_high = (predicted_value + adjusted_std).clamp(0.0, 100.0);

        let confidence_score = (1.0 / (1.0 + adjusted_std / 20.0)).clamp(0.0, 1.0);

        (confidence_low, confidence_high, confidence_score)
    }

    pub fn config(&self) -> &MlConfig {
        &self.config
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    pub fn last_training(&self) -> Option<DateTime<Utc>> {
        self.last_training
    }
}

fn normalize_timestamp(dt: DateTime<Utc>) -> DateTime<Utc> {
    dt.with_minute(0)
        .and_then(|d| d.with_second(0))
        .and_then(|d| d.with_nanosecond(0))
        .unwrap_or(dt)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_abs_diff_eq;
    use chrono::TimeZone;

    use super::*;
    use crate::{
        config::{ScheduleConfig, ScheduleHours},
        traits::MockClock,
    };

    fn test_schedule() -> GymSchedule {
        GymSchedule::new(&ScheduleConfig {
            weekday: ScheduleHours {
                open_hour: 6,
                close_hour: 23,
            },
            weekend: ScheduleHours {
                open_hour: 8,
                close_hour: 22,
            },
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn create_test_features(n: usize) -> Vec<PredictionFeatures> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                PredictionFeatures {
                    hour_sin: (t * 0.3).sin(),
                    hour_cos: (t * 0.31).cos(),
                    weekday_sin: (t * 0.07).sin(),
                    weekday_cos: (t * 0.071).cos(),
                    historical_avg: 30.0 + (t % 40.0),
                    historical_std: 5.0 + (t % 15.0),
                    recent_avg_1h: 35.0 + (t % 35.0),
                    recent_avg_3h: 40.0 + ((t * 1.3) % 30.0),
                    recent_trend: -10.0 + (t % 20.0),
                    day_avg_so_far: 30.0 + (t % 45.0),
                    prev_day_avg: 45.0 + ((t * 0.7) % 25.0),
                    is_weekend: if (i % 7) >= 5 { 1.0 } else { 0.0 },
                    is_holiday: if i % 30 == 0 { 1.0 } else { 0.0 },
                    week_of_year_sin: (t * 0.02).sin(),
                    week_of_year_cos: (t * 0.021).cos(),
                    hours_ahead: 1.0 + (t % 6.0),
                    raw_hour: t % 24.0,
                    raw_weekday: t % 7.0,
                    time_to_close: 5.0 + (t % 12.0),
                    occupancy_volatility: 2.0 + (t % 10.0),
                    recent_avg_6h: 42.0 + ((t * 0.9) % 28.0),
                    prev_week_same_slot: 38.0 + (t % 35.0),
                }
            })
            .collect()
    }

    fn predictor_with_lr_model() -> Result<OccupancyPredictor> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = model::ModelBuilder::new();
        let trained = builder.train(&features, &targets)?;

        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);
        predictor.set_model(trained, Utc::now());

        Ok(predictor)
    }

    fn predictor_with_rf_model() -> Result<OccupancyPredictor> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = model::ModelBuilder::new().n_trees(10).max_depth(5);
        let trained = builder.train_rf(&features, &targets)?;

        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);
        predictor.set_model(trained, Utc::now());

        Ok(predictor)
    }

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
            weekday: 0,
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let pred = predictor.fallback_predict(target, &baseline);

        assert_abs_diff_eq!(pred.predicted_value, 45.0, epsilon = 1e-5);
        assert!(matches!(pred.method, PredictionMethod::HistoricalAverage));
    }

    // ── Step 5 tests ─────────────────────────────────────────────────

    #[test]
    fn test_set_residual_quantiles() {
        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);

        assert!(predictor.residual_quantiles.is_none());

        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
            .collect();
        let quantiles = ResidualQuantiles::from_residuals(&residuals);

        predictor.set_residual_quantiles(quantiles);
        assert!(predictor.residual_quantiles.is_some());

        predictor.set_residual_quantiles(None);
        assert!(predictor.residual_quantiles.is_none());
    }

    #[test]
    fn test_predictor_uses_residual_quantiles() -> Result<()> {
        let mut predictor = predictor_with_lr_model()?;

        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
            .collect();
        let quantiles =
            ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());
        predictor.set_residual_quantiles(Some(quantiles.clone()));

        // Monday hour 10
        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let schedule = test_schedule();
        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let pred = predictor.ml_predict(target, 1, &baseline, &schedule);
        assert!(pred.is_some());
        let p = pred.unwrap_or_else(|| unreachable!());

        // Verify interval matches direct quantile computation
        let (expected_low, expected_high, expected_score) =
            quantiles.compute_confidence_interval(p.predicted_value, 0, 10, 1);

        assert_abs_diff_eq!(p.confidence_low, expected_low, epsilon = 1e-10);
        assert_abs_diff_eq!(p.confidence_high, expected_high, epsilon = 1e-10);
        assert_abs_diff_eq!(p.confidence_score, expected_score, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_predictor_falls_back_without_quantiles() -> Result<()> {
        let predictor = predictor_with_lr_model()?;

        assert!(predictor.residual_quantiles.is_none());

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let schedule = test_schedule();
        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let pred = predictor.ml_predict(target, 1, &baseline, &schedule);
        assert!(pred.is_some());
        let p = pred.unwrap_or_else(|| unreachable!());

        // Without quantiles, method is MachineLearning (not RandomForest)
        assert!(matches!(p.method, PredictionMethod::MachineLearning { .. }));
        assert!(p.confidence_low <= p.confidence_high);
        assert!(p.confidence_low >= 0.0);
        assert!(p.confidence_high <= 100.0);

        Ok(())
    }

    #[test]
    fn test_predictor_random_forest_method() -> Result<()> {
        let mut predictor = predictor_with_rf_model()?;

        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
            .collect();
        let quantiles = ResidualQuantiles::from_residuals(&residuals);
        predictor.set_residual_quantiles(quantiles);

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let schedule = test_schedule();
        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let pred = predictor.ml_predict(target, 1, &baseline, &schedule);
        assert!(pred.is_some());
        let p = pred.unwrap_or_else(|| unreachable!());

        assert!(
            matches!(p.method, PredictionMethod::RandomForest { n_trees, .. } if n_trees == 10)
        );

        Ok(())
    }

    #[test]
    fn test_predictor_ml_method_for_lr() -> Result<()> {
        let mut predictor = predictor_with_lr_model()?;

        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
            .collect();
        let quantiles = ResidualQuantiles::from_residuals(&residuals);
        predictor.set_residual_quantiles(quantiles);

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        let schedule = test_schedule();
        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 45.0,
            sample_count: 100,
        }];

        let pred = predictor.ml_predict(target, 1, &baseline, &schedule);
        assert!(pred.is_some());
        let p = pred.unwrap_or_else(|| unreachable!());

        // LR model with quantiles → still MachineLearning method
        assert!(matches!(p.method, PredictionMethod::MachineLearning { .. }));

        Ok(())
    }
}
