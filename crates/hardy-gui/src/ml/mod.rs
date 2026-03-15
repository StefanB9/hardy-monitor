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
use hardy_core::{db::HourlyAverage, schedule::GymSchedule, traits::Clock};
pub use model::TrainedModel;
pub use persistence::PersistedModel;
pub use residuals::ResidualQuantiles;
pub use training::TrainingResult;

use self::persistence::{SerializedCvScores, SerializedHyperparameters};

/// Summary of the last training run for GUI display.
#[derive(Debug, Clone)]
pub struct TrainingInfo {
    pub algorithm: String,
    pub training_samples: usize,
    pub training_window_days: i64,
    pub training_mse: f64,
    pub validation_mse: Option<f64>,
    pub cv_scores: Option<CvScoresSummary>,
    pub best_hyperparameters: Option<HyperparametersSummary>,
}

/// Displayable cross-validation score summary.
#[derive(Debug, Clone)]
pub struct CvScoresSummary {
    pub rmse_mean: f64,
    pub rmse_std: f64,
    pub mae_mean: f64,
    pub mae_std: f64,
    pub r_squared_mean: f64,
    pub r_squared_std: f64,
}

/// Displayable hyperparameters summary.
#[derive(Debug, Clone)]
pub struct HyperparametersSummary {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl TrainingInfo {
    /// Reconstruct `TrainingInfo` from a persisted model's metadata.
    pub fn from_persisted(persisted: &PersistedModel) -> Self {
        Self {
            algorithm: persisted.model_summary.model_type.clone(),
            training_samples: persisted.training_samples,
            training_window_days: persisted.training_window_days,
            training_mse: persisted.training_mse,
            validation_mse: persisted.validation_mse,
            cv_scores: persisted.cv_scores.as_ref().map(Self::cv_from_serialized),
            best_hyperparameters: persisted
                .best_hyperparameters
                .as_ref()
                .map(Self::hp_from_serialized),
        }
    }

    fn cv_from_serialized(cv: &SerializedCvScores) -> CvScoresSummary {
        CvScoresSummary {
            rmse_mean: cv.rmse_mean,
            rmse_std: cv.rmse_std,
            mae_mean: cv.mae_mean,
            mae_std: cv.mae_std,
            r_squared_mean: cv.r_squared_mean,
            r_squared_std: cv.r_squared_std,
        }
    }

    fn hp_from_serialized(hp: &SerializedHyperparameters) -> HyperparametersSummary {
        HyperparametersSummary {
            n_trees: hp.n_trees,
            max_depth: hp.max_depth,
            min_samples_leaf: hp.min_samples_leaf,
            max_features: hp.max_features,
        }
    }
}

pub struct OccupancyPredictor {
    model: Option<TrainedModel>,
    feature_extractor: FeatureExtractor,
    residual_quantiles: Option<ResidualQuantiles>,
    training_info: Option<TrainingInfo>,
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
            training_info: None,
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

    /// Set or clear the training info for GUI display.
    pub fn set_training_info(&mut self, info: Option<TrainingInfo>) {
        self.training_info = info;
    }

    /// Get the training info for GUI display.
    pub fn training_info(&self) -> Option<&TrainingInfo> {
        self.training_info.as_ref()
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

    /// Generate predictions for each open hour up to the configured horizon.
    ///
    /// Uses an autoregressive loop: each prediction is injected into a working
    /// copy of `recent_data` so that lag features (rolling averages, trend,
    /// volatility) evolve across the forecast horizon instead of staying frozen
    /// at the last observed value.
    pub fn predict(
        &self,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
        clock: &dyn Clock,
    ) -> Vec<PredictionWithConfidence> {
        let now = clock.now_utc();
        let mut predictions = Vec::new();
        let mut working_buffer = self.recent_data.clone();

        for hours_ahead in 1..=self.config.prediction_horizon_hours {
            let target_time = now + chrono::Duration::hours(hours_ahead);

            let local_target = target_time.with_timezone(&chrono::Local);
            if !schedule.is_open(&local_target) {
                continue;
            }

            let prediction = self.predict_single(
                target_time,
                hours_ahead,
                baseline,
                schedule,
                &working_buffer,
            );

            // Inject predicted value so subsequent iterations see evolving lag
            // features (mirrors the sliding window used during training).
            while working_buffer.len() >= 360 {
                working_buffer.pop_front();
            }
            working_buffer.push_back((target_time, prediction.predicted_value));

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
        recent_data: &VecDeque<(DateTime<Utc>, f64)>,
    ) -> PredictionWithConfidence {
        if self.can_use_ml()
            && let Some(pred) =
                self.ml_predict(target_time, hours_ahead, baseline, schedule, recent_data)
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
        recent_data: &VecDeque<(DateTime<Utc>, f64)>,
    ) -> Option<PredictionWithConfidence> {
        let model = self.model.as_ref()?;

        let features = self.feature_extractor.extract(
            target_time,
            hours_ahead,
            recent_data,
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
    use hardy_core::{
        config::{ScheduleConfig, ScheduleHours},
        traits::MockClock,
    };

    use super::*;

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

        let recent_data = VecDeque::new();
        let pred = predictor.ml_predict(target, 1, &baseline, &schedule, &recent_data);
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

        let recent_data = VecDeque::new();
        let pred = predictor.ml_predict(target, 1, &baseline, &schedule, &recent_data);
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

        let recent_data = VecDeque::new();
        let pred = predictor.ml_predict(target, 1, &baseline, &schedule, &recent_data);
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

        let recent_data = VecDeque::new();
        let pred = predictor.ml_predict(target, 1, &baseline, &schedule, &recent_data);
        assert!(pred.is_some());
        let p = pred.unwrap_or_else(|| unreachable!());

        // LR model with quantiles → still MachineLearning method
        assert!(matches!(p.method, PredictionMethod::MachineLearning { .. }));

        Ok(())
    }

    // ── TrainingInfo tests ──────────────────────────────────────────

    #[test]
    fn test_training_info_default_none() {
        let config = MlConfig::default();
        let predictor = OccupancyPredictor::new(config);
        assert!(predictor.training_info().is_none());
    }

    #[test]
    fn test_set_and_get_training_info() {
        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);

        let info = TrainingInfo {
            algorithm: "RandomForest".to_string(),
            training_samples: 2000,
            training_window_days: 56,
            training_mse: 10.5,
            validation_mse: Some(12.3),
            cv_scores: Some(CvScoresSummary {
                rmse_mean: 3.5,
                rmse_std: 0.2,
                mae_mean: 2.8,
                mae_std: 0.15,
                r_squared_mean: 0.85,
                r_squared_std: 0.02,
            }),
            best_hyperparameters: Some(HyperparametersSummary {
                n_trees: 150,
                max_depth: 12,
                min_samples_leaf: 3,
                max_features: Some(8),
            }),
        };
        predictor.set_training_info(Some(info));

        let retrieved = predictor.training_info();
        assert!(retrieved.is_some());
        let ti = retrieved.unwrap_or_else(|| unreachable!());
        assert_eq!(ti.algorithm, "RandomForest");
        assert_eq!(ti.training_samples, 2000);
    }

    #[test]
    fn test_training_info_cleared() {
        let config = MlConfig::default();
        let mut predictor = OccupancyPredictor::new(config);

        let info = TrainingInfo {
            algorithm: "LinearRegression".to_string(),
            training_samples: 1000,
            training_window_days: 28,
            training_mse: 5.0,
            validation_mse: None,
            cv_scores: None,
            best_hyperparameters: None,
        };
        predictor.set_training_info(Some(info));
        assert!(predictor.training_info().is_some());

        predictor.set_training_info(None);
        assert!(predictor.training_info().is_none());
    }

    #[test]
    fn test_training_info_from_persisted() {
        let quantiles = {
            let residuals: Vec<(u32, u32, f64)> = (0..20)
                .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
                .collect();
            ResidualQuantiles::from_residuals(&residuals)
        };

        let persisted = PersistedModel::new(
            56,
            2500,
            12.3,
            Some(14.1),
            vec![],
            persistence::ModelSummary {
                model_type: "RandomForest".to_string(),
                max_depth: Some(12),
                feature_importance: None,
            },
            quantiles.as_ref(),
            Some(persistence::SerializedHyperparameters {
                n_trees: 150,
                max_depth: 12,
                min_samples_leaf: 3,
                max_features: Some(8),
            }),
            Some(persistence::SerializedCvScores {
                rmse_mean: 4.21,
                rmse_std: 0.35,
                mae_mean: 3.12,
                mae_std: 0.28,
                r_squared_mean: 0.87,
                r_squared_std: 0.03,
                mse_mean: 17.72,
                mse_std: 2.95,
            }),
        );

        let info = TrainingInfo::from_persisted(&persisted);
        assert_eq!(info.algorithm, "RandomForest");
        assert_eq!(info.training_samples, 2500);
        assert_eq!(info.training_window_days, 56);

        let cv = info.cv_scores.unwrap_or_else(|| unreachable!());
        assert_abs_diff_eq!(cv.rmse_mean, 4.21, epsilon = 1e-10);
        assert_abs_diff_eq!(cv.r_squared_mean, 0.87, epsilon = 1e-10);

        let hp = info.best_hyperparameters.unwrap_or_else(|| unreachable!());
        assert_eq!(hp.n_trees, 150);
        assert_eq!(hp.max_depth, 12);
    }

    #[test]
    fn test_training_info_from_persisted_without_optional() {
        let persisted = PersistedModel::new(
            28,
            1000,
            5.5,
            None,
            vec![],
            persistence::ModelSummary {
                model_type: "LinearRegression".to_string(),
                max_depth: None,
                feature_importance: None,
            },
            None,
            None,
            None,
        );

        let info = TrainingInfo::from_persisted(&persisted);
        assert_eq!(info.algorithm, "LinearRegression");
        assert!(info.cv_scores.is_none());
        assert!(info.best_hyperparameters.is_none());
        assert!(info.validation_mse.is_none());
    }

    // ── Autoregressive prediction tests ─────────────────────────────

    /// Helper: create a predictor with a trained LR model and a populated
    /// `recent_data` buffer showing a clear upward trend.
    fn predictor_with_trending_data() -> Result<OccupancyPredictor> {
        let mut predictor = predictor_with_lr_model()?;

        // Monday 2024-06-17 12:00 UTC — populate 2 hours of minute-level data
        // with a clear upward trend (20% → 60%).
        let base = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
        #[allow(clippy::cast_precision_loss)]
        for i in 0..120 {
            let ts = base + chrono::Duration::minutes(i);
            let pct = 20.0 + (i as f64 / 120.0) * 40.0;
            predictor.add_observation(ts, pct);
        }

        // Provide baseline so historical features are populated
        let mut baseline = Vec::with_capacity(168);
        for wd in 0..7 {
            for h in 0..24 {
                baseline.push(HourlyAverage {
                    weekday: wd,
                    hour: h,
                    avg_percentage: 40.0 + f64::from(h),
                    sample_count: 20,
                });
            }
        }
        predictor.update_baseline(&baseline);

        Ok(predictor)
    }

    #[test]
    fn test_predict_autoregressive_features_differ_across_hours() -> Result<()> {
        let predictor = predictor_with_trending_data()?;

        // Clock at 12:00, observations end at 12:00, predict 6 hours ahead
        let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 12, 0, 0).unwrap());
        let schedule = test_schedule();
        let baseline: Vec<HourlyAverage> = (0..7)
            .flat_map(|wd| {
                (0..24).map(move |h| HourlyAverage {
                    weekday: wd,
                    hour: h,
                    avg_percentage: 40.0 + f64::from(h),
                    sample_count: 20,
                })
            })
            .collect();

        let predictions = predictor.predict(&baseline, &schedule, &clock);

        // With a 6-hour horizon on a weekday at 12:00 (open until 23:00),
        // we should get predictions for hours 13-18 (6 predictions).
        assert!(
            predictions.len() >= 2,
            "Expected at least 2 predictions, got {}",
            predictions.len()
        );

        // The key assertion: predictions should NOT all be identical.
        // With frozen lag features (the bug), a linear model would produce
        // very similar values since only cyclical features change.
        let values: Vec<f64> = predictions.iter().map(|p| p.predicted_value).collect();
        let all_same = values.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        assert!(
            !all_same,
            "Predictions should differ across hours due to autoregressive feedback, but all \
             values were identical: {values:?}"
        );

        Ok(())
    }

    #[test]
    fn test_predict_working_buffer_does_not_mutate_recent_data() -> Result<()> {
        let predictor = predictor_with_trending_data()?;
        let original_len = predictor.recent_data.len();
        let original_last = predictor.recent_data.back().copied();

        let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 12, 0, 0).unwrap());
        let schedule = test_schedule();
        let baseline: Vec<HourlyAverage> = (0..7)
            .flat_map(|wd| {
                (0..24).map(move |h| HourlyAverage {
                    weekday: wd,
                    hour: h,
                    avg_percentage: 40.0 + f64::from(h),
                    sample_count: 20,
                })
            })
            .collect();

        let _predictions = predictor.predict(&baseline, &schedule, &clock);

        // recent_data must be unchanged after predict()
        assert_eq!(predictor.recent_data.len(), original_len);
        assert_eq!(predictor.recent_data.back().copied(), original_last);

        Ok(())
    }

    #[test]
    fn test_predict_empty_recent_data_still_works() -> Result<()> {
        let predictor = predictor_with_lr_model()?;
        // No observations added — recent_data is empty

        let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 12, 0, 0).unwrap());
        let schedule = test_schedule();
        let baseline: Vec<HourlyAverage> = (0..7)
            .flat_map(|wd| {
                (0..24).map(move |h| HourlyAverage {
                    weekday: wd,
                    hour: h,
                    avg_percentage: 40.0 + f64::from(h),
                    sample_count: 20,
                })
            })
            .collect();

        let predictions = predictor.predict(&baseline, &schedule, &clock);

        // Should still produce predictions (from ML with default features)
        assert!(
            !predictions.is_empty(),
            "Predictions should be produced even with empty recent_data"
        );
        for p in &predictions {
            assert!(
                (0.0..=100.0).contains(&p.predicted_value),
                "Predicted value {} out of range",
                p.predicted_value
            );
        }

        Ok(())
    }

    #[test]
    fn test_predict_single_different_recent_data_different_results() -> Result<()> {
        let predictor = predictor_with_lr_model()?;

        let target = Utc.with_ymd_and_hms(2024, 6, 17, 14, 0, 0).unwrap();
        let schedule = test_schedule();
        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 14,
            avg_percentage: 50.0,
            sample_count: 20,
        }];

        // Buffer with high occupancy values
        let now = Utc.with_ymd_and_hms(2024, 6, 17, 13, 0, 0).unwrap();
        let high_buffer: VecDeque<(DateTime<Utc>, f64)> = (0..60)
            .map(|i| (now - chrono::Duration::minutes(i), 85.0))
            .collect();

        // Buffer with low occupancy values
        let low_buffer: VecDeque<(DateTime<Utc>, f64)> = (0..60)
            .map(|i| (now - chrono::Duration::minutes(i), 15.0))
            .collect();

        let pred_high = predictor.predict_single(target, 1, &baseline, &schedule, &high_buffer);
        let pred_low = predictor.predict_single(target, 1, &baseline, &schedule, &low_buffer);

        assert!(
            (pred_high.predicted_value - pred_low.predicted_value).abs() > 1e-10,
            "Different recent_data should produce different predictions, got high={} low={}",
            pred_high.predicted_value,
            pred_low.predicted_value
        );

        Ok(())
    }

    #[test]
    fn test_predict_all_values_in_valid_range() -> Result<()> {
        let predictor = predictor_with_trending_data()?;

        let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 12, 0, 0).unwrap());
        let schedule = test_schedule();
        let baseline: Vec<HourlyAverage> = (0..7)
            .flat_map(|wd| {
                (0..24).map(move |h| HourlyAverage {
                    weekday: wd,
                    hour: h,
                    avg_percentage: 40.0 + f64::from(h),
                    sample_count: 20,
                })
            })
            .collect();

        let predictions = predictor.predict(&baseline, &schedule, &clock);

        for p in &predictions {
            assert!(
                (0.0..=100.0).contains(&p.predicted_value),
                "predicted_value {} out of [0, 100]",
                p.predicted_value
            );
            assert!(
                (0.0..=100.0).contains(&p.confidence_low),
                "confidence_low {} out of [0, 100]",
                p.confidence_low
            );
            assert!(
                (0.0..=100.0).contains(&p.confidence_high),
                "confidence_high {} out of [0, 100]",
                p.confidence_high
            );
            assert!(
                p.confidence_low <= p.confidence_high,
                "confidence_low {} > confidence_high {}",
                p.confidence_low,
                p.confidence_high
            );
        }

        Ok(())
    }
}
