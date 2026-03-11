pub mod cross_validation;

use std::collections::VecDeque;

use chrono::{DateTime, Duration, Utc};

use super::{
    MlConfig,
    features::{FeatureExtractor, PredictionFeatures},
    model::{ModelBuilder, TrainedModel, TrainingError},
    persistence::{ModelSummary, PersistedModel, SerializedSlotStats},
};
use crate::{
    db::{Database, HourlyAverage, OccupancyLog},
    schedule::GymSchedule,
    traits::Clock,
};

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub model: TrainedModel,
    pub feature_extractor: FeatureExtractor,
    pub persisted: PersistedModel,
}

pub struct TrainingDataPreparer {
    config: MlConfig,
}

impl TrainingDataPreparer {
    pub fn new(config: MlConfig) -> Self {
        Self { config }
    }

    pub fn prepare(
        &self,
        logs: &[OccupancyLog],
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> Result<(Vec<PredictionFeatures>, Vec<f64>), TrainingError> {
        if logs.len() < self.config.min_samples_for_training {
            return Err(TrainingError::InsufficientData(logs.len()));
        }

        let mut feature_extractor = FeatureExtractor::new();
        feature_extractor.update_historical_stats(baseline);

        let mut features = Vec::with_capacity(logs.len());
        let mut targets = Vec::with_capacity(logs.len());

        let mut recent_window: VecDeque<(DateTime<Utc>, f64)> = VecDeque::with_capacity(180);

        for log in logs {
            let timestamp = log.timestamp;

            while recent_window.len() >= 180 {
                recent_window.pop_front();
            }
            recent_window.push_back((timestamp, log.percentage));

            let feature =
                feature_extractor.extract(timestamp, 0, &recent_window, baseline, schedule);

            features.push(feature);
            targets.push(log.percentage);
        }

        if features.len() < self.config.min_samples_for_training {
            return Err(TrainingError::InsufficientData(features.len()));
        }

        Ok((features, targets))
    }
}

pub async fn train_model(
    db: &Database,
    clock: &dyn Clock,
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    let end = clock.now_utc();
    let start = end - Duration::days(config.training_window_days);

    let logs = db
        .get_history_range(start, end)
        .await
        .map_err(|e| TrainingError::FitError(format!("Database error: {e}")))?;

    if logs.len() < config.min_samples_for_training {
        return Err(TrainingError::InsufficientData(logs.len()));
    }

    let baseline = db
        .get_averages_range(start, end)
        .await
        .map_err(|e| TrainingError::FitError(format!("Database error: {e}")))?;

    let preparer = TrainingDataPreparer::new(config.clone());
    let (features, targets) = preparer.prepare(&logs, &baseline, schedule)?;

    let builder = ModelBuilder::new()
        .max_depth(10)
        .min_samples_split(5)
        .min_samples_leaf(2)
        .ridge_lambda(1e-3);

    let model = builder.train_with_validation(&features, &targets, 0.2)?;

    let mut feature_extractor = FeatureExtractor::new();
    feature_extractor.update_historical_stats(&baseline);

    let slot_stats: Vec<SerializedSlotStats> = baseline
        .iter()
        .map(|avg| SerializedSlotStats {
            weekday: avg.weekday as u32,
            hour: avg.hour as u32,
            mean: avg.avg_percentage,
            std_dev: 10.0,
            sample_count: avg.sample_count,
        })
        .collect();

    let persisted = PersistedModel::new(
        config.training_window_days,
        model.training_samples,
        model.training_mse,
        model.validation_mse,
        slot_stats,
        ModelSummary {
            model_type: "LinearRegression".to_string(),
            max_depth: Some(10),
            feature_importance: None,
        },
    );

    Ok(TrainingResult {
        model,
        feature_extractor,
        persisted,
    })
}

pub fn train_model_sync(
    logs: &[OccupancyLog],
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    if logs.len() < config.min_samples_for_training {
        return Err(TrainingError::InsufficientData(logs.len()));
    }

    let preparer = TrainingDataPreparer::new(config.clone());
    let (features, targets) = preparer.prepare(logs, baseline, schedule)?;

    let builder = ModelBuilder::new()
        .max_depth(10)
        .min_samples_split(5)
        .min_samples_leaf(2)
        .ridge_lambda(1e-3);

    let model = builder.train_with_validation(&features, &targets, 0.2)?;

    let mut feature_extractor = FeatureExtractor::new();
    feature_extractor.update_historical_stats(baseline);

    let slot_stats: Vec<SerializedSlotStats> = baseline
        .iter()
        .map(|avg| SerializedSlotStats {
            weekday: avg.weekday as u32,
            hour: avg.hour as u32,
            mean: avg.avg_percentage,
            std_dev: 10.0,
            sample_count: avg.sample_count,
        })
        .collect();

    let persisted = PersistedModel::new(
        config.training_window_days,
        model.training_samples,
        model.training_mse,
        model.validation_mse,
        slot_stats,
        ModelSummary {
            model_type: "LinearRegression".to_string(),
            max_depth: Some(10),
            feature_importance: None,
        },
    );

    Ok(TrainingResult {
        model,
        feature_extractor,
        persisted,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::TimeZone;

    use super::*;

    fn create_test_logs(n: i32) -> Vec<OccupancyLog> {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();

        (0..n)
            .map(|i| {
                let timestamp = base_time + Duration::hours(i64::from(i));
                let hour = (6 + i) % 24;
                let weekday = f64::from((i / 24) % 7);
                let percentage =
                    30.0 + (f64::from(hour) * 2.0) + (weekday * 3.0) + f64::from(i % 10);
                OccupancyLog {
                    id: i64::from(i),
                    timestamp,
                    percentage: percentage.min(95.0),
                }
            })
            .collect()
    }

    fn create_test_baseline() -> Vec<HourlyAverage> {
        let mut baseline = Vec::new();
        for weekday in 0..7 {
            for hour in 0..24 {
                baseline.push(HourlyAverage {
                    weekday,
                    hour,
                    avg_percentage: 40.0 + f64::from(hour) + (f64::from(weekday) * 2.0),
                    sample_count: 10,
                });
            }
        }
        baseline
    }

    #[test]
    fn test_training_data_preparer_insufficient_data() {
        let config = MlConfig {
            min_samples_for_training: 100,
            ..Default::default()
        };

        let preparer = TrainingDataPreparer::new(config);
        let logs = create_test_logs(50);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = preparer.prepare(&logs, &baseline, &schedule);

        assert!(matches!(result, Err(TrainingError::InsufficientData(50))));
    }

    #[test]
    fn test_training_data_preparer_success() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            ..Default::default()
        };

        let preparer = TrainingDataPreparer::new(config);
        let logs = create_test_logs(200);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = preparer.prepare(&logs, &baseline, &schedule);

        assert!(result.is_ok());
        let (features, targets) = result?;
        assert_eq!(features.len(), targets.len());
        assert!(features.len() >= 100);
        Ok(())
    }

    #[test]
    fn test_train_model_sync() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config);

        match result {
            Ok(training_result) => {
                assert!(training_result.model.training_samples >= 100);
                assert!(training_result.persisted.training_mse >= 0.0);
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    #[test]
    fn test_train_model_sync_insufficient_data() {
        let config = MlConfig {
            min_samples_for_training: 1000,
            ..Default::default()
        };

        let logs = create_test_logs(100);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config);

        assert!(matches!(result, Err(TrainingError::InsufficientData(_))));
    }
}
