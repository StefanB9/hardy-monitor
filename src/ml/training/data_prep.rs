use std::collections::VecDeque;

use chrono::{DateTime, Utc};

use crate::{
    db::{HourlyAverage, OccupancyLog},
    ml::{
        MlConfig,
        features::{FeatureExtractor, PredictionFeatures},
        model::TrainingError,
    },
    schedule::GymSchedule,
};

/// Prepares training data (features + targets) from raw occupancy logs.
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

/// Estimate the number of samples per hour from the training data.
///
/// Uses the first and last timestamps plus total sample count to derive
/// average data density. Returns at least 1 to avoid division by zero.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn estimate_samples_per_hour(logs: &[OccupancyLog]) -> usize {
    if logs.len() < 2 {
        return 1;
    }

    let first = logs.first().map(|l| l.timestamp);
    let last = logs.last().map(|l| l.timestamp);

    match (first, last) {
        (Some(f), Some(l)) => {
            let duration_seconds = (l - f).num_seconds();
            if duration_seconds <= 0 {
                return 1;
            }
            let duration_hours = duration_seconds as f64 / 3600.0;
            let sph = logs.len() as f64 / duration_hours;
            (sph.round() as usize).max(1)
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{Duration, TimeZone};

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

    // ── TrainingDataPreparer ─────────────────────────────────────────

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

    // ── estimate_samples_per_hour ────────────────────────────────────

    #[test]
    fn test_estimate_samples_per_hour_minute_data() {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();

        // 121 samples spanning exactly 2 hours (0..=120 minutes) → 121/2.0 ≈ 61
        let logs: Vec<OccupancyLog> = (0..=120)
            .map(|i| OccupancyLog {
                id: i64::from(i),
                timestamp: base_time + Duration::minutes(i64::from(i)),
                percentage: 50.0,
            })
            .collect();

        let sph = estimate_samples_per_hour(&logs);
        // 121 samples / 2.0 hours = 60.5 → rounds to 61
        assert_eq!(sph, 61);
    }

    #[test]
    fn test_estimate_samples_per_hour_hourly_data() {
        let logs = create_test_logs(48); // 48 hourly samples
        let sph = estimate_samples_per_hour(&logs);
        // 48 samples over 47 hours ≈ 1 sph
        assert_eq!(sph, 1);
    }

    #[test]
    fn test_estimate_samples_per_hour_empty() {
        let logs: Vec<OccupancyLog> = Vec::new();
        assert_eq!(estimate_samples_per_hour(&logs), 1);
    }

    #[test]
    fn test_estimate_samples_per_hour_single() {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();
        let logs = vec![OccupancyLog {
            id: 1,
            timestamp: base_time,
            percentage: 50.0,
        }];
        assert_eq!(estimate_samples_per_hour(&logs), 1);
    }
}
