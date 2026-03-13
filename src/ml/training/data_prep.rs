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

        let mut recent_window: VecDeque<(DateTime<Utc>, f64)> = VecDeque::with_capacity(360);

        for log in logs {
            let timestamp = log.timestamp;

            while recent_window.len() >= 360 {
                recent_window.pop_front();
            }
            recent_window.push_back((timestamp, log.percentage));

            let mut feature =
                feature_extractor.extract(timestamp, 0, &recent_window, baseline, schedule);

            feature.prev_week_same_slot =
                lookup_prev_week(logs, timestamp).unwrap_or(feature.historical_avg);

            features.push(feature);
            targets.push(log.percentage);
        }

        if features.len() < self.config.min_samples_for_training {
            return Err(TrainingError::InsufficientData(features.len()));
        }

        Ok((features, targets))
    }
}

/// Look up the occupancy value from approximately 7 days before `target_time`.
///
/// Binary-searches the sorted `logs` for an entry within 1 hour of
/// `target_time - 7 days`. Returns `None` if no sufficiently close entry
/// exists.
fn lookup_prev_week(logs: &[OccupancyLog], target_time: DateTime<Utc>) -> Option<f64> {
    let target = target_time - chrono::Duration::days(7);

    let idx = logs.partition_point(|log| log.timestamp < target);

    // Check the entry at idx and idx-1 (the two nearest candidates)
    let candidates = [idx.checked_sub(1), Some(idx)]
        .into_iter()
        .flatten()
        .filter(|&i| i < logs.len());

    let tolerance_seconds = 3600; // 1 hour

    candidates
        .map(|i| &logs[i])
        .filter(|log| {
            let diff = (log.timestamp - target).num_seconds().abs();
            diff <= tolerance_seconds
        })
        .min_by_key(|log| (log.timestamp - target).num_seconds().abs())
        .map(|log| log.percentage)
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
    use approx::assert_relative_eq;
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

    // ── lookup_prev_week ──────────────────────────────────────────────

    #[test]
    fn test_lookup_prev_week_found() {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();

        // Create 2 weeks of hourly logs (336 hours)
        let logs: Vec<OccupancyLog> = (0..336)
            .map(|i| {
                let timestamp = base_time + Duration::hours(i64::from(i));
                OccupancyLog {
                    id: i64::from(i),
                    timestamp,
                    percentage: 30.0 + f64::from(i % 24),
                }
            })
            .collect();

        // Look up value at day 14 (hour 168+6=174), should find value from day 7 (hour
        // 6)
        let target_time = base_time + Duration::hours(174); // 7 days + 6 hours after base
        let result = lookup_prev_week(&logs, target_time);

        assert!(result.is_some(), "Should find a value from 7 days ago");
        // The log at hour 6 has percentage 30.0 + (6 % 24) = 36.0
        assert_relative_eq!(result.unwrap_or(0.0), 36.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lookup_prev_week_not_found() {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();

        // Create only 1 day of logs
        let logs: Vec<OccupancyLog> = (0..24)
            .map(|i| OccupancyLog {
                id: i64::from(i),
                timestamp: base_time + Duration::hours(i64::from(i)),
                percentage: 50.0,
            })
            .collect();

        // target_time - 7 days = base_time + 14 days - 7 days = base_time + 7 days
        // Logs only cover 24 hours from base_time, so 7 days later has no data
        let target_time = base_time + Duration::days(14);
        let result = lookup_prev_week(&logs, target_time);

        assert!(result.is_none(), "Should not find data from 7 days ago");
    }

    #[test]
    fn test_training_preparer_prev_week_values() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            ..Default::default()
        };

        // Create 2+ weeks of hourly logs
        let logs = create_test_logs(400);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config);
        let (features, _targets) = preparer.prepare(&logs, &baseline, &schedule)?;

        // Features after the first week should have non-default prev_week_same_slot
        // (i.e., different from historical_avg for at least some entries)
        let after_first_week = &features[168..]; // hour 168 = start of week 2
        let has_non_default = after_first_week
            .iter()
            .any(|f| (f.prev_week_same_slot - f.historical_avg).abs() > 1e-10);

        assert!(
            has_non_default,
            "After the first week, some prev_week_same_slot values should differ from \
             historical_avg"
        );

        Ok(())
    }
}
