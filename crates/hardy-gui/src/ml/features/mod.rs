mod cyclical;
mod momentum;

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Datelike, Local, Timelike, Utc};
use hardy_core::{
    db::HourlyAverage,
    schedule::{GymSchedule, is_bavarian_holiday},
};

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionFeatures {
    pub hour_sin: f64,
    pub hour_cos: f64,
    pub weekday_sin: f64,
    pub weekday_cos: f64,

    pub historical_avg: f64,
    pub historical_std: f64,

    pub recent_avg_1h: f64,
    pub recent_avg_3h: f64,
    pub recent_trend: f64,

    pub day_avg_so_far: f64,
    pub prev_day_avg: f64,

    pub is_weekend: f64,
    pub is_holiday: f64,
    pub week_of_year_sin: f64,
    pub week_of_year_cos: f64,

    pub hours_ahead: f64,

    pub raw_hour: f64,
    pub raw_weekday: f64,
    pub time_to_close: f64,
    pub occupancy_volatility: f64,
    pub recent_avg_6h: f64,
    pub prev_week_same_slot: f64,
}

impl PredictionFeatures {
    pub const NUM_FEATURES: usize = 22;

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.hour_sin,
            self.hour_cos,
            self.weekday_sin,
            self.weekday_cos,
            self.historical_avg,
            self.historical_std,
            self.recent_avg_1h,
            self.recent_avg_3h,
            self.recent_trend,
            self.day_avg_so_far,
            self.prev_day_avg,
            self.is_weekend,
            self.is_holiday,
            self.week_of_year_sin,
            self.week_of_year_cos,
            self.hours_ahead,
            self.raw_hour,
            self.raw_weekday,
            self.time_to_close,
            self.occupancy_volatility,
            self.recent_avg_6h,
            self.prev_week_same_slot,
        ]
    }

    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "historical_avg",
            "historical_std",
            "recent_avg_1h",
            "recent_avg_3h",
            "recent_trend",
            "day_avg_so_far",
            "prev_day_avg",
            "is_weekend",
            "is_holiday",
            "week_of_year_sin",
            "week_of_year_cos",
            "hours_ahead",
            "raw_hour",
            "raw_weekday",
            "time_to_close",
            "occupancy_volatility",
            "recent_avg_6h",
            "prev_week_same_slot",
        ]
    }
}

#[derive(Debug, Clone, Default)]
pub struct SlotStats {
    pub mean: f64,
    pub std_dev: f64,
    pub sample_count: i64,
}

#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    historical_stats: HashMap<(i32, i32), SlotStats>,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            historical_stats: HashMap::new(),
        }
    }

    pub fn update_historical_stats(&mut self, baseline: &[HourlyAverage]) {
        self.historical_stats.clear();

        let mut groups: HashMap<(i32, i32), Vec<f64>> = HashMap::new();

        for avg in baseline {
            let key = (avg.weekday, avg.hour);
            groups.entry(key).or_default().push(avg.avg_percentage);
        }

        for (key, values) in groups {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = values.len() as f64;

            let mean = values.iter().sum::<f64>() / count_f64;

            #[allow(clippy::cast_precision_loss)]
            let degrees_of_freedom = (values.len() - 1) as f64;
            let variance = if values.len() > 1 {
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / degrees_of_freedom
            } else {
                0.0
            };

            self.historical_stats.insert(
                key,
                SlotStats {
                    mean,
                    std_dev: variance.sqrt(),
                    sample_count: i64::try_from(values.len()).unwrap_or_default(),
                },
            );
        }
    }

    pub fn get_slot_std(&self, weekday: i32, hour: i32) -> Option<f64> {
        self.historical_stats
            .get(&(weekday, hour))
            .map(|s| s.std_dev)
    }

    pub fn get_slot_stats(&self, weekday: i32, hour: i32) -> Option<&SlotStats> {
        self.historical_stats.get(&(weekday, hour))
    }

    pub fn extract(
        &self,
        target_time: DateTime<Utc>,
        hours_ahead: i64,
        recent_data: &VecDeque<(DateTime<Utc>, f64)>,
        baseline: &[HourlyAverage],
        schedule: &GymSchedule,
    ) -> PredictionFeatures {
        let local_time = target_time.with_timezone(&Local);
        let hour = local_time.hour().cast_signed();
        let weekday = local_time.weekday().num_days_from_monday().cast_signed();
        let week_of_year = local_time.iso_week().week();

        let (hour_sin, hour_cos) = cyclical::cyclical_encode(f64::from(hour), 24.0);
        let (weekday_sin, weekday_cos) = cyclical::cyclical_encode(f64::from(weekday), 7.0);
        let (week_of_year_sin, week_of_year_cos) =
            cyclical::cyclical_encode(f64::from(week_of_year), 52.0);

        let (historical_avg, historical_std) = self
            .historical_stats
            .get(&(weekday, hour))
            .map(|s| (s.mean, s.std_dev))
            .or_else(|| {
                baseline
                    .iter()
                    .find(|b| b.weekday == weekday && b.hour == hour)
                    .map(|b| (b.avg_percentage, 10.0))
            })
            .unwrap_or((50.0, 15.0));

        let m = momentum::extract_all_momentum(recent_data, &local_time);

        let is_weekend = if weekday >= 5 { 1.0 } else { 0.0 };
        let is_holiday = if is_bavarian_holiday(local_time.date_naive()) {
            1.0
        } else {
            0.0
        };

        #[allow(clippy::cast_precision_loss)]
        let hours_ahead_f64 = hours_ahead as f64;

        let raw_hour = f64::from(hour);
        let raw_weekday = f64::from(weekday);

        let close_hour = schedule.get_close_hour(local_time.date_naive());
        let current_hour = local_time.hour();
        let time_to_close = if current_hour >= close_hour {
            0.0
        } else {
            f64::from(close_hour - current_hour)
        };

        let prev_week_same_slot = historical_avg;

        PredictionFeatures {
            hour_sin,
            hour_cos,
            weekday_sin,
            weekday_cos,
            historical_avg,
            historical_std,
            recent_avg_1h: m.recent_avg_1h,
            recent_avg_3h: m.recent_avg_3h,
            recent_trend: m.recent_trend,
            day_avg_so_far: m.day_avg_so_far,
            prev_day_avg: m.prev_day_avg,
            is_weekend,
            is_holiday,
            week_of_year_sin,
            week_of_year_cos,
            hours_ahead: hours_ahead_f64,
            raw_hour,
            raw_weekday,
            time_to_close,
            occupancy_volatility: m.occupancy_volatility,
            recent_avg_6h: m.recent_avg_6h,
            prev_week_same_slot,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new();
        assert!(extractor.historical_stats.is_empty());
    }

    #[test]
    fn test_update_historical_stats() -> Result<()> {
        let mut extractor = FeatureExtractor::new();

        let baseline = vec![
            HourlyAverage {
                weekday: 0,
                hour: 10,
                avg_percentage: 40.0,
                sample_count: 10,
            },
            HourlyAverage {
                weekday: 0,
                hour: 10,
                avg_percentage: 50.0,
                sample_count: 10,
            },
            HourlyAverage {
                weekday: 0,
                hour: 10,
                avg_percentage: 60.0,
                sample_count: 10,
            },
        ];

        extractor.update_historical_stats(&baseline);

        let stats = extractor.get_slot_stats(0, 10).ok_or_else(|| {
            anyhow::anyhow!("Expected stats for weekday 0 at hour 10 to be present after update")
        })?;

        assert_relative_eq!(stats.mean, 50.0, epsilon = 1e-10);
        assert!(stats.std_dev > 0.0);
        Ok(())
    }

    #[test]
    fn test_features_to_vec() {
        let features = PredictionFeatures {
            hour_sin: 0.5,
            hour_cos: 0.866,
            weekday_sin: 0.0,
            weekday_cos: 1.0,
            historical_avg: 45.0,
            historical_std: 10.0,
            recent_avg_1h: 50.0,
            recent_avg_3h: 48.0,
            recent_trend: 2.0,
            day_avg_so_far: 42.0,
            prev_day_avg: 55.0,
            is_weekend: 0.0,
            is_holiday: 0.0,
            week_of_year_sin: 0.5,
            week_of_year_cos: 0.866,
            hours_ahead: 1.0,
            raw_hour: 14.0,
            raw_weekday: 2.0,
            time_to_close: 9.0,
            occupancy_volatility: 5.0,
            recent_avg_6h: 47.0,
            prev_week_same_slot: 43.0,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), PredictionFeatures::NUM_FEATURES);
    }

    #[test]
    fn test_feature_names_count() {
        let names = PredictionFeatures::feature_names();
        assert_eq!(names.len(), PredictionFeatures::NUM_FEATURES);
    }

    // ── Property-based tests for PredictionFeatures (Step 7) ─────────

    fn arb_prediction_features() -> impl Strategy<Value = PredictionFeatures> {
        // Split into three groups to stay within proptest's 12-element tuple limit.
        let group1 = (
            -1.0_f64..=1.0,  // hour_sin
            -1.0_f64..=1.0,  // hour_cos
            -1.0_f64..=1.0,  // weekday_sin
            -1.0_f64..=1.0,  // weekday_cos
            0.0_f64..=100.0, // historical_avg
            0.0_f64..=50.0,  // historical_std
            0.0_f64..=100.0, // recent_avg_1h
            0.0_f64..=100.0, // recent_avg_3h
        );
        let group2 = (
            -50.0_f64..=50.0,                                // recent_trend
            0.0_f64..=100.0,                                 // day_avg_so_far
            0.0_f64..=100.0,                                 // prev_day_avg
            proptest::prop_oneof![Just(0.0_f64), Just(1.0)], // is_weekend
            proptest::prop_oneof![Just(0.0_f64), Just(1.0)], // is_holiday
            -1.0_f64..=1.0,                                  // week_of_year_sin
            -1.0_f64..=1.0,                                  // week_of_year_cos
            0.0_f64..=24.0,                                  // hours_ahead
        );
        let group3 = (
            0.0_f64..=23.0,  // raw_hour
            0.0_f64..=6.0,   // raw_weekday
            0.0_f64..=17.0,  // time_to_close
            0.0_f64..=50.0,  // occupancy_volatility
            0.0_f64..=100.0, // recent_avg_6h
            0.0_f64..=100.0, // prev_week_same_slot
        );
        (group1, group2, group3).prop_map(
            |(
                (
                    hour_sin,
                    hour_cos,
                    weekday_sin,
                    weekday_cos,
                    historical_avg,
                    historical_std,
                    recent_avg_1h,
                    recent_avg_3h,
                ),
                (
                    recent_trend,
                    day_avg_so_far,
                    prev_day_avg,
                    is_weekend,
                    is_holiday,
                    week_of_year_sin,
                    week_of_year_cos,
                    hours_ahead,
                ),
                (
                    raw_hour,
                    raw_weekday,
                    time_to_close,
                    occupancy_volatility,
                    recent_avg_6h,
                    prev_week_same_slot,
                ),
            )| {
                PredictionFeatures {
                    hour_sin,
                    hour_cos,
                    weekday_sin,
                    weekday_cos,
                    historical_avg,
                    historical_std,
                    recent_avg_1h,
                    recent_avg_3h,
                    recent_trend,
                    day_avg_so_far,
                    prev_day_avg,
                    is_weekend,
                    is_holiday,
                    week_of_year_sin,
                    week_of_year_cos,
                    hours_ahead,
                    raw_hour,
                    raw_weekday,
                    time_to_close,
                    occupancy_volatility,
                    recent_avg_6h,
                    prev_week_same_slot,
                }
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_to_vec_correct_length(features in arb_prediction_features()) {
            prop_assert_eq!(
                features.to_vec().len(),
                PredictionFeatures::NUM_FEATURES,
                "to_vec() length must equal NUM_FEATURES"
            );
        }

        #[test]
        fn prop_all_features_finite(features in arb_prediction_features()) {
            for (i, v) in features.to_vec().iter().enumerate() {
                prop_assert!(
                    v.is_finite(),
                    "Feature at index {i} is not finite: {v}"
                );
            }
        }

        #[test]
        fn prop_to_vec_deterministic(features in arb_prediction_features()) {
            let v1 = features.to_vec();
            let v2 = features.to_vec();
            prop_assert_eq!(v1, v2, "to_vec() must be deterministic");
        }
    }

    #[test]
    fn test_feature_names_match_num_features() {
        assert_eq!(
            PredictionFeatures::feature_names().len(),
            PredictionFeatures::NUM_FEATURES
        );
    }

    // ── New feature extraction tests ──────────────────────────────────

    #[test]
    fn test_extract_raw_hour_and_weekday() {
        let extractor = FeatureExtractor::new();
        // Monday 2024-06-17 at 14:00 UTC
        let timestamp = chrono::TimeZone::with_ymd_and_hms(&Utc, 2024, 6, 17, 14, 0, 0);
        let ts = timestamp.unwrap();
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let schedule = GymSchedule::default();

        let features = extractor.extract(ts, 0, &recent, &[], &schedule);

        let local = ts.with_timezone(&Local);
        let expected_hour = f64::from(local.hour().cast_signed());
        let expected_weekday = f64::from(local.weekday().num_days_from_monday().cast_signed());

        assert_relative_eq!(features.raw_hour, expected_hour, epsilon = 1e-10);
        assert_relative_eq!(features.raw_weekday, expected_weekday, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_time_to_close_weekday() {
        let extractor = FeatureExtractor::new();
        // Monday 2024-06-17 at 14:00 local → close at 23 → 9 hours to close
        // We need to pick a UTC time that maps to hour 14 local.
        let local_14 = chrono::TimeZone::with_ymd_and_hms(&Local, 2024, 6, 17, 14, 0, 0);
        let ts = local_14.unwrap().to_utc();
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let schedule = GymSchedule::default(); // weekday_close = 23

        let features = extractor.extract(ts, 0, &recent, &[], &schedule);

        assert_relative_eq!(features.time_to_close, 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_time_to_close_clamped() {
        let extractor = FeatureExtractor::new();
        // Monday at 23:00 local → close at 23 → clamped to 0.0
        let local_23 = chrono::TimeZone::with_ymd_and_hms(&Local, 2024, 6, 17, 23, 0, 0);
        let ts = local_23.unwrap().to_utc();
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let schedule = GymSchedule::default();

        let features = extractor.extract(ts, 0, &recent, &[], &schedule);

        assert_relative_eq!(features.time_to_close, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_volatility_integrated() {
        let extractor = FeatureExtractor::new();
        let now = Utc::now();
        let recent: VecDeque<(DateTime<Utc>, f64)> = (0..60)
            .map(|i: i64| {
                let remainder = i % 10;
                (
                    now - chrono::Duration::minutes(i),
                    30.0 + f64::from(i32::try_from(remainder).unwrap_or_default()),
                )
            })
            .collect();
        let schedule = GymSchedule::default();

        let features = extractor.extract(now, 0, &recent, &[], &schedule);

        assert!(features.occupancy_volatility.is_finite());
        assert!(features.occupancy_volatility >= 0.0);
    }

    #[test]
    fn test_extract_prev_week_defaults_to_historical() {
        let mut extractor = FeatureExtractor::new();
        let baseline = vec![HourlyAverage {
            weekday: 0, // Monday
            hour: 14,
            avg_percentage: 42.0,
            sample_count: 10,
        }];
        extractor.update_historical_stats(&baseline);

        // Monday at hour 14 local
        let local_14 = chrono::TimeZone::with_ymd_and_hms(&Local, 2024, 6, 17, 14, 0, 0);
        let ts = local_14.unwrap().to_utc();
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let schedule = GymSchedule::default();

        let features = extractor.extract(ts, 0, &recent, &baseline, &schedule);

        // prev_week_same_slot should default to historical_avg
        assert_relative_eq!(
            features.prev_week_same_slot,
            features.historical_avg,
            epsilon = 1e-10
        );
    }
}
