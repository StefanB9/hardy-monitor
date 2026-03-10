use std::{
    collections::{HashMap, VecDeque},
    f64::consts::PI,
};

use chrono::{DateTime, Datelike, Local, Timelike, Utc};

use crate::{
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
}

impl PredictionFeatures {
    pub const NUM_FEATURES: usize = 16;

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
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = if values.len() > 1 {
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
            } else {
                0.0
            };

            self.historical_stats.insert(
                key,
                SlotStats {
                    mean,
                    std_dev: variance.sqrt(),
                    sample_count: values.len() as i64,
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
        _schedule: &GymSchedule,
    ) -> PredictionFeatures {
        let local_time = target_time.with_timezone(&Local);
        let hour = local_time.hour() as i32;
        let weekday = local_time.weekday().num_days_from_monday() as i32;
        let week_of_year = local_time.iso_week().week();

        let (hour_sin, hour_cos) = cyclical_encode(f64::from(hour), 24.0);
        let (weekday_sin, weekday_cos) = cyclical_encode(f64::from(weekday), 7.0);
        let (week_of_year_sin, week_of_year_cos) = cyclical_encode(f64::from(week_of_year), 52.0);

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

        let (recent_avg_1h, recent_avg_3h, recent_trend) = self.extract_momentum(recent_data);

        let (day_avg_so_far, prev_day_avg) = self.extract_day_features(recent_data, &local_time);

        let is_weekend = if weekday >= 5 { 1.0 } else { 0.0 };
        let is_holiday = if is_bavarian_holiday(local_time.date_naive()) {
            1.0
        } else {
            0.0
        };

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
            hours_ahead: hours_ahead as f64,
        }
    }

    fn extract_momentum(&self, recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> (f64, f64, f64) {
        if recent_data.is_empty() {
            return (50.0, 50.0, 0.0);
        }

        let now = recent_data.back().map(|(t, _)| *t).unwrap_or_else(Utc::now);
        let one_hour_ago = now - chrono::Duration::hours(1);
        let three_hours_ago = now - chrono::Duration::hours(3);

        let recent_1h: Vec<f64> = recent_data
            .iter()
            .filter(|(t, _)| *t >= one_hour_ago)
            .map(|(_, v)| *v)
            .collect();
        let recent_avg_1h = if recent_1h.is_empty() {
            50.0
        } else {
            recent_1h.iter().sum::<f64>() / recent_1h.len() as f64
        };

        let recent_3h: Vec<f64> = recent_data
            .iter()
            .filter(|(t, _)| *t >= three_hours_ago)
            .map(|(_, v)| *v)
            .collect();
        let recent_avg_3h = if recent_3h.is_empty() {
            50.0
        } else {
            recent_3h.iter().sum::<f64>() / recent_3h.len() as f64
        };

        let recent_trend = self.calculate_trend(&recent_3h);

        (recent_avg_1h, recent_avg_3h, recent_trend)
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            (numerator / denominator) * 60.0
        }
    }

    fn extract_day_features(
        &self,
        recent_data: &VecDeque<(DateTime<Utc>, f64)>,
        local_time: &DateTime<Local>,
    ) -> (f64, f64) {
        let today = local_time.date_naive();
        let yesterday = today - chrono::Duration::days(1);

        let mut today_values = Vec::new();
        let mut yesterday_values = Vec::new();

        for (timestamp, value) in recent_data {
            let local_ts = timestamp.with_timezone(&Local);
            let date = local_ts.date_naive();

            if date == today {
                today_values.push(*value);
            } else if date == yesterday {
                yesterday_values.push(*value);
            }
        }

        let day_avg_so_far = if today_values.is_empty() {
            50.0
        } else {
            today_values.iter().sum::<f64>() / today_values.len() as f64
        };

        let prev_day_avg = if yesterday_values.is_empty() {
            50.0
        } else {
            yesterday_values.iter().sum::<f64>() / yesterday_values.len() as f64
        };

        (day_avg_so_far, prev_day_avg)
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

fn cyclical_encode(value: f64, period: f64) -> (f64, f64) {
    let angle = 2.0 * PI * value / period;
    (angle.sin(), angle.cos())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_cyclical_encoding_continuity() {
        let (sin_23, cos_23) = cyclical_encode(23.0, 24.0);
        let (sin_0, cos_0) = cyclical_encode(0.0, 24.0);

        let distance = ((sin_23 - sin_0).powi(2) + (cos_23 - cos_0).powi(2)).sqrt();

        assert!(distance < 0.5, "Distance was {distance}");
    }

    #[test]
    fn test_cyclical_encoding_opposite() {
        let (_sin_0, cos_0) = cyclical_encode(0.0, 24.0);
        let (_sin_12, cos_12) = cyclical_encode(12.0, 24.0);

        assert_relative_eq!(cos_0, -cos_12, epsilon = 1e-10);
    }

    #[test]
    fn test_cyclical_encoding_quarter() {
        let (sin_6, cos_6) = cyclical_encode(6.0, 24.0);

        assert_relative_eq!(sin_6, 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos_6, 0.0, epsilon = 1e-10);
    }

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

        let stats = extractor
            .get_slot_stats(0, 10)
            .ok_or_else(|| anyhow::anyhow!("Expected stats for weekday 0 at hour 10 to be present after update"))?;

        assert_relative_eq!(stats.mean, 50.0, epsilon = 1e-10);
        assert!(stats.std_dev > 0.0);
        Ok(())
    }

    #[test]
    fn test_calculate_trend_increasing() {
        let extractor = FeatureExtractor::new();
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let trend = extractor.calculate_trend(&values);

        assert!(
            trend > 0.0,
            "Trend should be positive for increasing values"
        );
    }

    #[test]
    fn test_calculate_trend_decreasing() {
        let extractor = FeatureExtractor::new();
        let values = vec![50.0, 40.0, 30.0, 20.0, 10.0];

        let trend = extractor.calculate_trend(&values);

        assert!(
            trend < 0.0,
            "Trend should be negative for decreasing values"
        );
    }

    #[test]
    fn test_calculate_trend_flat() {
        let extractor = FeatureExtractor::new();
        let values = vec![30.0, 30.0, 30.0, 30.0, 30.0];

        let trend = extractor.calculate_trend(&values);

        assert_relative_eq!(trend, 0.0, epsilon = 1e-10);
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
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), PredictionFeatures::NUM_FEATURES);
    }

    #[test]
    fn test_feature_names_count() {
        let names = PredictionFeatures::feature_names();
        assert_eq!(names.len(), PredictionFeatures::NUM_FEATURES);
    }

    #[test]
    fn test_extract_momentum_empty() {
        let extractor = FeatureExtractor::new();
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();

        let (avg_1h, avg_3h, trend) = extractor.extract_momentum(&recent);

        assert_relative_eq!(avg_1h, 50.0);
        assert_relative_eq!(avg_3h, 50.0);
        assert_relative_eq!(trend, 0.0);
    }
}
