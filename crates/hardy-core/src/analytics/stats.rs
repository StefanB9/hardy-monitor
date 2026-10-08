//! Summary statistics and per-weekday analysis.

use super::time::DAY_NAMES_LONG;
use crate::db::HourlyAverage;

/// Distribution of hourly averages.
#[derive(Debug, Clone)]
pub struct OccupancyStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub sample_count: usize,
    pub coefficient_of_variation: f64,
}

/// Averages, peak and quietest hour of one weekday.
#[derive(Debug, Clone)]
pub struct DayAnalysis {
    pub weekday: i32,
    pub day_name: &'static str,
    pub avg_occupancy: f64,
    pub peak_hour: Option<i32>,
    pub peak_occupancy: f64,
    pub quietest_hour: Option<i32>,
    pub quietest_occupancy: f64,
    pub sample_count: i64,
}

/// Statistics over all slots; `None` without data.
#[tracing::instrument(skip_all, fields(n = data.len()))]
pub fn calculate_stats(data: &[HourlyAverage]) -> Option<OccupancyStats> {
    if data.is_empty() {
        return None;
    }

    let n = data.len();

    #[allow(clippy::cast_precision_loss)]
    let n_f64 = n as f64;

    let mean = data.iter().map(|h| h.avg_percentage).sum::<f64>() / n_f64;

    let mut sorted: Vec<f64> = data.iter().map(|h| h.avg_percentage).collect();
    sorted.sort_by(f64::total_cmp);
    let median = if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    };

    let variance = data
        .iter()
        .map(|h| (h.avg_percentage - mean).powi(2))
        .sum::<f64>()
        / n_f64;
    let std_dev = variance.sqrt();

    let min = sorted[0];
    let max = sorted[n - 1];

    let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

    Some(OccupancyStats {
        mean,
        median,
        std_dev,
        min,
        max,
        sample_count: n,
        coefficient_of_variation,
    })
}

/// Per-weekday analysis for the weekdays present in `data`.
#[tracing::instrument(skip_all)]
pub fn analyze_days(data: &[HourlyAverage]) -> Vec<DayAnalysis> {
    (0..7)
        .map(|weekday| {
            let day_data: Vec<_> = data.iter().filter(|h| h.weekday == weekday).collect();

            let total_samples: i64 = day_data.iter().map(|h| h.sample_count).sum();
            let weighted_sum: f64 = day_data
                .iter()
                .map(|h| {
                    #[allow(clippy::cast_precision_loss)]
                    let count_f64 = h.sample_count as f64;
                    h.avg_percentage * count_f64
                })
                .sum();
            let avg_occupancy = if total_samples > 0 {
                #[allow(clippy::cast_precision_loss)]
                let total_samples_f64 = total_samples as f64;

                weighted_sum / total_samples_f64
            } else {
                0.0
            };

            let peak = day_data
                .iter()
                .max_by(|a, b| a.avg_percentage.total_cmp(&b.avg_percentage));

            let quietest = day_data
                .iter()
                .min_by(|a, b| a.avg_percentage.total_cmp(&b.avg_percentage));

            let weekday_idx = usize::try_from(weekday).unwrap_or(0);
            DayAnalysis {
                weekday,
                day_name: DAY_NAMES_LONG[weekday_idx],
                avg_occupancy,
                peak_hour: peak.map(|h| h.hour),
                peak_occupancy: peak.map_or(0.0, |h| h.avg_percentage),
                quietest_hour: quietest.map(|h| h.hour),
                quietest_occupancy: quietest.map_or(0.0, |h| h.avg_percentage),
                sample_count: total_samples,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    fn make_hourly_avg(weekday: i32, hour: i32, pct: f64, samples: i64) -> HourlyAverage {
        HourlyAverage {
            weekday,
            hour,
            avg_percentage: pct,
            sample_count: samples,
        }
    }

    #[test]
    fn test_calculate_stats_empty() {
        let result = calculate_stats(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_calculate_stats_single_value() -> Result<()> {
        let data = vec![make_hourly_avg(0, 10, 50.0, 5)];
        let result = calculate_stats(&data).ok_or_else(|| anyhow::anyhow!("Expected stats"))?;

        assert_relative_eq!(result.mean, 50.0);
        assert_relative_eq!(result.median, 50.0);
        assert_relative_eq!(result.std_dev, 0.0);
        assert_relative_eq!(result.min, 50.0);
        assert_relative_eq!(result.max, 50.0);
        assert_eq!(result.sample_count, 1);
        Ok(())
    }

    #[test]
    fn test_calculate_stats_multiple_values() -> Result<()> {
        let data = vec![
            make_hourly_avg(0, 10, 20.0, 5),
            make_hourly_avg(0, 11, 40.0, 5),
            make_hourly_avg(0, 12, 60.0, 5),
            make_hourly_avg(0, 13, 80.0, 5),
        ];
        let result = calculate_stats(&data)
            .ok_or_else(|| anyhow::anyhow!("Expected stats for multiple values"))?;

        assert_relative_eq!(result.mean, 50.0);
        assert_relative_eq!(result.median, 50.0);
        assert_relative_eq!(result.min, 20.0);
        assert_relative_eq!(result.max, 80.0);
        assert_eq!(result.sample_count, 4);
        assert!(result.std_dev > 0.0);
        Ok(())
    }

    #[test]
    fn test_analyze_days() {
        let data = vec![
            make_hourly_avg(0, 10, 30.0, 5),
            make_hourly_avg(0, 11, 50.0, 5),
            make_hourly_avg(1, 10, 40.0, 5),
        ];

        let result = analyze_days(&data);

        assert_eq!(result.len(), 7);

        assert_eq!(result[0].weekday, 0);
        assert_eq!(result[0].day_name, "Monday");
        assert_eq!(result[0].peak_hour, Some(11));
        assert_relative_eq!(result[0].peak_occupancy, 50.0);
        assert_eq!(result[0].quietest_hour, Some(10));
        assert_relative_eq!(result[0].quietest_occupancy, 30.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn calculate_stats_orders_its_summary(
            values in proptest::collection::vec(0.0f64..100.0, 1..60),
        ) {
            let data: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(i, v)| make_hourly_avg(i32::try_from(i % 7).unwrap_or(0), 10, *v, 1))
                .collect();
            let stats = calculate_stats(&data);
            prop_assert!(stats.is_some());
            if let Some(s) = stats {
                prop_assert_eq!(s.sample_count, values.len());
                prop_assert!(s.min <= s.median + 1e-9 && s.median <= s.max + 1e-9);
                prop_assert!(s.min <= s.mean + 1e-9 && s.mean <= s.max + 1e-9);
                prop_assert!(s.std_dev >= 0.0);
            }
        }
    }
}
