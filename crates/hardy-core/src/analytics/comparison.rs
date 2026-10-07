//! Comparing two periods hour by hour, and the resulting trend.

use std::collections::{BTreeSet, HashMap};

use super::{TrendDirection, determine_trend};
use crate::db::HourlyAverage;

/// Slots quieter than this (percent) are not reported as significant changes.
const SIGNIFICANT_CHANGE_MIN_LEVEL: f64 = 10.0;

/// Changes smaller than this (percentage points) are not significant.
const SIGNIFICANT_CHANGE_MIN_POINTS: f64 = 5.0;

/// How far apart the compared periods are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    WeekOverWeek,
    MonthOverMonth,
    CustomRange,
}

/// One weekday/hour slot in both periods.
#[derive(Debug, Clone)]
pub struct HourlyComparison {
    pub weekday: i32,
    pub hour: i32,
    pub baseline_avg: f64,
    pub current_avg: f64,
    pub absolute_change: f64,
    pub percent_change: f64,
    pub baseline_samples: i64,
    pub current_samples: i64,
}

impl HourlyComparison {
    /// Direction of this slot's change.
    pub fn trend(&self) -> TrendDirection {
        if self.baseline_samples < 2 || self.current_samples < 2 {
            return TrendDirection::Insufficient;
        }
        if self.percent_change > 5.0 {
            TrendDirection::Increasing
        } else if self.percent_change < -5.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Result of comparing a baseline period with the current one.
#[derive(Debug, Clone)]
pub struct PeriodComparison {
    pub mode: ComparisonMode,
    pub baseline_overall_avg: f64,
    pub current_overall_avg: f64,
    pub overall_change_percent: f64,
    pub overall_trend: TrendDirection,
    pub hourly_comparisons: Vec<HourlyComparison>,
    pub biggest_increases: Vec<(i32, i32, f64)>,
    pub biggest_decreases: Vec<(i32, i32, f64)>,
}

/// Pairs the slots present in either period.
#[tracing::instrument(skip_all, fields(baseline.len = baseline.len(), current.len = current.len()))]
pub fn build_hourly_comparisons(
    baseline: &[HourlyAverage],
    current: &[HourlyAverage],
) -> Vec<HourlyComparison> {
    let mut comparisons = Vec::new();

    let baseline_map: HashMap<(i32, i32), &HourlyAverage> =
        baseline.iter().map(|h| ((h.weekday, h.hour), h)).collect();

    let current_map: HashMap<(i32, i32), &HourlyAverage> =
        current.iter().map(|h| ((h.weekday, h.hour), h)).collect();

    let all_keys: BTreeSet<(i32, i32)> = baseline_map
        .keys()
        .chain(current_map.keys())
        .copied()
        .collect();

    for (weekday, hour) in all_keys {
        let baseline_data = baseline_map.get(&(weekday, hour));
        let current_data = current_map.get(&(weekday, hour));

        let baseline_avg = baseline_data.map_or(0.0, |d| d.avg_percentage);
        let current_avg = current_data.map_or(0.0, |d| d.avg_percentage);
        let baseline_samples = baseline_data.map_or(0, |d| d.sample_count);
        let current_samples = current_data.map_or(0, |d| d.sample_count);

        let absolute_change = current_avg - baseline_avg;
        let percent_change = if baseline_avg > 0.0 {
            (absolute_change / baseline_avg) * 100.0
        } else if current_avg > 0.0 {
            100.0
        } else {
            0.0
        };

        comparisons.push(HourlyComparison {
            weekday,
            hour,
            baseline_avg,
            current_avg,
            absolute_change,
            percent_change,
            baseline_samples,
            current_samples,
        });
    }

    comparisons
}

/// Compares two periods slot by slot: biggest changes and the overall trend.
#[tracing::instrument(skip_all, fields(mode = ?mode))]
pub fn compare_periods(
    baseline: &[HourlyAverage],
    current: &[HourlyAverage],
    mode: ComparisonMode,
) -> PeriodComparison {
    let hourly_comparisons = build_hourly_comparisons(baseline, current);

    let baseline_overall_avg = if baseline.is_empty() {
        0.0
    } else {
        let total: f64 = baseline
            .iter()
            .map(|h| {
                #[allow(clippy::cast_precision_loss)]
                let count_f64 = h.sample_count as f64;

                h.avg_percentage * count_f64
            })
            .sum();
        let count: i64 = baseline.iter().map(|h| h.sample_count).sum();
        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = count as f64;

            total / count_f64
        } else {
            0.0
        }
    };

    let current_overall_avg = if current.is_empty() {
        0.0
    } else {
        let total: f64 = current
            .iter()
            .map(|h| {
                #[allow(clippy::cast_precision_loss)]
                let count_f64 = h.sample_count as f64;

                h.avg_percentage * count_f64
            })
            .sum();
        let count: i64 = current.iter().map(|h| h.sample_count).sum();
        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = count as f64;

            total / count_f64
        } else {
            0.0
        }
    };

    let overall_change_percent = if baseline_overall_avg > 0.0 {
        ((current_overall_avg - baseline_overall_avg) / baseline_overall_avg) * 100.0
    } else {
        0.0
    };

    let overall_trend = determine_trend(&hourly_comparisons);

    // Relative changes on near-empty slots (1% -> 7% = +600%) and changes of
    // a few points are noise, not something worth avoiding.
    let mut sorted_by_increase: Vec<_> = hourly_comparisons
        .iter()
        .filter(|c| {
            c.baseline_samples >= 2
                && c.current_samples >= 2
                && c.baseline_avg >= SIGNIFICANT_CHANGE_MIN_LEVEL
                && c.absolute_change.abs() >= SIGNIFICANT_CHANGE_MIN_POINTS
        })
        .collect();
    sorted_by_increase.sort_by(|a, b| b.percent_change.total_cmp(&a.percent_change));

    let biggest_increases: Vec<(i32, i32, f64)> = sorted_by_increase
        .iter()
        .filter(|c| c.percent_change > 0.0)
        .take(3)
        .map(|c| (c.weekday, c.hour, c.percent_change))
        .collect();

    let biggest_decreases: Vec<(i32, i32, f64)> = sorted_by_increase
        .iter()
        .rev()
        .filter(|c| c.percent_change < 0.0)
        .take(3)
        .map(|c| (c.weekday, c.hour, c.percent_change))
        .collect();

    PeriodComparison {
        mode,
        baseline_overall_avg,
        current_overall_avg,
        overall_change_percent,
        overall_trend,
        hourly_comparisons,
        biggest_increases,
        biggest_decreases,
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

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
    fn test_compare_periods_ignores_changes_on_near_empty_slots() {
        // Mon 23:00 goes 1% -> 7% (+600%, irrelevant); Wed 18:00 goes
        // 30% -> 40% (+33%, worth knowing).
        let baseline = vec![
            make_hourly_avg(0, 23, 1.0, 10),
            make_hourly_avg(2, 18, 30.0, 10),
        ];
        let current = vec![
            make_hourly_avg(0, 23, 7.0, 10),
            make_hourly_avg(2, 18, 40.0, 10),
        ];

        let result = compare_periods(&baseline, &current, ComparisonMode::WeekOverWeek);
        let slots: Vec<(i32, i32)> = result
            .biggest_increases
            .iter()
            .map(|&(w, h, _)| (w, h))
            .collect();
        assert_eq!(slots, vec![(2, 18)]);
    }

    #[test]
    fn test_compare_periods_ignores_tiny_absolute_changes() {
        // 40% -> 43% is +7.5% relative but only 3 points: not significant.
        let baseline = vec![make_hourly_avg(1, 10, 40.0, 10)];
        let current = vec![make_hourly_avg(1, 10, 43.0, 10)];
        let result = compare_periods(&baseline, &current, ComparisonMode::WeekOverWeek);
        assert_eq!(result.biggest_increases.len(), 0);
    }

    #[test]
    fn test_build_hourly_comparisons_empty() {
        let result = build_hourly_comparisons(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_hourly_comparisons_basic() {
        let baseline = vec![make_hourly_avg(0, 10, 40.0, 5)];
        let current = vec![make_hourly_avg(0, 10, 50.0, 5)];

        let result = build_hourly_comparisons(&baseline, &current);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].weekday, 0);
        assert_eq!(result[0].hour, 10);
        assert_relative_eq!(result[0].baseline_avg, 40.0);
        assert_relative_eq!(result[0].current_avg, 50.0);
        assert_relative_eq!(result[0].absolute_change, 10.0);
        assert!((result[0].percent_change - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_build_hourly_comparisons_missing_baseline() {
        let baseline = vec![];
        let current = vec![make_hourly_avg(0, 10, 50.0, 5)];

        let result = build_hourly_comparisons(&baseline, &current);

        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].baseline_avg, 0.0);
        assert_relative_eq!(result[0].current_avg, 50.0);
        assert_relative_eq!(result[0].percent_change, 100.0);
    }

    #[test]
    fn test_build_hourly_comparisons_missing_current() {
        let baseline = vec![make_hourly_avg(0, 10, 50.0, 5)];
        let current = vec![];

        let result = build_hourly_comparisons(&baseline, &current);

        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].baseline_avg, 50.0);
        assert_relative_eq!(result[0].current_avg, 0.0);
        assert_relative_eq!(result[0].percent_change, -100.0);
    }

    #[test]
    fn test_compare_periods_basic() {
        let baseline = vec![
            make_hourly_avg(0, 10, 40.0, 10),
            make_hourly_avg(0, 11, 50.0, 10),
        ];
        let current = vec![
            make_hourly_avg(0, 10, 45.0, 10),
            make_hourly_avg(0, 11, 55.0, 10),
        ];

        let result = compare_periods(&baseline, &current, ComparisonMode::WeekOverWeek);

        assert_eq!(result.mode, ComparisonMode::WeekOverWeek);
        assert!(result.current_overall_avg > result.baseline_overall_avg);
        assert!(result.overall_change_percent > 0.0);
    }

    #[test]
    fn test_hourly_comparison_trend() {
        let increasing = HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 40.0,
            current_avg: 50.0,
            absolute_change: 10.0,
            percent_change: 25.0,
            baseline_samples: 10,
            current_samples: 10,
        };
        assert_eq!(increasing.trend(), TrendDirection::Increasing);

        let decreasing = HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 50.0,
            current_avg: 40.0,
            absolute_change: -10.0,
            percent_change: -20.0,
            baseline_samples: 10,
            current_samples: 10,
        };
        assert_eq!(decreasing.trend(), TrendDirection::Decreasing);

        let stable = HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 50.0,
            current_avg: 51.0,
            absolute_change: 1.0,
            percent_change: 2.0,
            baseline_samples: 10,
            current_samples: 10,
        };
        assert_eq!(stable.trend(), TrendDirection::Stable);
    }
}
