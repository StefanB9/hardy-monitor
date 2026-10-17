//! Comparing two periods hour by hour, and the resulting trend.

use std::collections::{BTreeSet, HashMap};

use super::{MIN_DAYS, TrendDirection, mean_point_change, trend_from_points};
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
    /// Distinct days the slot was seen on in each period.
    pub baseline_days: i64,
    pub current_days: i64,
}

impl HourlyComparison {
    /// Whether the slot was seen on enough days in both periods.
    pub fn is_reliable(&self) -> bool {
        self.baseline_days >= MIN_DAYS && self.current_days >= MIN_DAYS
    }

    /// Direction of this slot's change, by the overall trend's rule.
    pub fn trend(&self) -> TrendDirection {
        trend_from_points(self.is_reliable().then_some(self.absolute_change))
    }
}

/// Result of comparing a baseline period with the current one.
#[derive(Debug, Clone)]
pub struct PeriodComparison {
    pub mode: ComparisonMode,
    pub baseline_overall_avg: f64,
    pub current_overall_avg: f64,
    /// Relative change of the overall average, in percent.
    pub overall_change_percent: f64,
    /// Average change in points per hour, the number behind
    /// `overall_trend`.
    pub mean_point_change: Option<f64>,
    pub overall_trend: TrendDirection,
    pub hourly_comparisons: Vec<HourlyComparison>,
    /// The largest rises and falls in points, largest first.
    pub biggest_increases: Vec<HourlyComparison>,
    pub biggest_decreases: Vec<HourlyComparison>,
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
        let baseline_days = baseline_data.map_or(0, |d| d.days);
        let current_days = current_data.map_or(0, |d| d.days);

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
            baseline_days,
            current_days,
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

    let mean_point_change = mean_point_change(&hourly_comparisons);
    let overall_trend = trend_from_points(mean_point_change);

    // Relative changes on near-empty slots (1% -> 7% = +600%) and changes of
    // a few points are noise, not something worth avoiding.
    let mut significant: Vec<&HourlyComparison> = hourly_comparisons
        .iter()
        .filter(|c| {
            c.is_reliable()
                && c.baseline_avg >= SIGNIFICANT_CHANGE_MIN_LEVEL
                && c.absolute_change.abs() >= SIGNIFICANT_CHANGE_MIN_POINTS
        })
        .collect();
    significant.sort_by(|a, b| b.absolute_change.total_cmp(&a.absolute_change));

    let biggest_increases: Vec<HourlyComparison> = significant
        .iter()
        .filter(|c| c.absolute_change > 0.0)
        .take(3)
        .map(|&c| c.clone())
        .collect();

    let biggest_decreases: Vec<HourlyComparison> = significant
        .iter()
        .rev()
        .filter(|c| c.absolute_change < 0.0)
        .take(3)
        .map(|&c| c.clone())
        .collect();

    PeriodComparison {
        mode,
        baseline_overall_avg,
        current_overall_avg,
        overall_change_percent,
        mean_point_change,
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
            days: samples,
            std_dev: 0.0,
        }
    }

    /// An hour seen on `days` days, with an hour's worth of readings each.
    fn seen(weekday: i32, hour: i32, pct: f64, days: i64) -> HourlyAverage {
        HourlyAverage {
            weekday,
            hour,
            avg_percentage: pct,
            sample_count: 60 * days,
            days,
            std_dev: 0.0,
        }
    }

    #[test]
    fn test_biggest_changes_are_ranked_by_points() {
        // Mon 10:00 10% -> 15% (+50%, +5 points); Tue 18:00 40% -> 52%
        // (+30%, +12 points).
        let baseline = vec![seen(0, 10, 10.0, 4), seen(1, 18, 40.0, 4)];
        let current = vec![seen(0, 10, 15.0, 4), seen(1, 18, 52.0, 4)];
        let result = compare_periods(&baseline, &current, ComparisonMode::CustomRange);
        let ranked: Vec<(i32, i32)> = result
            .biggest_increases
            .iter()
            .map(|c| (c.weekday, c.hour))
            .collect();
        assert_eq!(ranked, vec![(1, 18), (0, 10)]);

        let result = compare_periods(&current, &baseline, ComparisonMode::CustomRange);
        let ranked: Vec<(i32, i32)> = result
            .biggest_decreases
            .iter()
            .map(|c| (c.weekday, c.hour))
            .collect();
        assert_eq!(ranked, vec![(1, 18), (0, 10)]);
    }

    #[test]
    fn test_comparison_needs_two_days_not_two_readings() {
        // 60 readings before, but all on one day.
        let baseline = vec![seen(2, 18, 30.0, 1)];
        let current = vec![seen(2, 18, 45.0, 4)];
        let result = compare_periods(&baseline, &current, ComparisonMode::CustomRange);
        assert!(result.biggest_increases.is_empty());
        assert!(!result.hourly_comparisons[0].is_reliable());
    }

    #[test]
    fn test_comparison_reports_the_trend_number() -> anyhow::Result<()> {
        let baseline: Vec<_> = (8..20).map(|h| seen(3, h, 30.0, 4)).collect();
        let current: Vec<_> = (8..20).map(|h| seen(3, h, 32.0, 4)).collect();
        let result = compare_periods(&baseline, &current, ComparisonMode::CustomRange);
        let change = result
            .mean_point_change
            .ok_or_else(|| anyhow::anyhow!("no change"))?;
        assert_relative_eq!(change, 2.0, epsilon = 1e-9);
        assert_eq!(result.overall_trend, TrendDirection::Increasing);
        Ok(())
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
            .map(|c| (c.weekday, c.hour))
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
            baseline_days: 10,
            current_days: 10,
        };
        assert_eq!(increasing.trend(), TrendDirection::Increasing);

        let decreasing = HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 50.0,
            current_avg: 40.0,
            absolute_change: -10.0,
            percent_change: -20.0,
            baseline_days: 10,
            current_days: 10,
        };
        assert_eq!(decreasing.trend(), TrendDirection::Decreasing);

        let stable = HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 50.0,
            current_avg: 51.0,
            absolute_change: 1.0,
            percent_change: 2.0,
            baseline_days: 10,
            current_days: 10,
        };
        assert_eq!(stable.trend(), TrendDirection::Stable);
    }
}
