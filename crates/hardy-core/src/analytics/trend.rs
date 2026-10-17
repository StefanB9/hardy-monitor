//! Overall direction of a period comparison.

use super::HourlyComparison;

/// Whether occupancy is rising, falling or steady.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Insufficient,
}

impl TrendDirection {
    /// Lower-case phrase, e.g. "getting busier".
    pub fn description(&self) -> &'static str {
        match self {
            TrendDirection::Increasing => "getting busier",
            TrendDirection::Decreasing => "getting quieter",
            TrendDirection::Stable => "staying consistent",
            TrendDirection::Insufficient => "insufficient data",
        }
    }
}

/// Hours seen in both periods needed before a trend is called.
const MIN_TREND_HOURS: usize = 5;

/// Average changes smaller than this (percentage points) count as stable.
const TREND_THRESHOLD_POINTS: f64 = 1.0;

/// The average change in percentage points over the hours seen on enough
/// days in both periods; `None` with fewer than five such hours.
pub fn mean_point_change(comparisons: &[HourlyComparison]) -> Option<f64> {
    let changes: Vec<f64> = comparisons
        .iter()
        .filter(|c| c.is_reliable())
        .map(|c| c.absolute_change)
        .collect();
    if changes.len() < MIN_TREND_HOURS {
        return None;
    }
    #[allow(clippy::cast_precision_loss)] // at most 168 hours
    let count = changes.len() as f64;
    Some(changes.iter().sum::<f64>() / count)
}

/// Direction of an average change in points (see [`mean_point_change`]).
pub fn trend_from_points(change: Option<f64>) -> TrendDirection {
    match change {
        None => TrendDirection::Insufficient,
        Some(c) if c > TREND_THRESHOLD_POINTS => TrendDirection::Increasing,
        Some(c) if c < -TREND_THRESHOLD_POINTS => TrendDirection::Decreasing,
        Some(_) => TrendDirection::Stable,
    }
}

/// Overall direction from slot comparisons: busier above +1 point on
/// average, quieter below -1 (needs enough hours).
pub fn determine_trend(comparisons: &[HourlyComparison]) -> TrendDirection {
    trend_from_points(mean_point_change(comparisons))
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

    use super::*;

    /// A slot going from `before` to `after`, seen on `days` days in each
    /// period.
    fn slot(hour: i32, before: f64, after: f64, days: i64) -> HourlyComparison {
        HourlyComparison {
            weekday: 0,
            hour,
            baseline_avg: before,
            current_avg: after,
            absolute_change: after - before,
            percent_change: (after - before) / before * 100.0,
            baseline_days: days,
            current_days: days,
        }
    }

    #[test]
    fn test_trend_follows_points_not_relative_changes() -> anyhow::Result<()> {
        // Ten near-empty hours +0.5 points (+25%), ten busy ones -0.5 points
        // (-1%): on average nothing changed.
        let mut comparisons: Vec<_> = (0..10).map(|h| slot(h, 2.0, 2.5, 4)).collect();
        comparisons.extend((10..20).map(|h| slot(h, 50.0, 49.5, 4)));
        let change = mean_point_change(&comparisons).ok_or_else(|| anyhow::anyhow!("no change"))?;
        assert_relative_eq!(change, 0.0, epsilon = 1e-9);
        assert_eq!(determine_trend(&comparisons), TrendDirection::Stable);
        Ok(())
    }

    #[test]
    fn test_trend_threshold_is_one_point() {
        let trend = |after: f64| {
            let comparisons: Vec<_> = (0..10).map(|h| slot(h, 30.0, after, 4)).collect();
            determine_trend(&comparisons)
        };
        assert_eq!(trend(31.5), TrendDirection::Increasing);
        assert_eq!(trend(28.5), TrendDirection::Decreasing);
        assert_eq!(trend(30.8), TrendDirection::Stable);
        assert_eq!(trend(29.2), TrendDirection::Stable);
    }

    #[test]
    fn test_trend_ignores_hours_seen_on_one_day() {
        let comparisons: Vec<_> = (0..10).map(|h| slot(h, 30.0, 40.0, 1)).collect();
        assert_eq!(mean_point_change(&comparisons), None);
        assert_eq!(determine_trend(&comparisons), TrendDirection::Insufficient);
    }

    #[test]
    fn test_determine_trend_insufficient_data() {
        let comparisons = vec![HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 40.0,
            current_avg: 50.0,
            absolute_change: 10.0,
            percent_change: 25.0,
            baseline_days: 1,
            current_days: 1,
        }];

        let result = determine_trend(&comparisons);
        assert_eq!(result, TrendDirection::Insufficient);
    }

    #[test]
    fn test_determine_trend_increasing() {
        let comparisons: Vec<HourlyComparison> = (0..10)
            .map(|i| HourlyComparison {
                weekday: 0,
                hour: i,
                baseline_avg: 40.0,
                current_avg: 50.0,
                absolute_change: 10.0,
                percent_change: 25.0,
                baseline_days: 10,
                current_days: 10,
            })
            .collect();

        let result = determine_trend(&comparisons);
        assert_eq!(result, TrendDirection::Increasing);
    }

    #[test]
    fn test_determine_trend_decreasing() {
        let comparisons: Vec<HourlyComparison> = (0..10)
            .map(|i| HourlyComparison {
                weekday: 0,
                hour: i,
                baseline_avg: 50.0,
                current_avg: 40.0,
                absolute_change: -10.0,
                percent_change: -20.0,
                baseline_days: 10,
                current_days: 10,
            })
            .collect();

        let result = determine_trend(&comparisons);
        assert_eq!(result, TrendDirection::Decreasing);
    }

    #[test]
    fn test_determine_trend_stable() {
        let comparisons: Vec<HourlyComparison> = (0..10)
            .map(|i| HourlyComparison {
                weekday: 0,
                hour: i,
                baseline_avg: 50.0,
                current_avg: 51.0,
                absolute_change: 1.0,
                percent_change: 2.0,
                baseline_days: 10,
                current_days: 10,
            })
            .collect();

        let result = determine_trend(&comparisons);
        assert_eq!(result, TrendDirection::Stable);
    }

    #[test]
    fn test_trend_direction_description() {
        assert_eq!(TrendDirection::Increasing.description(), "getting busier");
        assert_eq!(TrendDirection::Decreasing.description(), "getting quieter");
        assert_eq!(TrendDirection::Stable.description(), "staying consistent");
        assert_eq!(
            TrendDirection::Insufficient.description(),
            "insufficient data"
        );
    }
}
