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

/// Overall direction from slot comparisons (needs enough slots).
pub fn determine_trend(comparisons: &[HourlyComparison]) -> TrendDirection {
    let valid_comparisons: Vec<_> = comparisons
        .iter()
        .filter(|c| c.baseline_samples >= 2 && c.current_samples >= 2)
        .collect();

    if valid_comparisons.len() < 5 {
        return TrendDirection::Insufficient;
    }

    #[allow(clippy::cast_precision_loss)]
    let valid_comparisons_f64 = valid_comparisons.len() as f64;

    let avg_change: f64 = valid_comparisons
        .iter()
        .map(|c| c.percent_change)
        .sum::<f64>()
        / valid_comparisons_f64;

    if avg_change > 3.0 {
        TrendDirection::Increasing
    } else if avg_change < -3.0 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_determine_trend_insufficient_data() {
        let comparisons = vec![HourlyComparison {
            weekday: 0,
            hour: 10,
            baseline_avg: 40.0,
            current_avg: 50.0,
            absolute_change: 10.0,
            percent_change: 25.0,
            baseline_samples: 1,
            current_samples: 1,
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
                baseline_samples: 10,
                current_samples: 10,
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
                baseline_samples: 10,
                current_samples: 10,
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
                baseline_samples: 10,
                current_samples: 10,
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
