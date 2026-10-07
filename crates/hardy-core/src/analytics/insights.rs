//! Human-readable findings derived from the other analyses.

use super::{
    ComparisonMode, TrendDirection, analyze_days, calculate_stats, compare_periods,
    find_peak_hours, find_quiet_windows, weekday_short,
};
use crate::db::HourlyAverage;

/// A finding worth showing, e.g. "Saturday is the quietest day".
#[derive(Debug, Clone)]
pub struct Insight {
    pub category: InsightCategory,
    pub importance: u8,
    pub title: String,
    pub description: String,
    pub data: Option<(i32, i32, f64)>,
}

/// What an insight is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightCategory {
    Trend,
    Peak,
    QuietTime,
    Anomaly,
    DayPattern,
    Consistency,
}

/// Findings from the current period, compared with `baseline` when given;
/// most important first.
#[allow(clippy::too_many_lines)]
#[tracing::instrument(skip_all, fields(current.len = current.len(), has_baseline = baseline.is_some()))]
pub fn generate_insights(
    current: &[HourlyAverage],
    baseline: Option<&[HourlyAverage]>,
) -> Vec<Insight> {
    let mut insights = Vec::new();

    if let Some(stats) = calculate_stats(current) {
        let consistency_level = if stats.coefficient_of_variation < 0.3 {
            "very consistent"
        } else if stats.coefficient_of_variation < 0.5 {
            "moderately consistent"
        } else {
            "highly variable"
        };

        insights.push(Insight {
            category: InsightCategory::Consistency,
            importance: 2,
            title: format!("Occupancy is {consistency_level}"),
            description: format!(
                "Average occupancy is {:.1}% with a standard deviation of {:.1}%. Range: {:.1}% \
                 to {:.1}%.",
                stats.mean, stats.std_dev, stats.min, stats.max
            ),
            data: None,
        });
    }

    let day_analysis = analyze_days(current);
    if let Some(busiest_day) = day_analysis
        .iter()
        .max_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy))
        && busiest_day.sample_count >= 5
    {
        insights.push(Insight {
            category: InsightCategory::DayPattern,
            importance: 3,
            title: format!("{} is the busiest day", busiest_day.day_name),
            description: format!(
                "Average occupancy on {} is {:.1}%, peaking at {:.1}% around {}:00.",
                busiest_day.day_name,
                busiest_day.avg_occupancy,
                busiest_day.peak_occupancy,
                busiest_day.peak_hour.unwrap_or(0)
            ),
            data: Some((
                busiest_day.weekday,
                busiest_day.peak_hour.unwrap_or(0),
                busiest_day.avg_occupancy,
            )),
        });
    }

    if let Some(quietest_day) = day_analysis
        .iter()
        .filter(|d| d.sample_count >= 5)
        .min_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy))
    {
        insights.push(Insight {
            category: InsightCategory::QuietTime,
            importance: 4,
            title: format!("{} is the quietest day", quietest_day.day_name),
            description: format!(
                "Average occupancy on {} is only {:.1}%. Best time: around {}:00 ({:.1}%).",
                quietest_day.day_name,
                quietest_day.avg_occupancy,
                quietest_day.quietest_hour.unwrap_or(0),
                quietest_day.quietest_occupancy
            ),
            data: Some((
                quietest_day.weekday,
                quietest_day.quietest_hour.unwrap_or(0),
                quietest_day.quietest_occupancy,
            )),
        });
    }

    let peaks = find_peak_hours(current, 3);
    if !peaks.is_empty() {
        let peak_desc: Vec<String> = peaks
            .iter()
            .map(|(w, h, p)| format!("{} {}:00 ({:.0}%)", weekday_short(*w), h, p))
            .collect();

        insights.push(Insight {
            category: InsightCategory::Peak,
            importance: 3,
            title: "Busiest times to avoid".to_string(),
            description: format!("Peak hours: {}", peak_desc.join(", ")),
            data: Some(peaks[0]),
        });
    }

    let quiet_windows = find_quiet_windows(current, 40.0, 2);
    if !quiet_windows.is_empty() {
        let best_window = &quiet_windows[0];
        insights.push(Insight {
            category: InsightCategory::QuietTime,
            importance: 5,
            title: "Best workout window".to_string(),
            description: format!(
                "{} {}:00-{}:00 averages only {:.1}% occupancy. {} more quiet windows available.",
                weekday_short(best_window.weekday),
                best_window.start_hour,
                best_window.end_hour,
                best_window.avg_occupancy,
                quiet_windows.len().saturating_sub(1)
            ),
            data: Some((
                best_window.weekday,
                best_window.start_hour,
                best_window.avg_occupancy,
            )),
        });
    }

    if let Some(baseline_data) = baseline {
        let comparison = compare_periods(baseline_data, current, ComparisonMode::WeekOverWeek);

        let trend_desc = match comparison.overall_trend {
            TrendDirection::Increasing => {
                format!(
                    "Occupancy has increased by {:.1}% compared to the previous period. Consider \
                     adjusting your workout times.",
                    comparison.overall_change_percent.abs()
                )
            }
            TrendDirection::Decreasing => {
                format!(
                    "Good news! Occupancy has decreased by {:.1}% compared to the previous period.",
                    comparison.overall_change_percent.abs()
                )
            }
            TrendDirection::Stable => {
                "Occupancy patterns are stable compared to the previous period.".to_string()
            }
            TrendDirection::Insufficient => {
                "Not enough data to determine occupancy trends.".to_string()
            }
        };

        let importance = match comparison.overall_trend {
            TrendDirection::Increasing => 4,
            TrendDirection::Decreasing => 3,
            _ => 2,
        };

        insights.push(Insight {
            category: InsightCategory::Trend,
            importance,
            title: format!("Gym is {}", comparison.overall_trend.description()),
            description: trend_desc,
            data: None,
        });

        if !comparison.biggest_increases.is_empty() {
            let (w, h, change) = comparison.biggest_increases[0];
            insights.push(Insight {
                category: InsightCategory::Anomaly,
                importance: 3,
                title: "Significant occupancy increase".to_string(),
                description: format!(
                    "{} at {}:00 has seen a {:.0}% increase in occupancy. You may want to avoid \
                     this time slot.",
                    weekday_short(w),
                    h,
                    change
                ),
                data: Some((w, h, change)),
            });
        }
    }

    insights.sort_by_key(|a| std::cmp::Reverse(a.importance));
    insights
}

#[cfg(test)]
mod tests {

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
    fn test_generate_insights_empty_data() {
        let result = generate_insights(&[], None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_insights_basic() {
        let data: Vec<HourlyAverage> = (0..7)
            .flat_map(|weekday| {
                (8..20)
                    .map(move |hour| make_hourly_avg(weekday, hour, f64::from(20 + hour * 3), 10))
            })
            .collect();

        let result = generate_insights(&data, None);

        assert!(!result.is_empty());
        assert!(
            result
                .iter()
                .any(|i| i.category == InsightCategory::Consistency)
        );
        assert!(
            result
                .iter()
                .any(|i| i.category == InsightCategory::DayPattern)
        );
    }

    #[test]
    fn test_generate_insights_with_baseline() {
        let baseline: Vec<HourlyAverage> = (0..7)
            .flat_map(|weekday| (8..20).map(move |hour| make_hourly_avg(weekday, hour, 40.0, 10)))
            .collect();

        let current: Vec<HourlyAverage> = (0..7)
            .flat_map(|weekday| (8..20).map(move |hour| make_hourly_avg(weekday, hour, 60.0, 10)))
            .collect();

        let result = generate_insights(&current, Some(&baseline));

        assert!(result.iter().any(|i| i.category == InsightCategory::Trend));
    }

    #[test]
    fn test_insights_sorted_by_importance() {
        let data: Vec<HourlyAverage> = (0..7)
            .flat_map(|weekday| {
                (8..20)
                    .map(move |hour| make_hourly_avg(weekday, hour, f64::from(20 + hour * 3), 10))
            })
            .collect();

        let result = generate_insights(&data, None);

        for window in result.windows(2) {
            assert!(window[0].importance >= window[1].importance);
        }
    }
}
