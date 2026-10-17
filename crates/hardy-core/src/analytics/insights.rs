//! Human-readable findings derived from the other analyses.

use super::{
    ComparisonMode, MIN_DAYS, PeriodComparison, TrendDirection, analyze_days, calculate_stats,
    compare_periods, find_peak_hours, find_quiet_windows, weekday_short,
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
        .filter(|d| d.days >= MIN_DAYS)
        .max_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy))
    {
        insights.push(Insight {
            category: InsightCategory::DayPattern,
            importance: 3,
            title: format!("{} is the busiest day", busiest_day.day_name),
            description: format!(
                "Average occupancy on {} is {:.1}%, peaking at {:.1}% around {:02}:00.",
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
        .filter(|d| d.days >= MIN_DAYS)
        .min_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy))
    {
        insights.push(Insight {
            category: InsightCategory::QuietTime,
            importance: 4,
            title: format!("{} is the quietest day", quietest_day.day_name),
            description: format!(
                "Average occupancy on {} is only {:.1}%. Best time: around {:02}:00 ({:.1}%).",
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
            .map(|(w, h, p)| format!("{} {h:02}:00 ({p:.0}%)", weekday_short(*w)))
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
                "{} {:02}:00–{:02}:00 averages only {:.1}% occupancy. {} more quiet windows \
                 available.",
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

        let trend_desc = trend_description(&comparison);

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

        if let Some(c) = comparison.biggest_increases.first() {
            insights.push(Insight {
                category: InsightCategory::Anomaly,
                importance: 3,
                title: "Significant occupancy increase".to_string(),
                description: format!(
                    "{} {:02}:00: {:.0}% → {:.0}%, {:+.0} points. You may want to avoid this time \
                     slot.",
                    weekday_short(c.weekday),
                    c.hour,
                    c.baseline_avg,
                    c.current_avg,
                    c.absolute_change
                ),
                data: Some((c.weekday, c.hour, c.absolute_change)),
            });
        }
    }

    insights.sort_by_key(|a| std::cmp::Reverse(a.importance));
    insights
}

/// The trend in words, quoting the number it was decided on (the average
/// change per hour in points) and the relative change of the overall
/// average.
fn trend_description(comparison: &PeriodComparison) -> String {
    let Some(points) = comparison.mean_point_change else {
        return "Not enough data to determine occupancy trends.".to_string();
    };
    let change = format!(
        "{points:+.1} points per hour on average (overall {:+.0}%)",
        comparison.overall_change_percent
    );
    match comparison.overall_trend {
        TrendDirection::Increasing => format!(
            "Occupancy is up {change} compared to the previous period. Consider adjusting your \
             workout times."
        ),
        TrendDirection::Decreasing => {
            format!("Good news! Occupancy is down {change} compared to the previous period.")
        }
        TrendDirection::Stable | TrendDirection::Insufficient => {
            format!("Occupancy is stable compared to the previous period: {change}.")
        }
    }
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
            days: samples,
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
        }
    }

    /// Every weekday from 8:00 to 19:00 at `pct`.
    fn week_at(pct: f64) -> Vec<HourlyAverage> {
        (0..7)
            .flat_map(|w| (8..20).map(move |h| seen(w, h, pct, 4)))
            .collect()
    }

    fn find(insights: &[Insight], category: InsightCategory) -> anyhow::Result<&Insight> {
        insights
            .iter()
            .find(|i| i.category == category)
            .ok_or_else(|| anyhow::anyhow!("no {category:?} insight in {insights:?}"))
    }

    #[test]
    fn test_insights_trend_text_quotes_the_trend_number() -> anyhow::Result<()> {
        let insights = generate_insights(&week_at(32.0), Some(&week_at(30.0)));
        let trend = find(&insights, InsightCategory::Trend)?;
        assert_eq!(trend.title, "Gym is getting busier");
        assert!(
            trend.description.contains("+2.0 points"),
            "{}",
            trend.description
        );
        assert!(trend.description.contains("+7%"), "{}", trend.description);

        let insights = generate_insights(&week_at(30.5), Some(&week_at(30.0)));
        let trend = find(&insights, InsightCategory::Trend)?;
        assert_eq!(trend.title, "Gym is staying consistent");
        assert!(
            trend.description.contains("+0.5 points"),
            "{}",
            trend.description
        );
        Ok(())
    }

    #[test]
    fn test_busiest_day_is_picked_among_days_with_enough_data() -> anyhow::Result<()> {
        // Monday was seen once (and was packed), Tuesday and Wednesday
        // four times.
        let mut data: Vec<_> = (8..20).map(|h| seen(0, h, 90.0, 1)).collect();
        data.extend((8..20).map(|h| seen(1, h, 50.0, 4)));
        data.extend((8..20).map(|h| seen(2, h, 30.0, 4)));
        let insights = generate_insights(&data, None);
        let busiest = find(&insights, InsightCategory::DayPattern)?;
        assert_eq!(busiest.title, "Tuesday is the busiest day");
        Ok(())
    }

    #[test]
    fn test_insights_write_hours_with_two_digits() -> anyhow::Result<()> {
        let mut data = vec![seen(0, 9, 90.0, 4), seen(0, 10, 50.0, 4)];
        data.extend((17..21).map(|h| seen(5, h, 6.0, 4)));
        let insights = generate_insights(&data, None);
        let peak = find(&insights, InsightCategory::Peak)?;
        assert!(
            peak.description.contains("Mon 09:00 (90%)"),
            "{}",
            peak.description
        );
        let window = insights
            .iter()
            .find(|i| i.title == "Best workout window")
            .ok_or_else(|| anyhow::anyhow!("no window in {insights:?}"))?;
        assert!(
            window.description.starts_with("Sat 17:00–21:00"),
            "{}",
            window.description
        );
        Ok(())
    }

    #[test]
    fn test_increase_insight_is_in_points() -> anyhow::Result<()> {
        let baseline = vec![seen(2, 18, 30.0, 4)];
        let current = vec![seen(2, 18, 40.0, 4)];
        let insights = generate_insights(&current, Some(&baseline));
        let increase = find(&insights, InsightCategory::Anomaly)?;
        assert!(
            increase
                .description
                .starts_with("Wed 18:00: 30% → 40%, +10 points."),
            "{}",
            increase.description
        );
        Ok(())
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
