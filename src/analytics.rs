use std::collections::{BTreeSet, HashMap};

use chrono::{
    DateTime, Datelike, Duration as ChronoDuration, Local, NaiveDate, NaiveTime, Offset, TimeZone,
    Timelike, Utc,
};

use crate::{db::HourlyAverage, schedule::GymSchedule, traits::Clock};

const DAY_NAMES_LONG: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

const DAY_NAMES_SHORT: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    WeekOverWeek,
    MonthOverMonth,
    CustomRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Insufficient,
}

impl TrendDirection {
    pub fn description(&self) -> &'static str {
        match self {
            TrendDirection::Increasing => "getting busier",
            TrendDirection::Decreasing => "getting quieter",
            TrendDirection::Stable => "staying consistent",
            TrendDirection::Insufficient => "insufficient data",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            TrendDirection::Increasing => "📈",
            TrendDirection::Decreasing => "📉",
            TrendDirection::Stable => "➡️",
            TrendDirection::Insufficient => "❓",
        }
    }
}

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

#[derive(Debug, Clone)]
pub struct TimePeriod {
    pub weekday: i32,
    pub start_hour: i32,
    pub end_hour: i32,
    pub avg_occupancy: f64,
}

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

#[derive(Debug, Clone)]
pub struct Insight {
    pub category: InsightCategory,
    pub importance: u8,
    pub title: String,
    pub description: String,
    pub data: Option<(i32, i32, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightCategory {
    Trend,
    Peak,
    QuietTime,
    Anomaly,
    DayPattern,
    Consistency,
}

pub fn midnight_utc(date: NaiveDate) -> DateTime<Utc> {
    date.and_time(NaiveTime::MIN).and_utc()
}

pub fn midnight_local_as_utc(date: NaiveDate) -> DateTime<Utc> {
    Local
        .from_local_datetime(&date.and_time(NaiveTime::MIN))
        .earliest()
        .map_or_else(
            || date.and_time(NaiveTime::MIN).and_utc(),
            |dt| dt.with_timezone(&Utc),
        )
}

pub fn calculate_predictions(baseline: &[HourlyAverage]) -> Vec<(DateTime<Utc>, f64)> {
    calculate_predictions_with_schedule(baseline, &GymSchedule::default())
}

pub fn calculate_predictions_with_schedule(
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
) -> Vec<(DateTime<Utc>, f64)> {
    calculate_predictions_with_clock(baseline, schedule, &crate::traits::SystemClock)
}

#[tracing::instrument(skip_all, fields(baseline.len = baseline.len()))]
pub fn calculate_predictions_with_clock<C: Clock>(
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    clock: &C,
) -> Vec<(DateTime<Utc>, f64)> {
    let mut predictions = Vec::new();
    if baseline.is_empty() {
        return predictions;
    }

    let now = clock.now_utc();

    for i in 1..=2 {
        let target_time = now + ChronoDuration::hours(i);
        let target_hour = target_time.hour() as i32;
        let target_weekday = target_time.weekday().num_days_from_monday() as i32;

        let local_target = target_time.with_timezone(&Local);
        if !schedule.is_open(&local_target) {
            continue;
        }

        if let Some(avg) = baseline
            .iter()
            .find(|x| x.weekday == target_weekday && x.hour == target_hour)
        {
            let plot_time = target_time
                .with_minute(0)
                .and_then(|t| t.with_second(0))
                .and_then(|t| t.with_nanosecond(0))
                .unwrap_or(target_time);

            predictions.push((plot_time, avg.avg_percentage));
        }
    }
    predictions
}

pub fn find_best_time_today(data: &[HourlyAverage]) -> Option<(i32, f64)> {
    find_best_time_today_with_clock(data, &crate::traits::SystemClock)
}

pub fn find_best_time_today_with_clock<C: Clock>(
    data: &[HourlyAverage],
    clock: &C,
) -> Option<(i32, f64)> {
    let now = clock.now_local();
    let today_idx = now.weekday().num_days_from_monday() as i32;

    let offset_seconds = now.offset().fix().local_minus_utc();
    let seconds_per_week = 7 * 24 * 3600;

    data.iter()
        .map(|d| {
            let utc_seconds = (i64::from(d.weekday) * 24 + i64::from(d.hour)) * 3600;
            let local_seconds = utc_seconds + i64::from(offset_seconds);

            let wrapped_local =
                ((local_seconds % seconds_per_week) + seconds_per_week) % seconds_per_week;

            let local_w = (wrapped_local / 3600) / 24;
            let local_h = (wrapped_local / 3600) % 24;

            (local_w as i32, local_h as i32, d.avg_percentage)
        })
        .filter(|(w, _, _)| *w == today_idx)
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, h, avg)| (h, avg))
}

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
            .map(|h| h.avg_percentage * h.sample_count as f64)
            .sum();
        let count: i64 = baseline.iter().map(|h| h.sample_count).sum();
        if count > 0 { total / count as f64 } else { 0.0 }
    };

    let current_overall_avg = if current.is_empty() {
        0.0
    } else {
        let total: f64 = current
            .iter()
            .map(|h| h.avg_percentage * h.sample_count as f64)
            .sum();
        let count: i64 = current.iter().map(|h| h.sample_count).sum();
        if count > 0 { total / count as f64 } else { 0.0 }
    };

    let overall_change_percent = if baseline_overall_avg > 0.0 {
        ((current_overall_avg - baseline_overall_avg) / baseline_overall_avg) * 100.0
    } else {
        0.0
    };

    let overall_trend = determine_trend(&hourly_comparisons);

    let mut sorted_by_increase: Vec<_> = hourly_comparisons
        .iter()
        .filter(|c| c.baseline_samples >= 2 && c.current_samples >= 2)
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

pub fn determine_trend(comparisons: &[HourlyComparison]) -> TrendDirection {
    let valid_comparisons: Vec<_> = comparisons
        .iter()
        .filter(|c| c.baseline_samples >= 2 && c.current_samples >= 2)
        .collect();

    if valid_comparisons.len() < 5 {
        return TrendDirection::Insufficient;
    }

    let avg_change: f64 = valid_comparisons
        .iter()
        .map(|c| c.percent_change)
        .sum::<f64>()
        / valid_comparisons.len() as f64;

    if avg_change > 3.0 {
        TrendDirection::Increasing
    } else if avg_change < -3.0 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

#[tracing::instrument(skip_all, fields(n = data.len()))]
pub fn calculate_stats(data: &[HourlyAverage]) -> Option<OccupancyStats> {
    if data.is_empty() {
        return None;
    }

    let n = data.len();

    let mean = data.iter().map(|h| h.avg_percentage).sum::<f64>() / n as f64;

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
        / n as f64;
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

#[tracing::instrument(skip_all)]
pub fn analyze_days(data: &[HourlyAverage]) -> Vec<DayAnalysis> {
    (0..7)
        .map(|weekday| {
            let day_data: Vec<_> = data.iter().filter(|h| h.weekday == weekday).collect();

            let total_samples: i64 = day_data.iter().map(|h| h.sample_count).sum();
            let weighted_sum: f64 = day_data
                .iter()
                .map(|h| h.avg_percentage * h.sample_count as f64)
                .sum();
            let avg_occupancy = if total_samples > 0 {
                weighted_sum / total_samples as f64
            } else {
                0.0
            };

            let peak = day_data
                .iter()
                .max_by(|a, b| a.avg_percentage.total_cmp(&b.avg_percentage));

            let quietest = day_data
                .iter()
                .min_by(|a, b| a.avg_percentage.total_cmp(&b.avg_percentage));

            DayAnalysis {
                weekday,
                day_name: DAY_NAMES_LONG[weekday as usize],
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

pub fn find_peak_hours(data: &[HourlyAverage], top_n: usize) -> Vec<(i32, i32, f64)> {
    let mut sorted: Vec<_> = data
        .iter()
        .filter(|h| h.sample_count >= 2)
        .map(|h| (h.weekday, h.hour, h.avg_percentage))
        .collect();

    sorted.sort_by(|a, b| b.2.total_cmp(&a.2));
    sorted.truncate(top_n);
    sorted
}

pub fn find_quiet_hours(data: &[HourlyAverage], top_n: usize) -> Vec<(i32, i32, f64)> {
    let mut sorted: Vec<_> = data
        .iter()
        .filter(|h| h.sample_count >= 2 && h.avg_percentage > 0.0)
        .map(|h| (h.weekday, h.hour, h.avg_percentage))
        .collect();

    sorted.sort_by(|a, b| a.2.total_cmp(&b.2));
    sorted.truncate(top_n);
    sorted
}

#[tracing::instrument(skip_all, fields(threshold, min_hours))]
pub fn find_quiet_windows(
    data: &[HourlyAverage],
    threshold: f64,
    min_hours: usize,
) -> Vec<TimePeriod> {
    let mut windows = Vec::new();

    for weekday in 0i32..7 {
        let mut day_hours: Vec<_> = data
            .iter()
            .filter(|h| h.weekday == weekday && h.sample_count >= 2)
            .collect();
        day_hours.sort_by_key(|h| h.hour);

        let mut window_start: Option<i32> = None;
        let mut window_sum = 0.0;
        let mut window_count = 0;

        for h in &day_hours {
            if h.avg_percentage <= threshold {
                if window_start.is_none() {
                    window_start = Some(h.hour);
                    window_sum = 0.0;
                    window_count = 0;
                }
                window_sum += h.avg_percentage;
                window_count += 1;
            } else {
                if let Some(start) = window_start
                    && window_count >= min_hours
                {
                    windows.push(TimePeriod {
                        weekday,
                        start_hour: start,
                        end_hour: h.hour,
                        avg_occupancy: window_sum / window_count as f64,
                    });
                }
                window_start = None;
            }
        }

        if let Some(start) = window_start
            && window_count >= min_hours
        {
            windows.push(TimePeriod {
                weekday,
                start_hour: start,
                end_hour: 24,
                avg_occupancy: window_sum / window_count as f64,
            });
        }
    }

    windows.sort_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy));
    windows
}

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

pub fn weekday_name(weekday: i32) -> &'static str {
    usize::try_from(weekday)
        .ok()
        .and_then(|i| DAY_NAMES_LONG.get(i))
        .copied()
        .unwrap_or("Unknown")
}

pub fn weekday_short(weekday: i32) -> &'static str {
    usize::try_from(weekday)
        .ok()
        .and_then(|i| DAY_NAMES_SHORT.get(i))
        .copied()
        .unwrap_or("???")
}

#[cfg(test)]
mod tests {
    use chrono::{Datelike, NaiveDate, Timelike};

    use super::*;

    #[test]
    fn test_midnight_utc_basic() {
        let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let result = midnight_utc(date);

        assert_eq!(result.year(), 2024);
        assert_eq!(result.month(), 6);
        assert_eq!(result.day(), 15);
        assert_eq!(result.hour(), 0);
        assert_eq!(result.minute(), 0);
        assert_eq!(result.second(), 0);
    }

    #[test]
    fn test_midnight_utc_leap_year() {
        let date = NaiveDate::from_ymd_opt(2024, 2, 29).unwrap();
        let result = midnight_utc(date);

        assert_eq!(result.month(), 2);
        assert_eq!(result.day(), 29);
    }

    #[test]
    fn test_midnight_utc_year_boundary() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
        let result = midnight_utc(date);

        assert_eq!(result.year(), 2024);
        assert_eq!(result.month(), 12);
        assert_eq!(result.day(), 31);
    }

    #[test]
    fn test_calculate_predictions_empty_baseline() {
        let baseline: Vec<HourlyAverage> = vec![];
        let result = calculate_predictions(&baseline);
        assert!(result.is_empty());
    }

    #[test]
    fn test_calculate_predictions_with_schedule_empty_baseline() {
        let baseline: Vec<HourlyAverage> = vec![];
        let schedule = GymSchedule::default();
        let result = calculate_predictions_with_schedule(&baseline, &schedule);
        assert!(result.is_empty());
    }

    #[test]
    fn test_calculate_predictions_returns_at_most_two() {
        let mut baseline = Vec::new();
        for weekday in 0..7 {
            for hour in 0..24 {
                baseline.push(HourlyAverage {
                    weekday,
                    hour,
                    avg_percentage: 50.0,
                    sample_count: 10,
                });
            }
        }

        let result = calculate_predictions(&baseline);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_calculate_predictions_respects_schedule() {
        let schedule = GymSchedule::new_for_test(0, 0, 0, 0);

        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 30.0,
            sample_count: 5,
        }];

        let result = calculate_predictions_with_schedule(&baseline, &schedule);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_best_time_empty_data() {
        let data: Vec<HourlyAverage> = vec![];
        let result = find_best_time_today(&data);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_time_returns_lowest_percentage() {
        let today_idx = Local::now().weekday().num_days_from_monday() as i32;

        let data = vec![
            HourlyAverage {
                weekday: today_idx,
                hour: 10,
                avg_percentage: 50.0,
                sample_count: 5,
            },
            HourlyAverage {
                weekday: today_idx,
                hour: 14,
                avg_percentage: 20.0,
                sample_count: 5,
            },
            HourlyAverage {
                weekday: today_idx,
                hour: 18,
                avg_percentage: 80.0,
                sample_count: 5,
            },
        ];

        let result = find_best_time_today(&data);
        assert!(result.is_some());
        let (_hour, avg) = result.unwrap();
        assert_eq!(avg, 20.0);
    }

    #[test]
    fn test_find_best_time_filters_by_today() {
        let today_idx = Local::now().weekday().num_days_from_monday() as i32;
        let other_day = (today_idx + 1) % 7;

        let data = vec![
            HourlyAverage {
                weekday: other_day,
                hour: 10,
                avg_percentage: 10.0,
                sample_count: 5,
            },
            HourlyAverage {
                weekday: today_idx,
                hour: 14,
                avg_percentage: 30.0,
                sample_count: 5,
            },
        ];

        let result = find_best_time_today(&data);
        assert!(result.is_some());
    }

    #[test]
    fn test_predictions_with_open_schedule() {
        let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

        let mut baseline = Vec::new();
        for weekday in 0..7 {
            for hour in 0..24 {
                baseline.push(HourlyAverage {
                    weekday,
                    hour,
                    avg_percentage: (hour as f64) * 2.0,
                    sample_count: 10,
                });
            }
        }

        let result = calculate_predictions_with_schedule(&baseline, &schedule);
        assert!(result.len() <= 2);
    }

    mod clock_tests {
        use chrono::TimeZone;

        use super::*;
        use crate::traits::MockClock;

        #[test]
        fn test_predictions_with_mock_clock() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 11,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 12,
                    avg_percentage: 50.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 30.0);
            assert_eq!(predictions[1].1, 50.0);
        }

        #[test]
        fn test_predictions_clock_advances_correctly() {
            let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 11,
                    avg_percentage: 25.0,
                    sample_count: 5,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 12,
                    avg_percentage: 45.0,
                    sample_count: 5,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 13,
                    avg_percentage: 65.0,
                    sample_count: 5,
                },
            ];

            let predictions1 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
            assert_eq!(predictions1.len(), 2);
            assert_eq!(predictions1[0].1, 25.0);
            assert_eq!(predictions1[1].1, 45.0);

            clock.advance(ChronoDuration::hours(1));

            let predictions2 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
            assert_eq!(predictions2.len(), 2);
            assert_eq!(predictions2[0].1, 45.0);
            assert_eq!(predictions2[1].1, 65.0);
        }

        #[test]
        fn test_find_best_time_with_mock_clock() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);

            let data = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 8,
                    avg_percentage: 60.0,
                    sample_count: 5,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 14,
                    avg_percentage: 15.0,
                    sample_count: 5,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 18,
                    avg_percentage: 80.0,
                    sample_count: 5,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            assert!(result.is_some());
            let (_, avg) = result.unwrap();
            assert_eq!(avg, 15.0);
        }
    }

    mod week_boundary_tests {
        use chrono::TimeZone;

        use super::*;
        use crate::traits::MockClock;

        #[test]
        fn test_predictions_crossing_sunday_to_monday() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 23, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 0,
                    avg_percentage: 25.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 1,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 25.0);
            assert_eq!(predictions[1].1, 30.0);
        }

        #[test]
        fn test_predictions_crossing_saturday_to_sunday() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 15, 22, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![
                HourlyAverage {
                    weekday: 5,
                    hour: 23,
                    avg_percentage: 40.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 6,
                    hour: 0,
                    avg_percentage: 15.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 40.0);
            assert_eq!(predictions[1].1, 15.0);
        }

        #[test]
        fn test_predictions_at_year_boundary() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 12, 31, 23, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![
                HourlyAverage {
                    weekday: 2,
                    hour: 0,
                    avg_percentage: 10.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 2,
                    hour: 1,
                    avg_percentage: 20.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 10.0);
            assert_eq!(predictions[1].1, 20.0);
        }

        #[test]
        fn test_find_best_time_near_midnight_start_of_week() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 0, 30, 0).unwrap();
            let clock = MockClock::new(fixed_time);

            let data = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 0,
                    avg_percentage: 5.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0,
                    hour: 12,
                    avg_percentage: 70.0,
                    sample_count: 10,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            assert!(result.is_some());
            let (_, avg) = result.unwrap();
            assert_eq!(avg, 5.0);
        }

        #[test]
        fn test_find_best_time_near_midnight_end_of_week() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 23, 30, 0).unwrap();
            let clock = MockClock::new(fixed_time);

            let data = vec![
                HourlyAverage {
                    weekday: 6,
                    hour: 10,
                    avg_percentage: 35.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 6,
                    hour: 23,
                    avg_percentage: 8.0,
                    sample_count: 10,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            assert!(result.is_some());
            let (_, avg) = result.unwrap();
            assert_eq!(avg, 8.0);
        }

        #[test]
        fn test_predictions_week_wrapping_with_missing_data() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 22, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline = vec![HourlyAverage {
                weekday: 6,
                hour: 23,
                avg_percentage: 45.0,
                sample_count: 10,
            }];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 1);
            assert_eq!(predictions[0].1, 45.0);
        }

        #[test]
        fn test_find_best_time_no_data_for_current_day() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 19, 10, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);

            let data = vec![
                HourlyAverage {
                    weekday: 0,
                    hour: 10,
                    avg_percentage: 20.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 1,
                    hour: 10,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            assert!(result.is_none());
        }

        #[test]
        fn test_predictions_all_week_data_available() {
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 21, 11, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let mut baseline = Vec::new();
            for weekday in 0..7 {
                for hour in 0..24 {
                    baseline.push(HourlyAverage {
                        weekday,
                        hour,
                        avg_percentage: (weekday * 10 + hour) as f64,
                        sample_count: 10,
                    });
                }
            }

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 52.0);
            assert_eq!(predictions[1].1, 53.0);
        }

        #[test]
        fn test_monday_to_sunday_full_cycle() {
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            let baseline: Vec<HourlyAverage> = (0..7)
                .map(|weekday| HourlyAverage {
                    weekday,
                    hour: 10,
                    avg_percentage: (weekday as f64) * 10.0 + 5.0,
                    sample_count: 10,
                })
                .collect();

            for day in 0..7 {
                let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17 + day, 9, 0, 0).unwrap();
                let clock = MockClock::new(fixed_time);

                let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

                if !predictions.is_empty() {
                    let expected_weekday = day % 7;
                    let expected_pct = (expected_weekday as f64) * 10.0 + 5.0;
                    assert_eq!(
                        predictions[0].1, expected_pct,
                        "Day {} should have percentage {}",
                        day, expected_pct
                    );
                }
            }
        }
    }

    mod proptest_tests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn midnight_utc_always_at_midnight(
                year in 2000i32..2100,
                month in 1u32..=12,
                day in 1u32..=28
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let result = midnight_utc(date);
                    prop_assert_eq!(result.hour(), 0);
                    prop_assert_eq!(result.minute(), 0);
                    prop_assert_eq!(result.second(), 0);
                    prop_assert_eq!(result.nanosecond(), 0);
                }
            }

            #[test]
            fn midnight_utc_preserves_date(
                year in 2000i32..2100,
                month in 1u32..=12,
                day in 1u32..=28
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                    let result = midnight_utc(date);
                    prop_assert_eq!(result.year(), year);
                    prop_assert_eq!(result.month(), month);
                    prop_assert_eq!(result.day(), day);
                }
            }

            #[test]
            fn predictions_never_exceed_two(
                baseline_size in 0usize..200
            ) {
                let mut baseline = Vec::new();
                for i in 0..baseline_size {
                    baseline.push(HourlyAverage {
                        weekday: (i % 7) as i32,
                        hour: (i % 24) as i32,
                        avg_percentage: (i as f64) * 1.5,
                        sample_count: 1,
                    });
                }
                let result = calculate_predictions(&baseline);
                prop_assert!(result.len() <= 2,
                    "Predictions should never exceed 2, got {}", result.len());
            }

            #[test]
            fn find_best_time_returns_lowest_if_found(
                percentages in prop::collection::vec(0.0f64..=100.0, 1..50)
            ) {
                let today_idx = Local::now().weekday().num_days_from_monday() as i32;
                let data: Vec<HourlyAverage> = percentages
                    .iter()
                    .enumerate()
                    .map(|(i, &pct)| HourlyAverage {
                        weekday: today_idx,
                        hour: (i % 24) as i32,
                        avg_percentage: pct,
                        sample_count: 1,
                    })
                    .collect();

                if let Some((_, avg)) = find_best_time_today(&data) {
                    prop_assert!(percentages.iter().any(|&p| (p - avg).abs() < 0.001),
                        "Returned avg {} not found in input", avg);
                }
            }
        }
    }

    mod comparative_tests {
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
            assert_eq!(result[0].baseline_avg, 40.0);
            assert_eq!(result[0].current_avg, 50.0);
            assert_eq!(result[0].absolute_change, 10.0);
            assert!((result[0].percent_change - 25.0).abs() < 0.01);
        }

        #[test]
        fn test_build_hourly_comparisons_missing_baseline() {
            let baseline = vec![];
            let current = vec![make_hourly_avg(0, 10, 50.0, 5)];

            let result = build_hourly_comparisons(&baseline, &current);

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].baseline_avg, 0.0);
            assert_eq!(result[0].current_avg, 50.0);
            assert_eq!(result[0].percent_change, 100.0);
        }

        #[test]
        fn test_build_hourly_comparisons_missing_current() {
            let baseline = vec![make_hourly_avg(0, 10, 50.0, 5)];
            let current = vec![];

            let result = build_hourly_comparisons(&baseline, &current);

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].baseline_avg, 50.0);
            assert_eq!(result[0].current_avg, 0.0);
            assert_eq!(result[0].percent_change, -100.0);
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

    mod stats_tests {
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
        fn test_calculate_stats_single_value() {
            let data = vec![make_hourly_avg(0, 10, 50.0, 5)];
            let result = calculate_stats(&data).unwrap();

            assert_eq!(result.mean, 50.0);
            assert_eq!(result.median, 50.0);
            assert_eq!(result.std_dev, 0.0);
            assert_eq!(result.min, 50.0);
            assert_eq!(result.max, 50.0);
            assert_eq!(result.sample_count, 1);
        }

        #[test]
        fn test_calculate_stats_multiple_values() {
            let data = vec![
                make_hourly_avg(0, 10, 20.0, 5),
                make_hourly_avg(0, 11, 40.0, 5),
                make_hourly_avg(0, 12, 60.0, 5),
                make_hourly_avg(0, 13, 80.0, 5),
            ];
            let result = calculate_stats(&data).unwrap();

            assert_eq!(result.mean, 50.0);
            assert_eq!(result.median, 50.0);
            assert_eq!(result.min, 20.0);
            assert_eq!(result.max, 80.0);
            assert_eq!(result.sample_count, 4);
            assert!(result.std_dev > 0.0);
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
            assert_eq!(result[0].peak_occupancy, 50.0);
            assert_eq!(result[0].quietest_hour, Some(10));
            assert_eq!(result[0].quietest_occupancy, 30.0);
        }

        #[test]
        fn test_find_peak_hours() {
            let data = vec![
                make_hourly_avg(0, 10, 30.0, 5),
                make_hourly_avg(0, 11, 80.0, 5),
                make_hourly_avg(1, 10, 70.0, 5),
                make_hourly_avg(2, 15, 90.0, 5),
            ];

            let result = find_peak_hours(&data, 2);

            assert_eq!(result.len(), 2);
            assert_eq!(result[0], (2, 15, 90.0));
            assert_eq!(result[1], (0, 11, 80.0));
        }

        #[test]
        fn test_find_quiet_hours() {
            let data = vec![
                make_hourly_avg(0, 10, 10.0, 5),
                make_hourly_avg(0, 11, 80.0, 5),
                make_hourly_avg(1, 10, 20.0, 5),
                make_hourly_avg(2, 15, 90.0, 5),
            ];

            let result = find_quiet_hours(&data, 2);

            assert_eq!(result.len(), 2);
            assert_eq!(result[0], (0, 10, 10.0));
            assert_eq!(result[1], (1, 10, 20.0));
        }

        #[test]
        fn test_find_quiet_windows() {
            let data = vec![
                make_hourly_avg(0, 6, 20.0, 5),
                make_hourly_avg(0, 7, 25.0, 5),
                make_hourly_avg(0, 8, 30.0, 5),
                make_hourly_avg(0, 9, 70.0, 5),
                make_hourly_avg(0, 10, 80.0, 5),
            ];

            let result = find_quiet_windows(&data, 40.0, 2);

            assert!(!result.is_empty());
            let window = &result[0];
            assert_eq!(window.weekday, 0);
            assert_eq!(window.start_hour, 6);
            assert!(window.end_hour >= 8);
        }
    }

    mod insight_tests {
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
                        .map(move |hour| make_hourly_avg(weekday, hour, (20 + hour * 3) as f64, 10))
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
                .flat_map(|weekday| {
                    (8..20).map(move |hour| make_hourly_avg(weekday, hour, 40.0, 10))
                })
                .collect();

            let current: Vec<HourlyAverage> = (0..7)
                .flat_map(|weekday| {
                    (8..20).map(move |hour| make_hourly_avg(weekday, hour, 60.0, 10))
                })
                .collect();

            let result = generate_insights(&current, Some(&baseline));

            assert!(result.iter().any(|i| i.category == InsightCategory::Trend));
        }

        #[test]
        fn test_insights_sorted_by_importance() {
            let data: Vec<HourlyAverage> = (0..7)
                .flat_map(|weekday| {
                    (8..20)
                        .map(move |hour| make_hourly_avg(weekday, hour, (20 + hour * 3) as f64, 10))
                })
                .collect();

            let result = generate_insights(&data, None);

            for window in result.windows(2) {
                assert!(window[0].importance >= window[1].importance);
            }
        }
    }

    mod utility_tests {
        use super::*;

        #[test]
        fn test_weekday_name() {
            assert_eq!(weekday_name(0), "Monday");
            assert_eq!(weekday_name(1), "Tuesday");
            assert_eq!(weekday_name(2), "Wednesday");
            assert_eq!(weekday_name(3), "Thursday");
            assert_eq!(weekday_name(4), "Friday");
            assert_eq!(weekday_name(5), "Saturday");
            assert_eq!(weekday_name(6), "Sunday");
            assert_eq!(weekday_name(7), "Unknown");
        }

        #[test]
        fn test_weekday_short() {
            assert_eq!(weekday_short(0), "Mon");
            assert_eq!(weekday_short(1), "Tue");
            assert_eq!(weekday_short(2), "Wed");
            assert_eq!(weekday_short(3), "Thu");
            assert_eq!(weekday_short(4), "Fri");
            assert_eq!(weekday_short(5), "Sat");
            assert_eq!(weekday_short(6), "Sun");
            assert_eq!(weekday_short(7), "???");
        }
    }
}
