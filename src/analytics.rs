use std::collections::HashMap;

use chrono::{
    DateTime, Datelike, Duration as ChronoDuration, Local, NaiveDate, Offset, Timelike, Utc,
};

use crate::{db::HourlyAverage, schedule::GymSchedule, traits::Clock};

// ==================== Comparison Types ====================

/// Mode for comparing time periods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    /// Compare current week to previous week
    WeekOverWeek,
    /// Compare current week to same week last month (4 weeks ago)
    MonthOverMonth,
    /// Compare two custom date ranges
    CustomRange,
}

/// Direction of a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Occupancy is increasing
    Increasing,
    /// Occupancy is decreasing
    Decreasing,
    /// Occupancy is relatively stable (within threshold)
    Stable,
    /// Not enough data to determine trend
    Insufficient,
}

impl TrendDirection {
    /// Returns a human-readable description of the trend.
    pub fn description(&self) -> &'static str {
        match self {
            TrendDirection::Increasing => "getting busier",
            TrendDirection::Decreasing => "getting quieter",
            TrendDirection::Stable => "staying consistent",
            TrendDirection::Insufficient => "insufficient data",
        }
    }

    /// Returns an emoji representation of the trend.
    pub fn emoji(&self) -> &'static str {
        match self {
            TrendDirection::Increasing => "📈",
            TrendDirection::Decreasing => "📉",
            TrendDirection::Stable => "➡️",
            TrendDirection::Insufficient => "❓",
        }
    }
}

/// Comparison of occupancy for a specific hour between two periods.
#[derive(Debug, Clone)]
pub struct HourlyComparison {
    /// Day of week (0=Monday, 6=Sunday)
    pub weekday: i32,
    /// Hour of day (0-23)
    pub hour: i32,
    /// Average percentage in the baseline/previous period
    pub baseline_avg: f64,
    /// Average percentage in the current/comparison period
    pub current_avg: f64,
    /// Absolute change (current - baseline)
    pub absolute_change: f64,
    /// Percentage change relative to baseline
    pub percent_change: f64,
    /// Sample count in baseline period
    pub baseline_samples: i64,
    /// Sample count in current period
    pub current_samples: i64,
}

impl HourlyComparison {
    /// Returns the trend direction for this hour.
    pub fn trend(&self) -> TrendDirection {
        if self.baseline_samples < 2 || self.current_samples < 2 {
            return TrendDirection::Insufficient;
        }
        // Use 5% as threshold for "stable"
        if self.percent_change > 5.0 {
            TrendDirection::Increasing
        } else if self.percent_change < -5.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Comparison between two time periods.
#[derive(Debug, Clone)]
pub struct PeriodComparison {
    /// Mode used for this comparison
    pub mode: ComparisonMode,
    /// Overall average in baseline period
    pub baseline_overall_avg: f64,
    /// Overall average in current period
    pub current_overall_avg: f64,
    /// Overall change percentage
    pub overall_change_percent: f64,
    /// Overall trend direction
    pub overall_trend: TrendDirection,
    /// Hour-by-hour comparisons
    pub hourly_comparisons: Vec<HourlyComparison>,
    /// Hours with biggest increases
    pub biggest_increases: Vec<(i32, i32, f64)>, // (weekday, hour, change%)
    /// Hours with biggest decreases
    pub biggest_decreases: Vec<(i32, i32, f64)>, // (weekday, hour, change%)
}

// ==================== Statistical Analysis ====================

/// Statistical summary of occupancy data.
#[derive(Debug, Clone)]
pub struct OccupancyStats {
    /// Arithmetic mean of occupancy
    pub mean: f64,
    /// Median occupancy
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum occupancy
    pub min: f64,
    /// Maximum occupancy
    pub max: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Coefficient of variation (std_dev / mean) - measures consistency
    pub coefficient_of_variation: f64,
}

/// Represents a peak or quiet period.
#[derive(Debug, Clone)]
pub struct TimePeriod {
    /// Day of week (0=Monday, 6=Sunday)
    pub weekday: i32,
    /// Starting hour
    pub start_hour: i32,
    /// Ending hour (exclusive)
    pub end_hour: i32,
    /// Average occupancy during this period
    pub avg_occupancy: f64,
}

/// Day-of-week analysis result.
#[derive(Debug, Clone)]
pub struct DayAnalysis {
    /// Day of week (0=Monday, 6=Sunday)
    pub weekday: i32,
    /// Day name
    pub day_name: &'static str,
    /// Average occupancy for this day
    pub avg_occupancy: f64,
    /// Peak hour for this day
    pub peak_hour: Option<i32>,
    /// Peak occupancy
    pub peak_occupancy: f64,
    /// Quietest hour for this day
    pub quietest_hour: Option<i32>,
    /// Quietest occupancy
    pub quietest_occupancy: f64,
    /// Sample count
    pub sample_count: i64,
}

/// Generated insight about occupancy patterns.
#[derive(Debug, Clone)]
pub struct Insight {
    /// Category of the insight
    pub category: InsightCategory,
    /// Severity/importance level (1-5, higher = more important)
    pub importance: u8,
    /// Short title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Associated data (optional - weekday, hour, value)
    pub data: Option<(i32, i32, f64)>,
}

/// Categories of insights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightCategory {
    /// Trend-related insight
    Trend,
    /// Peak hour insight
    Peak,
    /// Quiet time recommendation
    QuietTime,
    /// Unusual pattern detected
    Anomaly,
    /// Day-specific insight
    DayPattern,
    /// Consistency/predictability insight
    Consistency,
}

pub fn midnight_utc(date: NaiveDate) -> DateTime<Utc> {
    date.and_hms_opt(0, 0, 0)
        .expect("midnight (0,0,0) is always valid")
        .and_utc()
}

/// Calculate predictions using the system clock.
/// This is a convenience wrapper for backwards compatibility.
pub fn calculate_predictions(baseline: &[HourlyAverage]) -> Vec<(DateTime<Utc>, f64)> {
    calculate_predictions_with_schedule(baseline, &GymSchedule::default())
}

/// Calculate predictions with a custom schedule using the system clock.
/// This is a convenience wrapper for backwards compatibility.
pub fn calculate_predictions_with_schedule(
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
) -> Vec<(DateTime<Utc>, f64)> {
    calculate_predictions_with_clock(baseline, schedule, &crate::traits::SystemClock)
}

/// Calculate predictions with a custom schedule and clock.
/// This is the core implementation that allows for testability.
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
                .unwrap()
                .with_second(0)
                .unwrap()
                .with_nanosecond(0)
                .unwrap();

            predictions.push((plot_time, avg.avg_percentage));
        }
    }
    predictions
}

/// Find the best time today using the system clock.
/// This is a convenience wrapper for backwards compatibility.
pub fn find_best_time_today(data: &[HourlyAverage]) -> Option<(i32, f64)> {
    find_best_time_today_with_clock(data, &crate::traits::SystemClock)
}

/// Find the best time today with a custom clock.
/// This is the core implementation that allows for testability.
pub fn find_best_time_today_with_clock<C: Clock>(
    data: &[HourlyAverage],
    clock: &C,
) -> Option<(i32, f64)> {
    let now = clock.now_local();
    let today_idx = now.weekday().num_days_from_monday() as i32;

    // Logic Fix: Data is UTC, but we need to find the best time in Local terms.
    let offset_seconds = now.offset().fix().local_minus_utc();
    let seconds_per_week = 7 * 24 * 3600;

    data.iter()
        .map(|d| {
            // Convert UTC record -> Local
            // Local = UTC + Offset
            let utc_seconds = (d.weekday as i64 * 24 + d.hour as i64) * 3600;
            let local_seconds = utc_seconds + offset_seconds as i64;

            // Handle wrapping
            let wrapped_local =
                ((local_seconds % seconds_per_week) + seconds_per_week) % seconds_per_week;

            let local_w = (wrapped_local / 3600) / 24;
            let local_h = (wrapped_local / 3600) % 24;

            (local_w as i32, local_h as i32, d.avg_percentage)
        })
        .filter(|(w, _, _)| *w == today_idx) // Filter for *Local* today
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, h, avg)| (h, avg)) // Return *Local* hour
}

// ==================== Comparative Analytics ====================

/// Build hour-by-hour comparisons between two sets of hourly averages.
///
/// Compares each hour slot between the baseline and current period,
/// calculating changes and trends.
pub fn build_hourly_comparisons(
    baseline: &[HourlyAverage],
    current: &[HourlyAverage],
) -> Vec<HourlyComparison> {
    let mut comparisons = Vec::new();

    // Build lookup map for baseline data
    let baseline_map: HashMap<(i32, i32), &HourlyAverage> =
        baseline.iter().map(|h| ((h.weekday, h.hour), h)).collect();

    // Build lookup map for current data
    let current_map: HashMap<(i32, i32), &HourlyAverage> =
        current.iter().map(|h| ((h.weekday, h.hour), h)).collect();

    // Collect all unique (weekday, hour) pairs
    let mut all_keys: Vec<(i32, i32)> = baseline_map.keys().copied().collect();
    for key in current_map.keys() {
        if !all_keys.contains(key) {
            all_keys.push(*key);
        }
    }
    all_keys.sort();

    for (weekday, hour) in all_keys {
        let baseline_data = baseline_map.get(&(weekday, hour));
        let current_data = current_map.get(&(weekday, hour));

        let baseline_avg = baseline_data.map(|d| d.avg_percentage).unwrap_or(0.0);
        let current_avg = current_data.map(|d| d.avg_percentage).unwrap_or(0.0);
        let baseline_samples = baseline_data.map(|d| d.sample_count).unwrap_or(0);
        let current_samples = current_data.map(|d| d.sample_count).unwrap_or(0);

        let absolute_change = current_avg - baseline_avg;
        let percent_change = if baseline_avg > 0.0 {
            (absolute_change / baseline_avg) * 100.0
        } else if current_avg > 0.0 {
            100.0 // From 0 to something is 100% increase
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

/// Compare two time periods and generate a comprehensive comparison.
///
/// # Arguments
/// * `baseline` - Hourly averages from the baseline/previous period
/// * `current` - Hourly averages from the current/comparison period
/// * `mode` - The comparison mode used
pub fn compare_periods(
    baseline: &[HourlyAverage],
    current: &[HourlyAverage],
    mode: ComparisonMode,
) -> PeriodComparison {
    let hourly_comparisons = build_hourly_comparisons(baseline, current);

    // Calculate overall averages
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

    // Find biggest changes
    let mut sorted_by_increase: Vec<_> = hourly_comparisons
        .iter()
        .filter(|c| c.baseline_samples >= 2 && c.current_samples >= 2)
        .collect();
    sorted_by_increase.sort_by(|a, b| b.percent_change.partial_cmp(&a.percent_change).unwrap());

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

/// Determine the overall trend direction from hourly comparisons.
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

    // Use 3% as threshold for overall trend
    if avg_change > 3.0 {
        TrendDirection::Increasing
    } else if avg_change < -3.0 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

// ==================== Statistical Analysis ====================

/// Calculate statistical summary from hourly averages.
pub fn calculate_stats(data: &[HourlyAverage]) -> Option<OccupancyStats> {
    if data.is_empty() {
        return None;
    }

    let percentages: Vec<f64> = data.iter().map(|h| h.avg_percentage).collect();
    let n = percentages.len();

    let mean = percentages.iter().sum::<f64>() / n as f64;

    let mut sorted = percentages.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let variance = percentages.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
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

/// Analyze patterns for each day of the week.
pub fn analyze_days(data: &[HourlyAverage]) -> Vec<DayAnalysis> {
    const DAY_NAMES: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

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
                .max_by(|a, b| a.avg_percentage.partial_cmp(&b.avg_percentage).unwrap());

            let quietest = day_data
                .iter()
                .min_by(|a, b| a.avg_percentage.partial_cmp(&b.avg_percentage).unwrap());

            DayAnalysis {
                weekday,
                day_name: DAY_NAMES[weekday as usize],
                avg_occupancy,
                peak_hour: peak.map(|h| h.hour),
                peak_occupancy: peak.map(|h| h.avg_percentage).unwrap_or(0.0),
                quietest_hour: quietest.map(|h| h.hour),
                quietest_occupancy: quietest.map(|h| h.avg_percentage).unwrap_or(0.0),
                sample_count: total_samples,
            }
        })
        .collect()
}

/// Find peak hours across the week.
///
/// Returns the top N hours with highest average occupancy.
pub fn find_peak_hours(data: &[HourlyAverage], top_n: usize) -> Vec<(i32, i32, f64)> {
    let mut sorted: Vec<_> = data
        .iter()
        .filter(|h| h.sample_count >= 2)
        .map(|h| (h.weekday, h.hour, h.avg_percentage))
        .collect();

    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    sorted.truncate(top_n);
    sorted
}

/// Find quiet hours across the week.
///
/// Returns the top N hours with lowest average occupancy.
pub fn find_quiet_hours(data: &[HourlyAverage], top_n: usize) -> Vec<(i32, i32, f64)> {
    let mut sorted: Vec<_> = data
        .iter()
        .filter(|h| h.sample_count >= 2 && h.avg_percentage > 0.0)
        .map(|h| (h.weekday, h.hour, h.avg_percentage))
        .collect();

    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    sorted.truncate(top_n);
    sorted
}

/// Find continuous quiet windows (consecutive hours below threshold).
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
                if let Some(start) = window_start {
                    if window_count >= min_hours {
                        windows.push(TimePeriod {
                            weekday,
                            start_hour: start,
                            end_hour: h.hour,
                            avg_occupancy: window_sum / window_count as f64,
                        });
                    }
                }
                window_start = None;
            }
        }

        // Handle window extending to end of day
        if let Some(start) = window_start {
            if window_count >= min_hours {
                windows.push(TimePeriod {
                    weekday,
                    start_hour: start,
                    end_hour: 24,
                    avg_occupancy: window_sum / window_count as f64,
                });
            }
        }
    }

    windows.sort_by(|a, b| a.avg_occupancy.partial_cmp(&b.avg_occupancy).unwrap());
    windows
}

// ==================== Insight Generation ====================

/// Generate human-readable insights from occupancy data.
///
/// Analyzes the data and produces actionable insights about patterns,
/// trends, and recommendations.
pub fn generate_insights(
    current: &[HourlyAverage],
    baseline: Option<&[HourlyAverage]>,
) -> Vec<Insight> {
    let mut insights = Vec::new();

    // Get statistics
    if let Some(stats) = calculate_stats(current) {
        // Consistency insight
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
            title: format!("Occupancy is {}", consistency_level),
            description: format!(
                "Average occupancy is {:.1}% with a standard deviation of {:.1}%. Range: {:.1}% \
                 to {:.1}%.",
                stats.mean, stats.std_dev, stats.min, stats.max
            ),
            data: None,
        });
    }

    // Day analysis insights
    let day_analysis = analyze_days(current);
    if let Some(busiest_day) = day_analysis
        .iter()
        .max_by(|a, b| a.avg_occupancy.partial_cmp(&b.avg_occupancy).unwrap())
    {
        if busiest_day.sample_count >= 5 {
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
    }

    if let Some(quietest_day) = day_analysis
        .iter()
        .filter(|d| d.sample_count >= 5)
        .min_by(|a, b| a.avg_occupancy.partial_cmp(&b.avg_occupancy).unwrap())
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

    // Peak hours insight
    let peaks = find_peak_hours(current, 3);
    if !peaks.is_empty() {
        const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        let peak_desc: Vec<String> = peaks
            .iter()
            .map(|(w, h, p)| format!("{} {}:00 ({:.0}%)", DAY_NAMES[*w as usize], h, p))
            .collect();

        insights.push(Insight {
            category: InsightCategory::Peak,
            importance: 3,
            title: "Busiest times to avoid".to_string(),
            description: format!("Peak hours: {}", peak_desc.join(", ")),
            data: Some(peaks[0]),
        });
    }

    // Quiet windows insight
    let quiet_windows = find_quiet_windows(current, 40.0, 2);
    if !quiet_windows.is_empty() {
        const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        let best_window = &quiet_windows[0];
        insights.push(Insight {
            category: InsightCategory::QuietTime,
            importance: 5,
            title: "Best workout window".to_string(),
            description: format!(
                "{} {}:00-{}:00 averages only {:.1}% occupancy. {} more quiet windows available.",
                DAY_NAMES[best_window.weekday as usize],
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

    // Trend insights (if baseline provided)
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

        // Biggest changes
        if !comparison.biggest_increases.is_empty() {
            const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
            let (w, h, change) = comparison.biggest_increases[0];
            insights.push(Insight {
                category: InsightCategory::Anomaly,
                importance: 3,
                title: "Significant occupancy increase".to_string(),
                description: format!(
                    "{} at {}:00 has seen a {:.0}% increase in occupancy. You may want to avoid \
                     this time slot.",
                    DAY_NAMES[w as usize], h, change
                ),
                data: Some((w, h, change)),
            });
        }
    }

    // Sort by importance (highest first)
    insights.sort_by(|a, b| b.importance.cmp(&a.importance));
    insights
}

/// Get the weekday name from index (0=Monday).
pub fn weekday_name(weekday: i32) -> &'static str {
    const DAY_NAMES: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    DAY_NAMES.get(weekday as usize).unwrap_or(&"Unknown")
}

/// Get the short weekday name from index (0=Monday).
pub fn weekday_short(weekday: i32) -> &'static str {
    const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    DAY_NAMES.get(weekday as usize).unwrap_or(&"???")
}

#[cfg(test)]
mod tests {
    use chrono::{Datelike, NaiveDate, Timelike};

    use super::*;

    // ==================== midnight_utc Tests ====================

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

    // ==================== calculate_predictions Tests ====================

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
        // Create baseline with all hours for all days
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
        // At most 2 predictions (for +1h and +2h)
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_calculate_predictions_respects_schedule() {
        // Create a schedule that's always closed
        let schedule = GymSchedule::new_for_test(0, 0, 0, 0);

        let baseline = vec![HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 30.0,
            sample_count: 5,
        }];

        let result = calculate_predictions_with_schedule(&baseline, &schedule);
        // Should be empty since gym is always closed
        assert!(result.is_empty());
    }

    // ==================== find_best_time_today Tests ====================

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
                avg_percentage: 20.0, // Lowest
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
        // Note: hour might be adjusted for timezone, but avg should be lowest
    }

    #[test]
    fn test_find_best_time_filters_by_today() {
        let today_idx = Local::now().weekday().num_days_from_monday() as i32;
        let other_day = (today_idx + 1) % 7;

        let data = vec![
            HourlyAverage {
                weekday: other_day, // Different day
                hour: 10,
                avg_percentage: 10.0, // Lower but wrong day
                sample_count: 5,
            },
            HourlyAverage {
                weekday: today_idx, // Today
                hour: 14,
                avg_percentage: 30.0,
                sample_count: 5,
            },
        ];

        let result = find_best_time_today(&data);
        // Should find the one for today, not the lower one on another day
        // (The exact behavior depends on timezone, but it should find something for
        // today)
        assert!(result.is_some());
    }

    #[test]
    fn test_predictions_with_open_schedule() {
        // Schedule open 24/7
        let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

        // Create full week of data
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
        // Should have predictions since gym is always open
        // (might be 0-2 depending on current time)
        assert!(result.len() <= 2);
    }

    // ==================== Clock-Aware Function Tests ====================

    mod clock_tests {
        use chrono::TimeZone;

        use super::*;
        use crate::traits::MockClock;

        #[test]
        fn test_predictions_with_mock_clock() {
            // Set clock to Monday 10:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap(); // Monday
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24); // 24/7 open

            // Create baseline with data for hours 11 and 12 on Monday (weekday 0)
            let baseline = vec![
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 11,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 12,
                    avg_percentage: 50.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            // Should get predictions for 11:00 and 12:00 (now + 1h and now + 2h)
            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 30.0); // Hour 11
            assert_eq!(predictions[1].1, 50.0); // Hour 12
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

            // At 10:00, should get predictions for 11:00 and 12:00
            let predictions1 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
            assert_eq!(predictions1.len(), 2);
            assert_eq!(predictions1[0].1, 25.0);
            assert_eq!(predictions1[1].1, 45.0);

            // Advance clock by 1 hour to 11:00
            clock.advance(ChronoDuration::hours(1));

            // Now should get predictions for 12:00 and 13:00
            let predictions2 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
            assert_eq!(predictions2.len(), 2);
            assert_eq!(predictions2[0].1, 45.0);
            assert_eq!(predictions2[1].1, 65.0);
        }

        #[test]
        fn test_find_best_time_with_mock_clock() {
            // Set clock to Monday
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap(); // Monday
            let clock = MockClock::new(fixed_time);

            // Data for Monday (weekday 0 in UTC)
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
                    avg_percentage: 15.0, // Lowest
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
            // The best time should have the lowest percentage
            assert_eq!(avg, 15.0);
        }
    }

    // ==================== Week Boundary Tests ====================

    mod week_boundary_tests {
        use chrono::TimeZone;

        use super::*;
        use crate::traits::MockClock;

        #[test]
        fn test_predictions_crossing_sunday_to_monday() {
            // Set clock to Sunday 23:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 23, 0, 0).unwrap(); // Sunday
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24); // 24/7 open

            // Data for Sunday (weekday 6) hour 23 doesn't matter for predictions
            // Predictions look at +1h (Monday 00:00) and +2h (Monday 01:00)
            let baseline = vec![
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 0,    // Midnight
                    avg_percentage: 25.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 1,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            // Should get predictions for Monday 00:00 and 01:00
            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 25.0); // Monday 00:00
            assert_eq!(predictions[1].1, 30.0); // Monday 01:00
        }

        #[test]
        fn test_predictions_crossing_saturday_to_sunday() {
            // Set clock to Saturday 22:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 15, 22, 0, 0).unwrap(); // Saturday
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24); // 24/7 open

            // Predictions for Saturday 23:00 and Sunday 00:00
            let baseline = vec![
                HourlyAverage {
                    weekday: 5, // Saturday
                    hour: 23,
                    avg_percentage: 40.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 6, // Sunday
                    hour: 0,
                    avg_percentage: 15.0, // Lower on Sunday morning
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 40.0); // Saturday 23:00
            assert_eq!(predictions[1].1, 15.0); // Sunday 00:00
        }

        #[test]
        fn test_predictions_at_year_boundary() {
            // Set clock to December 31, 23:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 12, 31, 23, 0, 0).unwrap(); // Tuesday
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            // Dec 31, 2024 is Tuesday (weekday 1), Jan 1, 2025 is Wednesday (weekday 2)
            let baseline = vec![
                HourlyAverage {
                    weekday: 2, // Wednesday (Jan 1)
                    hour: 0,
                    avg_percentage: 10.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 2, // Wednesday (Jan 1)
                    hour: 1,
                    avg_percentage: 20.0,
                    sample_count: 10,
                },
            ];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            // Should correctly handle year boundary
            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0].1, 10.0);
            assert_eq!(predictions[1].1, 20.0);
        }

        #[test]
        fn test_find_best_time_near_midnight_start_of_week() {
            // Set clock to Monday 00:30 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17, 0, 30, 0).unwrap(); // Monday
            let clock = MockClock::new(fixed_time);

            // Data for Monday (weekday 0)
            let data = vec![
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 0,
                    avg_percentage: 5.0, // Very low at midnight
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 12,
                    avg_percentage: 70.0,
                    sample_count: 10,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            assert!(result.is_some());
            let (_, avg) = result.unwrap();
            // Should find the lowest (5.0)
            assert_eq!(avg, 5.0);
        }

        #[test]
        fn test_find_best_time_near_midnight_end_of_week() {
            // Set clock to Sunday 23:30 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 23, 30, 0).unwrap(); // Sunday
            let clock = MockClock::new(fixed_time);

            // Data for Sunday (weekday 6)
            let data = vec![
                HourlyAverage {
                    weekday: 6, // Sunday
                    hour: 10,
                    avg_percentage: 35.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 6, // Sunday
                    hour: 23,
                    avg_percentage: 8.0, // Low late Sunday
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
            // Set clock to Sunday 22:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 16, 22, 0, 0).unwrap();
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            // Only have data for Sunday 23:00, missing Monday 00:00
            let baseline = vec![HourlyAverage {
                weekday: 6, // Sunday
                hour: 23,
                avg_percentage: 45.0,
                sample_count: 10,
            }];

            let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

            // Should only get 1 prediction (Sunday 23:00), not Monday 00:00
            assert_eq!(predictions.len(), 1);
            assert_eq!(predictions[0].1, 45.0);
        }

        #[test]
        fn test_find_best_time_no_data_for_current_day() {
            // Set clock to Wednesday
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 19, 10, 0, 0).unwrap(); // Wednesday
            let clock = MockClock::new(fixed_time);

            // Only have data for Monday and Tuesday, not Wednesday
            let data = vec![
                HourlyAverage {
                    weekday: 0, // Monday
                    hour: 10,
                    avg_percentage: 20.0,
                    sample_count: 10,
                },
                HourlyAverage {
                    weekday: 1, // Tuesday
                    hour: 10,
                    avg_percentage: 30.0,
                    sample_count: 10,
                },
            ];

            let result = find_best_time_today_with_clock(&data, &clock);
            // Should return None since no data for Wednesday
            assert!(result.is_none());
        }

        #[test]
        fn test_predictions_all_week_data_available() {
            // Set clock to Friday 11:00 UTC
            let fixed_time = Utc.with_ymd_and_hms(2024, 6, 21, 11, 0, 0).unwrap(); // Friday
            let clock = MockClock::new(fixed_time);
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            // Full week of data
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

            // Should get 2 predictions for Friday 12:00 and 13:00
            assert_eq!(predictions.len(), 2);
            // Friday is weekday 4, hour 12 -> 4*10 + 12 = 52
            assert_eq!(predictions[0].1, 52.0);
            // Friday is weekday 4, hour 13 -> 4*10 + 13 = 53
            assert_eq!(predictions[1].1, 53.0);
        }

        #[test]
        fn test_monday_to_sunday_full_cycle() {
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);

            // Create data for all weekdays at hour 10
            let baseline: Vec<HourlyAverage> = (0..7)
                .map(|weekday| HourlyAverage {
                    weekday,
                    hour: 10,
                    avg_percentage: (weekday as f64) * 10.0 + 5.0,
                    sample_count: 10,
                })
                .collect();

            // Test predictions for each day of the week
            for day in 0..7 {
                // June 17, 2024 is Monday (weekday 0)
                let fixed_time = Utc.with_ymd_and_hms(2024, 6, 17 + day, 9, 0, 0).unwrap();
                let clock = MockClock::new(fixed_time);

                let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

                // At 09:00, should predict for 10:00 (now + 1h) if data exists
                if !predictions.is_empty() {
                    // The percentage should match the day's data
                    let expected_weekday = (day as u32) % 7;
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

    // ==================== Property-Based Tests ====================

    mod proptest_tests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn midnight_utc_always_at_midnight(
                year in 2000i32..2100,
                month in 1u32..=12,
                day in 1u32..=28  // Safe range for all months
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
                    // The returned avg should be one of the values we provided
                    // (may be adjusted for timezone, but percentage shouldn't change)
                    prop_assert!(percentages.iter().any(|&p| (p - avg).abs() < 0.001),
                        "Returned avg {} not found in input", avg);
                }
            }
        }
    }

    // ==================== Comparative Analytics Tests ====================

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
            assert!((result[0].percent_change - 25.0).abs() < 0.01); // 10/40 = 25%
        }

        #[test]
        fn test_build_hourly_comparisons_missing_baseline() {
            let baseline = vec![];
            let current = vec![make_hourly_avg(0, 10, 50.0, 5)];

            let result = build_hourly_comparisons(&baseline, &current);

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].baseline_avg, 0.0);
            assert_eq!(result[0].current_avg, 50.0);
            assert_eq!(result[0].percent_change, 100.0); // From 0 to something
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
                current_samples: 1, // Too few samples
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
                    percent_change: 2.0, // Within ±3%
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

    // ==================== Statistical Analysis Tests ====================

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
            assert_eq!(result.median, 50.0); // (40 + 60) / 2
            assert_eq!(result.min, 20.0);
            assert_eq!(result.max, 80.0);
            assert_eq!(result.sample_count, 4);
            assert!(result.std_dev > 0.0);
        }

        #[test]
        fn test_analyze_days() {
            let data = vec![
                make_hourly_avg(0, 10, 30.0, 5), // Monday 10:00
                make_hourly_avg(0, 11, 50.0, 5), // Monday 11:00
                make_hourly_avg(1, 10, 40.0, 5), // Tuesday 10:00
            ];

            let result = analyze_days(&data);

            assert_eq!(result.len(), 7);

            // Check Monday
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
                make_hourly_avg(0, 11, 80.0, 5), // Peak
                make_hourly_avg(1, 10, 70.0, 5),
                make_hourly_avg(2, 15, 90.0, 5), // Highest
            ];

            let result = find_peak_hours(&data, 2);

            assert_eq!(result.len(), 2);
            assert_eq!(result[0], (2, 15, 90.0)); // Highest first
            assert_eq!(result[1], (0, 11, 80.0));
        }

        #[test]
        fn test_find_quiet_hours() {
            let data = vec![
                make_hourly_avg(0, 10, 10.0, 5), // Quietest
                make_hourly_avg(0, 11, 80.0, 5),
                make_hourly_avg(1, 10, 20.0, 5), // Second quietest
                make_hourly_avg(2, 15, 90.0, 5),
            ];

            let result = find_quiet_hours(&data, 2);

            assert_eq!(result.len(), 2);
            assert_eq!(result[0], (0, 10, 10.0)); // Quietest first
            assert_eq!(result[1], (1, 10, 20.0));
        }

        #[test]
        fn test_find_quiet_windows() {
            let data = vec![
                make_hourly_avg(0, 6, 20.0, 5),
                make_hourly_avg(0, 7, 25.0, 5),
                make_hourly_avg(0, 8, 30.0, 5),
                make_hourly_avg(0, 9, 70.0, 5), // Break
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

    // ==================== Insight Generation Tests ====================

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
            // Should have at least consistency, day pattern, and peak insights
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
                    (8..20).map(move |hour| {
                        make_hourly_avg(weekday, hour, 60.0, 10) // Higher than baseline
                    })
                })
                .collect();

            let result = generate_insights(&current, Some(&baseline));

            // Should have trend insight
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

            // Check that insights are sorted by importance (descending)
            for window in result.windows(2) {
                assert!(window[0].importance >= window[1].importance);
            }
        }
    }

    // ==================== Utility Function Tests ====================

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
