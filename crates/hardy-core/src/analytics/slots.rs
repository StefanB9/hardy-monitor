//! Busiest and quietest slots, and longer quiet windows.

use crate::db::HourlyAverage;

/// Consecutive hours of one weekday.
#[derive(Debug, Clone)]
pub struct TimePeriod {
    pub weekday: i32,
    pub start_hour: i32,
    pub end_hour: i32,
    pub avg_occupancy: f64,
}

/// The `top_n` busiest `(weekday, hour, average)` slots.
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

/// The `top_n` quietest `(weekday, hour, average)` slots.
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

/// Runs of at least `min_hours` consecutive hours below `threshold`.
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
                    #[allow(clippy::cast_precision_loss)]
                    let window_count_f64 = window_count as f64;

                    windows.push(TimePeriod {
                        weekday,
                        start_hour: start,
                        end_hour: h.hour,
                        avg_occupancy: window_sum / window_count_f64,
                    });
                }
                window_start = None;
            }
        }

        if let Some(start) = window_start
            && window_count >= min_hours
        {
            #[allow(clippy::cast_precision_loss)]
            let window_count_f64 = window_count as f64;

            windows.push(TimePeriod {
                weekday,
                start_hour: start,
                end_hour: 24,
                avg_occupancy: window_sum / window_count_f64,
            });
        }
    }

    windows.sort_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy));
    windows
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
