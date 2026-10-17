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

/// Days an hour must have been seen on before its average counts.
pub const MIN_DAYS: i64 = 2;

/// Whether `slot` was seen on enough days for its average to count.
pub(super) fn is_reliable(slot: &HourlyAverage) -> bool {
    slot.days >= MIN_DAYS
}

/// The `top_n` busiest `(weekday, hour, average)` slots.
pub fn find_peak_hours(data: &[HourlyAverage], top_n: usize) -> Vec<(i32, i32, f64)> {
    let mut sorted: Vec<_> = data
        .iter()
        .filter(|h| is_reliable(h))
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
        .filter(|h| is_reliable(h) && h.avg_percentage > 0.0)
        .map(|h| (h.weekday, h.hour, h.avg_percentage))
        .collect();
    sorted.sort_by(|a, b| a.2.total_cmp(&b.2));
    sorted.truncate(top_n);
    sorted
}

/// Runs of at least `min_hours` consecutive hours at or below `threshold`,
/// quietest first. A window ends after its last hour (so one running until
/// closing ends at closing time).
#[tracing::instrument(skip_all, fields(threshold, min_hours))]
pub fn find_quiet_windows(
    data: &[HourlyAverage],
    threshold: f64,
    min_hours: usize,
) -> Vec<TimePeriod> {
    let mut windows = Vec::new();
    for weekday in 0i32..7 {
        let mut hours: Vec<&HourlyAverage> = data
            .iter()
            .filter(|h| h.weekday == weekday && is_reliable(h))
            .collect();
        hours.sort_by_key(|h| h.hour);
        let mut run: Vec<&HourlyAverage> = Vec::with_capacity(hours.len());
        for h in hours {
            let continues = run.last().is_none_or(|last| last.hour + 1 == h.hour);
            if !continues {
                close_run(&mut windows, &mut run, min_hours);
            }
            if h.avg_percentage <= threshold {
                run.push(h);
            } else {
                close_run(&mut windows, &mut run, min_hours);
            }
        }
        close_run(&mut windows, &mut run, min_hours);
    }
    windows.sort_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy));
    windows
}

/// Adds `run` (consecutive quiet hours of one weekday) to `windows` if it
/// is long enough, and empties it.
fn close_run(windows: &mut Vec<TimePeriod>, run: &mut Vec<&HourlyAverage>, min_hours: usize) {
    if run.len() >= min_hours
        && let (Some(first), Some(last)) = (run.first(), run.last())
    {
        #[allow(clippy::cast_precision_loss)] // a run is at most 24 hours
        let count = run.len() as f64;
        windows.push(TimePeriod {
            weekday: first.weekday,
            start_hour: first.hour,
            end_hour: last.hour + 1,
            avg_occupancy: run.iter().map(|h| h.avg_percentage).sum::<f64>() / count,
        });
    }
    run.clear();
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

    #[test]
    fn test_slots_need_two_days_not_two_readings() {
        // 60 readings, but all from one day.
        let data = vec![seen(0, 10, 90.0, 1), seen(0, 11, 50.0, 2)];
        assert_eq!(find_peak_hours(&data, 5), vec![(0, 11, 50.0)]);
        assert_eq!(find_quiet_hours(&data, 5), vec![(0, 11, 50.0)]);
        let windows = find_quiet_windows(&[seen(1, 9, 5.0, 1), seen(1, 10, 5.0, 1)], 40.0, 2);
        assert!(windows.is_empty(), "{windows:?}");
    }

    #[test]
    fn test_quiet_window_until_closing_ends_after_its_last_hour() {
        // Saturday, open until 21:00: the last hour is 20:00.
        let data: Vec<_> = (17..21).map(|h| seen(5, h, 6.0, 4)).collect();
        let windows = find_quiet_windows(&data, 40.0, 2);
        assert_eq!(windows.len(), 1);
        assert_eq!((windows[0].start_hour, windows[0].end_hour), (17, 21));
    }

    #[test]
    fn test_quiet_window_splits_at_missing_hours() {
        // 9:00 and 10:00 quiet, 11:00 missing, 12:00 and 13:00 quiet.
        let data = vec![
            seen(2, 9, 10.0, 4),
            seen(2, 10, 10.0, 4),
            seen(2, 12, 20.0, 4),
            seen(2, 13, 20.0, 4),
        ];
        let windows = find_quiet_windows(&data, 40.0, 2);
        let spans: Vec<(i32, i32)> = windows.iter().map(|w| (w.start_hour, w.end_hour)).collect();
        assert_eq!(spans, vec![(9, 11), (12, 14)]);
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
