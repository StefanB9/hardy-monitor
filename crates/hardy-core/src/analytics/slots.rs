//! Whether a slot has enough data, and longer quiet windows.

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

/// Spread between days, in points, above which an hour's average is a poor
/// guide.
pub const UNPREDICTABLE_SPREAD: f64 = 10.0;

/// Whether days differing by `spread` points (standard deviation) make an
/// average unpredictable.
pub fn is_unpredictable(spread: f64) -> bool {
    spread > UNPREDICTABLE_SPREAD
}

/// Whether `slot` was seen on enough days for its average to count.
pub(super) fn is_reliable(slot: &HourlyAverage) -> bool {
    slot.days >= MIN_DAYS
}

/// Runs of at least `min_hours` consecutive hours at or below `threshold`,
/// quietest first. A run longer than `max_hours` gives its quietest stretch
/// of `max_hours` (the earliest of equally quiet ones): "the whole Saturday
/// is quiet" is no advice on when to go. A window ends after its last hour
/// (so one running until closing ends at closing time).
#[tracing::instrument(skip_all, fields(threshold, min_hours, max_hours))]
pub fn find_quiet_windows(
    data: &[HourlyAverage],
    threshold: f64,
    min_hours: usize,
    max_hours: usize,
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
                close_run(&mut windows, &mut run, min_hours, max_hours);
            }
            if h.avg_percentage <= threshold {
                run.push(h);
            } else {
                close_run(&mut windows, &mut run, min_hours, max_hours);
            }
        }
        close_run(&mut windows, &mut run, min_hours, max_hours);
    }
    windows.sort_by(|a, b| a.avg_occupancy.total_cmp(&b.avg_occupancy));
    windows
}

/// Adds `run` (consecutive quiet hours of one weekday) to `windows` if it
/// is long enough, cut to its quietest `max_hours`, and empties it.
fn close_run(
    windows: &mut Vec<TimePeriod>,
    run: &mut Vec<&HourlyAverage>,
    min_hours: usize,
    max_hours: usize,
) {
    if run.len() >= min_hours {
        let len = run.len().min(max_hours.max(min_hours));
        let quietest = run
            .windows(len)
            .map(|stretch| stretch.iter().map(|h| h.avg_percentage).sum::<f64>())
            .enumerate()
            // `min_by` keeps the first of equal sums: the earliest stretch.
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((start, sum)) = quietest
            && let (Some(first), Some(last)) = (run.get(start), run.get(start + len - 1))
        {
            #[allow(clippy::cast_precision_loss)] // a run is at most 24 hours
            let count = len as f64;
            windows.push(TimePeriod {
                weekday: first.weekday,
                start_hour: first.hour,
                end_hour: last.hour + 1,
                avg_occupancy: sum / count,
            });
        }
    }
    run.clear();
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

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
    fn test_unpredictable_above_ten_points() {
        assert!(!is_unpredictable(0.0));
        assert!(!is_unpredictable(UNPREDICTABLE_SPREAD));
        assert!(is_unpredictable(10.1));
        assert!(is_unpredictable(25.8));
    }

    #[test]
    fn test_slots_need_two_days_not_two_readings() {
        // 60 readings, but all from one day.
        let windows = find_quiet_windows(&[seen(1, 9, 5.0, 1), seen(1, 10, 5.0, 1)], 40.0, 2, 24);
        assert!(windows.is_empty(), "{windows:?}");
    }

    #[test]
    fn test_quiet_window_until_closing_ends_after_its_last_hour() {
        // Saturday, open until 21:00: the last hour is 20:00.
        let data: Vec<_> = (17..21).map(|h| seen(5, h, 6.0, 4)).collect();
        let windows = find_quiet_windows(&data, 40.0, 2, 24);
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
        let windows = find_quiet_windows(&data, 40.0, 2, 24);
        let spans: Vec<(i32, i32)> = windows.iter().map(|w| (w.start_hour, w.end_hour)).collect();
        assert_eq!(spans, vec![(9, 11), (12, 14)]);
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

        let result = find_quiet_windows(&data, 40.0, 2, 24);

        assert!(!result.is_empty());
        let window = &result[0];
        assert_eq!(window.weekday, 0);
        assert_eq!(window.start_hour, 6);
        assert!(window.end_hour >= 8);
    }

    #[test]
    fn test_long_quiet_run_keeps_its_quietest_stretch() {
        // Saturday 9:00–21:00 all quiet, quietest from 17:00 to 20:00.
        let data: Vec<_> = (9..21)
            .map(|h| seen(5, h, if (17..20).contains(&h) { 5.0 } else { 15.0 }, 4))
            .collect();
        let windows = find_quiet_windows(&data, 40.0, 2, 3);
        assert_eq!(windows.len(), 1, "{windows:?}");
        assert_eq!((windows[0].start_hour, windows[0].end_hour), (17, 20));
        assert!((windows[0].avg_occupancy - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_quiet_run_within_the_cap_is_kept_whole() {
        let data: Vec<_> = (17..20).map(|h| seen(5, h, 6.0, 4)).collect();
        let windows = find_quiet_windows(&data, 40.0, 2, 3);
        assert_eq!(windows.len(), 1);
        assert_eq!((windows[0].start_hour, windows[0].end_hour), (17, 20));
    }

    #[test]
    fn test_capped_window_prefers_the_earliest_of_equal_stretches() {
        let data: Vec<_> = (9..15).map(|h| seen(1, h, 10.0, 4)).collect();
        let windows = find_quiet_windows(&data, 40.0, 2, 3);
        assert_eq!((windows[0].start_hour, windows[0].end_hour), (9, 12));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_quiet_windows_stay_within_bounds_and_threshold(
            pcts in prop::collection::vec(0.0f64..100.0, 1..18),
            min_hours in 1usize..4,
            extra in 0usize..4,
        ) {
            let max_hours = min_hours + extra;
            let data: Vec<_> = pcts
                .iter()
                .enumerate()
                .map(|(i, &p)| seen(3, 6 + i32::try_from(i).unwrap_or(0), p, 3))
                .collect();
            for w in find_quiet_windows(&data, 40.0, min_hours, max_hours) {
                let len = usize::try_from(w.end_hour - w.start_hour).unwrap_or(0);
                prop_assert!((min_hours..=max_hours).contains(&len), "{w:?}");
                for h in w.start_hour..w.end_hour {
                    let slot = data.iter().find(|d| d.hour == h);
                    prop_assert!(slot.is_some_and(|d| d.avg_percentage <= 40.0), "{w:?}");
                }
            }
        }
    }
}
