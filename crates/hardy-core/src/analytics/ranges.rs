//! Busiest and quietest times as ranges: the top hours of the week, with
//! neighbouring hours of a weekday joined.

use super::{
    slots::{is_reliable, is_unpredictable},
    weekday_short,
};
use crate::db::HourlyAverage;

/// Hours taken before joining: about the top seventh of a week's open
/// hours.
pub const TOP_HOURS: usize = 15;
/// Ranges the Insights lists show.
pub const LISTED_RANGES: usize = 5;

/// Consecutive hours of one weekday, with their average and span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotRange {
    pub weekday: i32,
    pub start_hour: i32,
    /// After the last hour: 17–20 covers 17:00, 18:00 and 19:00.
    pub end_hour: i32,
    pub average: f64,
    pub min: f64,
    pub max: f64,
    /// The largest spread between days among its hours, in points.
    pub max_spread: f64,
}

impl SlotRange {
    /// "17:00–20:00".
    pub fn hours_text(&self) -> String {
        format!("{:02}:00–{:02}:00", self.start_hour, self.end_hour)
    }

    /// Whether any of its hours varies a lot between days.
    pub fn varies(&self) -> bool {
        is_unpredictable(self.max_spread)
    }

    /// "Mon 17:00–20:00".
    pub fn label(&self) -> String {
        format!("{} {}", weekday_short(self.weekday), self.hours_text())
    }
}

/// The `top_hours` busiest hours seen on enough days, joined into ranges,
/// busiest (by average) first.
pub fn busiest_ranges(data: &[HourlyAverage], top_hours: usize) -> Vec<SlotRange> {
    top_ranges(data, top_hours, true)
}

/// The `top_hours` quietest hours seen on enough days (above 0%: a closed
/// gym is not quiet), joined into ranges, quietest first.
pub fn quietest_ranges(data: &[HourlyAverage], top_hours: usize) -> Vec<SlotRange> {
    top_ranges(data, top_hours, false)
}

fn top_ranges(data: &[HourlyAverage], top_hours: usize, busy: bool) -> Vec<SlotRange> {
    let strongest_first = |a: f64, b: f64| {
        let order = a.total_cmp(&b);
        if busy { order.reverse() } else { order }
    };
    let mut picked: Vec<&HourlyAverage> = data
        .iter()
        .filter(|h| is_reliable(h) && (busy || h.avg_percentage > 0.0))
        .collect();
    picked.sort_by(|a, b| strongest_first(a.avg_percentage, b.avg_percentage));
    picked.truncate(top_hours);
    picked.sort_by_key(|h| (h.weekday, h.hour));

    let mut ranges = Vec::new();
    let mut run: Vec<&HourlyAverage> = Vec::with_capacity(picked.len());
    for hour in picked {
        let joins = run
            .last()
            .is_some_and(|last| last.weekday == hour.weekday && last.hour + 1 == hour.hour);
        if !joins {
            ranges.extend(range_of(&run));
            run.clear();
        }
        run.push(hour);
    }
    ranges.extend(range_of(&run));
    ranges.sort_by(|a, b| strongest_first(a.average, b.average));
    ranges
}

/// `run` (consecutive hours of one weekday) as a range; `None` if empty.
fn range_of(run: &[&HourlyAverage]) -> Option<SlotRange> {
    let (first, last) = (run.first()?, run.last()?);
    let values = run.iter().map(|h| h.avg_percentage);
    #[allow(clippy::cast_precision_loss)] // a run is at most 24 hours
    let average = values.clone().sum::<f64>() / run.len() as f64;
    Some(SlotRange {
        weekday: first.weekday,
        start_hour: first.hour,
        end_hour: last.hour + 1,
        average,
        min: values.clone().fold(f64::INFINITY, f64::min),
        max: values.fold(f64::NEG_INFINITY, f64::max),
        max_spread: run.iter().map(|h| h.std_dev).fold(0.0, f64::max),
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

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

    fn spans(ranges: &[SlotRange]) -> Vec<(i32, i32, i32)> {
        ranges
            .iter()
            .map(|r| (r.weekday, r.start_hour, r.end_hour))
            .collect()
    }

    #[test]
    fn test_neighbouring_busy_hours_join() {
        // Monday 17–19 busy, Tuesday 18 busy, the rest quiet.
        let mut data: Vec<_> = (6..23).map(|h| seen(0, h, 10.0, 4)).collect();
        data.extend((6..23).map(|h| seen(1, h, 10.0, 4)));
        for (h, pct) in [(17, 42.0), (18, 55.0), (19, 42.0)] {
            data.push(seen(0, h, pct, 4));
        }
        data.push(seen(1, 18, 50.0, 4));
        let data: Vec<_> = dedupe_last(data);

        let ranges = busiest_ranges(&data, 4);
        assert_eq!(spans(&ranges), vec![(1, 18, 19), (0, 17, 20)]);
        let monday = ranges[1];
        assert_relative_eq!(monday.average, 139.0 / 3.0);
        assert_relative_eq!(monday.min, 42.0);
        assert_relative_eq!(monday.max, 55.0);
    }

    /// Keeps the last entry per (weekday, hour), so tests can override.
    fn dedupe_last(data: Vec<HourlyAverage>) -> Vec<HourlyAverage> {
        let mut out: Vec<HourlyAverage> = Vec::with_capacity(data.len());
        for h in data {
            out.retain(|o| (o.weekday, o.hour) != (h.weekday, h.hour));
            out.push(h);
        }
        out
    }

    #[test]
    fn test_quiet_evening_is_one_range() {
        let mut data: Vec<_> = (9..21).map(|h| seen(5, h, 20.0, 4)).collect();
        data.extend((16..21).map(|h| seen(5, h, 6.0, 4)));
        data.push(seen(4, 22, 7.0, 4));
        let data = dedupe_last(data);
        let ranges = quietest_ranges(&data, 6);
        assert_eq!(spans(&ranges), vec![(5, 16, 21), (4, 22, 23)]);
    }

    #[test]
    fn test_ranges_skip_thin_and_closed_hours() {
        let data = vec![
            seen(2, 10, 1.0, 1), // one day only
            seen(2, 11, 0.0, 4), // closed
            seen(2, 12, 5.0, 4),
        ];
        assert_eq!(spans(&quietest_ranges(&data, 3)), vec![(2, 12, 13)]);
        // A closed hour is not quiet, but it can be among the "busiest" of
        // a thin week.
        assert_eq!(spans(&busiest_ranges(&data, 2)), vec![(2, 11, 13)]);
    }

    #[test]
    fn test_hours_of_different_days_do_not_join() {
        // Sunday 22:00 and Monday 06:00 are not neighbours, nor are
        // Monday 09:00 and Monday 11:00.
        let data = vec![
            seen(6, 22, 5.0, 4),
            seen(0, 6, 5.0, 4),
            seen(0, 9, 5.0, 4),
            seen(0, 11, 5.0, 4),
        ];
        assert_eq!(quietest_ranges(&data, 4).len(), 4);
    }

    #[test]
    fn test_range_varies_when_any_hour_does() {
        let spread = |weekday, hour, std_dev| HourlyAverage {
            std_dev,
            ..seen(weekday, hour, 6.0, 4)
        };
        let data = vec![spread(6, 17, 2.0), spread(6, 18, 25.8), spread(6, 19, 3.0)];
        let ranges = quietest_ranges(&data, 3);
        assert_eq!(ranges.len(), 1);
        assert_relative_eq!(ranges[0].max_spread, 25.8);
        assert!(ranges[0].varies());
        let calm = quietest_ranges(&[spread(2, 21, 4.0), spread(2, 22, 9.0)], 2);
        assert!(!calm[0].varies());
    }

    #[test]
    fn test_range_labels() {
        let range = SlotRange {
            weekday: 5,
            start_hour: 9,
            end_hour: 21,
            average: 6.0,
            min: 5.0,
            max: 7.0,
            max_spread: 0.0,
        };
        assert_eq!(range.hours_text(), "09:00–21:00");
        assert_eq!(range.label(), "Sat 09:00–21:00");
    }

    #[test]
    fn test_no_data_no_ranges() {
        assert_eq!(busiest_ranges(&[], TOP_HOURS), Vec::new());
        assert_eq!(quietest_ranges(&[], TOP_HOURS), Vec::new());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_ranges_partition_the_top_hours(
            cells in prop::collection::vec((0.0f64..100.0, 1i64..5), 7 * 17),
            top in 1usize..30,
            busy in any::<bool>(),
        ) {
            let data: Vec<_> = cells
                .iter()
                .enumerate()
                .map(|(i, &(pct, days))| {
                    let i = i32::try_from(i).unwrap_or(0);
                    seen(i / 17, 6 + i % 17, pct, days)
                })
                .collect();
            let ranges = if busy {
                busiest_ranges(&data, top)
            } else {
                quietest_ranges(&data, top)
            };
            let mut eligible: Vec<_> = data
                .iter()
                .filter(|h| h.days >= 2 && (busy || h.avg_percentage > 0.0))
                .collect();
            eligible.sort_by(|a, b| {
                let order = a.avg_percentage.total_cmp(&b.avg_percentage);
                if busy { order.reverse() } else { order }
            });
            let cutoff = eligible.get(top.min(eligible.len()).saturating_sub(1)).map(|h| h.avg_percentage);
            let covered: usize = ranges
                .iter()
                .map(|r| usize::try_from(r.end_hour - r.start_hour).unwrap_or(0))
                .sum();
            prop_assert_eq!(covered, top.min(eligible.len()));
            for r in &ranges {
                prop_assert!(r.start_hour < r.end_hour);
                prop_assert!(r.min <= r.average + 1e-9 && r.average <= r.max + 1e-9);
                for h in r.start_hour..r.end_hour {
                    let cell = data.iter().find(|d| d.weekday == r.weekday && d.hour == h);
                    prop_assert!(cell.is_some_and(|c| c.days >= 2), "{r:?}");
                    if let (Some(c), Some(cut)) = (cell, cutoff) {
                        let within = if busy {
                            c.avg_percentage >= cut
                        } else {
                            c.avg_percentage <= cut
                        };
                        prop_assert!(within, "{r:?}");
                    }
                }
            }
            for pair in ranges.windows(2) {
                let ordered = if busy {
                    pair[0].average >= pair[1].average
                } else {
                    pair[0].average <= pair[1].average
                };
                prop_assert!(ordered, "{pair:?}");
            }
            for (i, a) in ranges.iter().enumerate() {
                for b in &ranges[i + 1..] {
                    // Disjoint, and never touching on the same day (they
                    // would have been joined).
                    prop_assert!(
                        a.weekday != b.weekday
                            || a.end_hour < b.start_hour
                            || b.end_hour < a.start_hour
                    );
                }
            }
        }
    }
}
