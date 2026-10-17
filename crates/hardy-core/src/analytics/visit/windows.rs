//! The quietest windows of a day: overall and per part of the day.

use chrono::{DateTime, NaiveTime, TimeDelta, Timelike, Utc};
use chrono_tz::Tz;

use super::{EstimateSource, Expected, QUARTER, SlotEstimate, VisitWindow};

/// Part of the day a window starts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DayPart {
    /// Before 11:00.
    Morning,
    /// 11:00–14:00.
    Midday,
    /// 14:00–17:00.
    Afternoon,
    /// From 17:00.
    Evening,
}

impl DayPart {
    /// Every part, in the order of the day.
    pub const ALL: [Self; 4] = [Self::Morning, Self::Midday, Self::Afternoon, Self::Evening];

    /// The part a gym-local `hour` belongs to.
    pub fn of_hour(hour: u32) -> Self {
        match hour {
            0..11 => Self::Morning,
            11..14 => Self::Midday,
            14..17 => Self::Afternoon,
            _ => Self::Evening,
        }
    }
}

/// What a suggested visit must fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisitConstraints {
    pub length: TimeDelta,
    /// Earliest start, as time after gym-local midnight.
    pub earliest: TimeDelta,
    /// Latest end, as time after gym-local midnight (24 h: end of day).
    pub latest: TimeDelta,
    /// No window starts before this (now, for today).
    pub not_before: DateTime<Utc>,
}

/// The day's quietest window and the quietest per part of the day.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisitPlan {
    pub best: Option<VisitWindow>,
    /// In the order of the day; parts without a possible window are left
    /// out.
    pub by_part: Vec<(DayPart, VisitWindow)>,
}

/// Windows of `constraints.length` made of consecutive, known quarter hours
/// of `estimates` (one day, in order), within the time range in `tz`;
/// quietest by mean, the earliest of equally quiet ones.
pub fn plan_day(estimates: &[SlotEstimate], constraints: &VisitConstraints, tz: Tz) -> VisitPlan {
    let quarters =
        usize::try_from(constraints.length.num_minutes() / QUARTER.num_minutes()).unwrap_or(0);
    if quarters == 0 {
        return VisitPlan::default();
    }
    let mut best: Option<VisitWindow> = None;
    let mut by_part: [Option<VisitWindow>; DayPart::ALL.len()] = [None; DayPart::ALL.len()];
    for stretch in estimates.windows(quarters) {
        let Some(window) = window_of(stretch, constraints, tz) else {
            continue;
        };
        let part = DayPart::of_hour(window.start.with_timezone(&tz).hour());
        // Strictly quieter only: the earliest of equal windows stays.
        let quieter = |current: &Option<VisitWindow>| {
            current.is_none_or(|c| window.expected.mean < c.expected.mean)
        };
        if quieter(&best) {
            best = Some(window);
        }
        let slot = &mut by_part[part as usize];
        if quieter(slot) {
            *slot = Some(window);
        }
    }
    VisitPlan {
        best,
        by_part: DayPart::ALL
            .into_iter()
            .zip(by_part)
            .filter_map(|(part, window)| window.map(|w| (part, w)))
            .collect(),
    }
}

/// `stretch` as a window, if it is consecutive, fully known and fits the
/// constraints.
fn window_of(
    stretch: &[SlotEstimate],
    constraints: &VisitConstraints,
    tz: Tz,
) -> Option<VisitWindow> {
    let first = stretch.first()?;
    let consecutive = stretch
        .windows(2)
        .all(|pair| pair[1].start - pair[0].start == QUARTER);
    let start = first.start;
    let since_midnight = start
        .with_timezone(&tz)
        .time()
        .signed_duration_since(NaiveTime::MIN);
    if !consecutive
        || start < constraints.not_before
        || since_midnight < constraints.earliest
        || since_midnight + constraints.length > constraints.latest
    {
        return None;
    }
    let (mut mean, mut low, mut high) = (0.0, 0.0, 0.0);
    let mut forecast = false;
    for estimate in stretch {
        let expected = estimate.expected?;
        mean += expected.mean;
        low += expected.low;
        high += expected.high;
        forecast |= estimate.source == EstimateSource::Forecast;
    }
    #[allow(clippy::cast_precision_loss)] // a day has at most 96 quarter hours
    let n = stretch.len() as f64;
    Some(VisitWindow {
        start,
        end: start + constraints.length,
        expected: Expected {
            mean: mean / n,
            low: low / n,
            high: high / n,
        },
        source: if forecast {
            EstimateSource::Forecast
        } else {
            first.source
        },
    })
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use chrono_tz::Europe::Berlin;
    use proptest::prelude::*;

    use super::*;

    fn berlin(h: u32, mi: u32) -> DateTime<Utc> {
        Berlin
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map_or(DateTime::<Utc>::MIN_UTC, |t| t.with_timezone(&Utc))
    }

    /// A day from 06:00 with one estimate per quarter hour (`None`: unknown).
    fn day(means: &[Option<f64>]) -> Vec<SlotEstimate> {
        means
            .iter()
            .enumerate()
            .map(|(i, m)| SlotEstimate {
                start: berlin(6, 0) + QUARTER * i32::try_from(i).unwrap_or(0),
                expected: m.map(|mean| Expected {
                    mean,
                    low: mean - 1.0,
                    high: mean + 1.0,
                }),
                source: EstimateSource::RecentWeeks,
            })
            .collect()
    }

    fn hours(h: i64) -> TimeDelta {
        TimeDelta::hours(h)
    }

    fn whole_day(length: TimeDelta) -> VisitConstraints {
        VisitConstraints {
            length,
            earliest: TimeDelta::zero(),
            latest: hours(24),
            not_before: DateTime::<Utc>::MIN_UTC,
        }
    }

    /// 06:00–23:00 at 50%, with `quiet` quarter hours (from 06:00) at 10%.
    fn with_quiet(quiet: std::ops::Range<usize>) -> Vec<SlotEstimate> {
        let means: Vec<_> = (0..68)
            .map(|i| Some(if quiet.contains(&i) { 10.0 } else { 50.0 }))
            .collect();
        day(&means)
    }

    #[test]
    fn test_best_window_is_the_quietest_stretch() -> Result<()> {
        // 14:00–15:30 is quiet (quarter hours 32..38 from 06:00).
        let plan = plan_day(
            &with_quiet(32..38),
            &whole_day(TimeDelta::minutes(90)),
            Berlin,
        );
        let best = plan.best.context("a window")?;
        assert_eq!((best.start, best.end), (berlin(14, 0), berlin(15, 30)));
        assert_relative_eq!(best.expected.mean, 10.0);
        assert_relative_eq!(best.expected.low, 9.0);
        assert_relative_eq!(best.expected.high, 11.0);
        Ok(())
    }

    #[test]
    fn test_window_stays_in_the_time_range() -> Result<()> {
        let constraints = VisitConstraints {
            earliest: hours(17),
            latest: TimeDelta::minutes(20 * 60 + 30),
            ..whole_day(hours(1))
        };
        // Quiet in the morning and at 20:00–21:00; 20:30 is the latest end.
        let mut estimates = with_quiet(0..8);
        for e in estimates.iter_mut().skip(56).take(4) {
            e.expected = Some(Expected {
                mean: 20.0,
                low: 20.0,
                high: 20.0,
            });
        }
        let best = plan_day(&estimates, &constraints, Berlin)
            .best
            .context("a window")?;
        assert!(best.start >= berlin(17, 0));
        assert!(best.end <= berlin(20, 30));
        assert_eq!(best.start, berlin(19, 30));
        Ok(())
    }

    #[test]
    fn test_window_skips_unknown_quarter_hours() -> Result<()> {
        let mut means: Vec<_> = (0..12).map(|_| Some(10.0)).collect();
        means[1] = None;
        let best = plan_day(&day(&means), &whole_day(hours(1)), Berlin)
            .best
            .context("a window")?;
        // 06:15 is unknown, so the first full hour without it starts at
        // 06:30.
        assert_eq!(best.start, berlin(6, 30));
        Ok(())
    }

    #[test]
    fn test_window_does_not_start_in_the_past() -> Result<()> {
        let constraints = VisitConstraints {
            not_before: berlin(8, 5),
            ..whole_day(hours(1))
        };
        let best = plan_day(&with_quiet(0..12), &constraints, Berlin)
            .best
            .context("a window")?;
        assert_eq!(best.start, berlin(8, 15));
        Ok(())
    }

    #[test]
    fn test_equal_windows_prefer_the_earliest() -> Result<()> {
        let best = plan_day(&with_quiet(0..0), &whole_day(hours(1)), Berlin)
            .best
            .context("a window")?;
        assert_eq!(best.start, berlin(6, 0));
        Ok(())
    }

    #[test]
    fn test_no_window_longer_than_the_day() {
        let plan = plan_day(&day(&[Some(5.0); 6]), &whole_day(hours(2)), Berlin);
        assert_eq!(plan, VisitPlan::default());
    }

    #[test]
    fn test_window_with_a_forecast_quarter_hour_is_forecast() -> Result<()> {
        let mut estimates = with_quiet(0..4);
        estimates[2].source = EstimateSource::Forecast;
        let best = plan_day(&estimates, &whole_day(hours(1)), Berlin)
            .best
            .context("a window")?;
        assert_eq!(best.source, EstimateSource::Forecast);
        Ok(())
    }

    #[test]
    fn test_day_parts_by_start_hour() {
        assert_eq!(DayPart::of_hour(6), DayPart::Morning);
        assert_eq!(DayPart::of_hour(10), DayPart::Morning);
        assert_eq!(DayPart::of_hour(11), DayPart::Midday);
        assert_eq!(DayPart::of_hour(13), DayPart::Midday);
        assert_eq!(DayPart::of_hour(14), DayPart::Afternoon);
        assert_eq!(DayPart::of_hour(16), DayPart::Afternoon);
        assert_eq!(DayPart::of_hour(17), DayPart::Evening);
        assert_eq!(DayPart::of_hour(23), DayPart::Evening);
    }

    #[test]
    fn test_best_window_per_part_of_the_day() {
        // Quiet 07:00–08:00 (morning) and 18:00–19:00 (evening, quieter).
        let mut estimates = with_quiet(4..8);
        for e in estimates.iter_mut().skip(48).take(4) {
            e.expected = Some(Expected {
                mean: 5.0,
                low: 5.0,
                high: 5.0,
            });
        }
        let plan = plan_day(&estimates, &whole_day(hours(1)), Berlin);
        let parts: Vec<_> = plan
            .by_part
            .iter()
            .map(|(p, w)| (*p, w.start.with_timezone(&Berlin).hour()))
            .collect();
        assert_eq!(parts[0], (DayPart::Morning, 7));
        assert_eq!(parts[3], (DayPart::Evening, 18));
        assert_eq!(plan.by_part.len(), 4);
        assert_eq!(plan.best.map(|w| w.start), Some(berlin(18, 0)));
    }

    #[test]
    fn test_part_without_a_window_is_left_out() {
        let constraints = VisitConstraints {
            earliest: hours(17),
            ..whole_day(hours(1))
        };
        let plan = plan_day(&with_quiet(0..0), &constraints, Berlin);
        let parts: Vec<_> = plan.by_part.iter().map(|(p, _)| *p).collect();
        assert_eq!(parts, vec![DayPart::Evening]);
    }

    /// Every valid window's start and mean, by brute force.
    fn valid_windows(
        estimates: &[SlotEstimate],
        constraints: &VisitConstraints,
    ) -> Vec<(DateTime<Utc>, f64)> {
        let len = usize::try_from(constraints.length.num_minutes() / 15).unwrap_or(0);
        let midnight = berlin(0, 0);
        estimates
            .windows(len)
            .filter(|w| w.iter().all(|e| e.expected.is_some()))
            .filter(|w| {
                let start = w[0].start;
                let end = start + constraints.length;
                start >= constraints.not_before
                    && start - midnight >= constraints.earliest
                    && end - midnight <= constraints.latest
            })
            .map(|w| {
                let sum: f64 = w.iter().filter_map(|e| e.expected).map(|x| x.mean).sum();
                #[allow(clippy::cast_precision_loss)]
                (w[0].start, sum / len as f64)
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_best_window_is_valid_and_quietest(
            means in prop::collection::vec(prop::option::weighted(0.9, 0.0f64..100.0), 0..68),
            quarters in 4i64..9,
            earliest in 0i64..96,
            span in 0i64..96,
            past in 0i64..40,
        ) {
            let estimates = day(&means);
            let constraints = VisitConstraints {
                length: QUARTER * i32::try_from(quarters).unwrap_or(4),
                earliest: TimeDelta::minutes(earliest * 15),
                latest: TimeDelta::minutes(((earliest + span) * 15).min(24 * 60)),
                not_before: berlin(6, 0) + TimeDelta::minutes(past * 15),
            };
            let plan = plan_day(&estimates, &constraints, Berlin);
            let valid = valid_windows(&estimates, &constraints);
            match plan.best {
                None => prop_assert!(valid.is_empty()),
                Some(best) => {
                    prop_assert_eq!(best.end - best.start, constraints.length);
                    prop_assert!(valid.iter().any(|(s, _)| *s == best.start));
                    for (_, mean) in &valid {
                        prop_assert!(best.expected.mean <= mean + 1e-9);
                    }
                }
            }
            for (part, window) in &plan.by_part {
                prop_assert!(valid.iter().any(|(s, _)| *s == window.start));
                prop_assert_eq!(
                    DayPart::of_hour(window.start.with_timezone(&Berlin).hour()),
                    *part
                );
            }
        }
    }
}
