//! Axis scaling and tick placement for the occupancy chart.

use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc, Weekday};
use hardy_core::{Tz, analytics::midnight_local_as_utc};

/// Upper end of the value axis: the data maximum plus headroom, rounded up to
/// a multiple of 20, at least 40 and at most 100. Occupancy rarely reaches
/// 100%, so a fixed 0–100 axis wastes most of the chart.
pub(super) fn value_axis_max(values: impl IntoIterator<Item = f64>) -> f64 {
    let max = values
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    (((max + 5.0) / 20.0).ceil() * 20.0).clamp(40.0, 100.0)
}

/// Labelled time ticks in the gym's wall clock: every two hours for ranges
/// up to a day, every midnight up to ~10 days, every Monday beyond.
pub(super) fn time_ticks(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    tz: Tz,
) -> Vec<(DateTime<Utc>, String)> {
    let span = end - start;
    if span <= TimeDelta::zero() {
        return Vec::new();
    }
    let mut ticks = Vec::new();
    if span <= TimeDelta::hours(26) {
        let local = start.with_timezone(&tz);
        let mut t = midnight_local_as_utc(local.date_naive(), tz)
            + TimeDelta::hours(i64::from(local.hour()));
        while t < end {
            let hour = t.with_timezone(&tz).hour();
            if t >= start && hour.is_multiple_of(2) {
                ticks.push((t, t.with_timezone(&tz).format("%H:%M").to_string()));
            }
            t += TimeDelta::hours(1);
        }
    } else {
        let weekly = span > TimeDelta::days(10);
        let mut day = start.with_timezone(&tz).date_naive();
        loop {
            let t = midnight_local_as_utc(day, tz);
            if t >= end {
                break;
            }
            if t >= start && (!weekly || day.weekday() == Weekday::Mon) {
                let label = if weekly {
                    day.format("%d %b").to_string()
                } else {
                    day.format("%a %d").to_string()
                };
                ticks.push((t, label));
            }
            let Some(next) = day.succ_opt() else { break };
            day = next;
        }
    }
    ticks
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    fn tz() -> Tz {
        hardy_core::GymSchedule::default().timezone()
    }

    fn local(d: u32, h: u32) -> Result<DateTime<Utc>> {
        Ok(tz()
            .with_ymd_and_hms(2024, 6, d, h, 0, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    #[test]
    fn test_value_axis_max_adds_headroom_and_rounds() {
        assert_relative_eq!(value_axis_max([]), 40.0);
        assert_relative_eq!(value_axis_max([12.0, 31.0]), 40.0);
        assert_relative_eq!(value_axis_max([36.0]), 60.0);
        assert_relative_eq!(value_axis_max([58.0, f64::NAN]), 80.0);
        assert_relative_eq!(value_axis_max([140.0]), 100.0);
    }

    #[test]
    fn test_time_ticks_today_every_two_hours() -> Result<()> {
        // Monday 07:00–23:00 gym-local.
        let ticks = time_ticks(local(17, 7)?, local(17, 23)?, tz());
        let labels: Vec<_> = ticks.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(
            labels,
            [
                "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"
            ]
        );
        assert_eq!(ticks[0].0, local(17, 8)?);
        Ok(())
    }

    #[test]
    fn test_time_ticks_week_at_local_midnights() -> Result<()> {
        let ticks = time_ticks(local(11, 0)?, local(18, 0)?, tz());
        assert_eq!(ticks.len(), 7);
        assert_eq!(ticks[0], (local(11, 0)?, "Tue 11".to_string()));
        Ok(())
    }

    #[test]
    fn test_time_ticks_month_on_mondays() -> Result<()> {
        let ticks = time_ticks(local(1, 0)?, local(30, 0)?, tz());
        let labels: Vec<_> = ticks.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(labels, ["03 Jun", "10 Jun", "17 Jun", "24 Jun"]);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn time_ticks_are_sorted_and_in_range(
            start_min in 0i64..(60 * 24 * 60),
            len_min in 1i64..(40 * 24 * 60),
        ) {
            let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).single()
                .unwrap_or_default() + TimeDelta::minutes(start_min);
            let end = start + TimeDelta::minutes(len_min);
            let ticks = time_ticks(start, end, tz());
            prop_assert!(ticks.len() <= 16);
            for pair in ticks.windows(2) {
                prop_assert!(pair[0].0 < pair[1].0);
            }
            for (t, _) in &ticks {
                prop_assert!(*t >= start && *t < end);
            }
        }

        #[test]
        fn value_axis_covers_data(values in proptest::collection::vec(0.0f64..100.0, 0..50)) {
            let max = value_axis_max(values.iter().copied());
            prop_assert!((40.0..=100.0).contains(&max));
            prop_assert!(values.iter().all(|v| *v <= max));
        }
    }
}
