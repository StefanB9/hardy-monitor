//! The time ranges the views offer and the instants they cover.

use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Utc};
use hardy_core::{Tz, analytics::midnight_local_as_utc, schedule::GymSchedule};

/// Range of the occupancy chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChartRange {
    /// Today's opening hours.
    #[default]
    Today,
    Days7,
    Days30,
    /// The dates entered by the user (end inclusive).
    Custom,
}

/// Range of the weekly heatmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnalyticsRange {
    ThisWeek,
    Last2Weeks,
    #[default]
    Last4Weeks,
    Last8Weeks,
}

impl ChartRange {
    /// `[start, end)` shown for this range; `None` for invalid custom dates.
    pub fn window(
        self,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
        custom: (&str, &str),
    ) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let tz = schedule.timezone();
        let today = now.with_timezone(&tz).date_naive();
        let days_ending_today = |days: u64| {
            let first = today.checked_sub_days(chrono::Days::new(days - 1))?;
            Some((
                midnight_local_as_utc(first, tz),
                midnight_local_as_utc(today.succ_opt()?, tz),
            ))
        };
        match self {
            ChartRange::Today => {
                let open = schedule.opening_time_on(today);
                Some((open, schedule.next_closing_after(open)))
            }
            ChartRange::Days7 => days_ending_today(7),
            ChartRange::Days30 => days_ending_today(30),
            ChartRange::Custom => {
                let start = parse_date(custom.0)?;
                let end = parse_date(custom.1)?;
                (start <= end).then(|| {
                    (
                        midnight_local_as_utc(start, tz),
                        midnight_local_as_utc(end.succ_opt().unwrap_or(end), tz),
                    )
                })
            }
        }
    }
}

impl AnalyticsRange {
    /// Start of the range: gym-local midnight of a Monday.
    pub fn start(self, now: DateTime<Utc>, tz: Tz) -> DateTime<Utc> {
        let weeks_back = match self {
            AnalyticsRange::ThisWeek => 0,
            AnalyticsRange::Last2Weeks => 1,
            AnalyticsRange::Last4Weeks => 3,
            AnalyticsRange::Last8Weeks => 7,
        };
        gym_week_start(now, tz) - TimeDelta::weeks(weeks_back)
    }
}

/// Parses `YYYY-MM-DD`.
pub fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
}

/// Gym-local midnight of the Monday starting the week that contains `now`.
pub fn gym_week_start(now: DateTime<Utc>, tz: Tz) -> DateTime<Utc> {
    let today = now.with_timezone(&tz).date_naive();
    let days_since_monday = i64::from(today.weekday().num_days_from_monday());
    midnight_local_as_utc(today - TimeDelta::days(days_since_monday), tz)
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    fn schedule() -> GymSchedule {
        GymSchedule::default()
    }

    /// 2024-06-`d` `h`:00 gym-local (17th is a Monday).
    fn local(d: u32, h: u32) -> Result<DateTime<Utc>> {
        Ok(schedule()
            .timezone()
            .with_ymd_and_hms(2024, 6, d, h, 0, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    #[test]
    fn test_chart_today_covers_opening_hours() -> Result<()> {
        let window = ChartRange::Today
            .window(local(17, 15)?, &schedule(), ("", ""))
            .context("window")?;
        assert_eq!(window, (local(17, 6)?, local(17, 23)?));
        // Before opening it still shows today.
        let early = ChartRange::Today
            .window(local(18, 3)?, &schedule(), ("", ""))
            .context("window")?;
        assert_eq!(early, (local(18, 6)?, local(18, 23)?));
        Ok(())
    }

    #[test]
    fn test_chart_days_end_at_tomorrow_midnight() -> Result<()> {
        let window = ChartRange::Days7
            .window(local(17, 15)?, &schedule(), ("", ""))
            .context("window")?;
        assert_eq!(window, (local(11, 0)?, local(18, 0)?));
        Ok(())
    }

    #[test]
    fn test_chart_custom_is_end_inclusive_and_validated() -> Result<()> {
        let s = schedule();
        let now = local(17, 15)?;
        let window = ChartRange::Custom
            .window(now, &s, ("2024-06-10", "2024-06-12"))
            .context("window")?;
        assert_eq!(window, (local(10, 0)?, local(13, 0)?));
        assert!(
            ChartRange::Custom
                .window(now, &s, ("2024-06-12", "2024-06-10"))
                .is_none()
        );
        assert!(
            ChartRange::Custom
                .window(now, &s, ("June", "2024-06-10"))
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn test_analytics_range_starts_on_monday() -> Result<()> {
        let tz = schedule().timezone();
        let now = local(19, 12)?;
        assert_eq!(AnalyticsRange::ThisWeek.start(now, tz), local(17, 0)?);
        assert_eq!(
            AnalyticsRange::Last4Weeks.start(now, tz),
            local(17, 0)? - TimeDelta::weeks(3)
        );
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn chart_window_contains_now_for_presets(minute in 0i64..(366 * 24 * 60)) {
            let s = schedule();
            let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).single().unwrap_or_default()
                + TimeDelta::minutes(minute);
            for range in [ChartRange::Days7, ChartRange::Days30] {
                let window = range.window(now, &s, ("", ""));
                prop_assert!(window.is_some_and(|(a, b)| a <= now && now < b));
            }
            let today = ChartRange::Today.window(now, &s, ("", ""));
            prop_assert!(today.is_some_and(|(a, b)| a < b && b - a <= TimeDelta::hours(19)));
        }
    }
}
