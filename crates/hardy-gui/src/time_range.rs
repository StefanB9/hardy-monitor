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
        offset: u32,
    ) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let tz = schedule.timezone();
        let today = now.with_timezone(&tz).date_naive();
        // `offset` whole periods back; a period is the range's own length.
        let back = |days: u64| today.checked_sub_days(chrono::Days::new(days * u64::from(offset)));
        let days_ending = |last: NaiveDate, days: u64| {
            let first = last.checked_sub_days(chrono::Days::new(days - 1))?;
            Some((
                midnight_local_as_utc(first, tz),
                midnight_local_as_utc(last.succ_opt()?, tz),
            ))
        };
        match self {
            ChartRange::Today => {
                let open = schedule.opening_time_on(back(1)?);
                Some((open, schedule.next_closing_after(open)))
            }
            ChartRange::Days7 => days_ending(back(7)?, 7),
            ChartRange::Days30 => days_ending(back(30)?, 30),
            // Steps move the custom dates themselves (see `shift_custom`).
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

    /// Whether a later period exists: presets stop at the current one, a
    /// custom range once its last day is today.
    pub fn can_step_forward(
        self,
        offset: u32,
        window_end: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> bool {
        match self {
            ChartRange::Custom => window_end <= now,
            _ => offset > 0,
        }
    }

    /// The shown period in words: "Today", "Fri, 9 Oct" or "3 – 9 Oct".
    pub fn period_label(
        self,
        offset: u32,
        window: (DateTime<Utc>, DateTime<Utc>),
        tz: Tz,
    ) -> String {
        if self == ChartRange::Today && offset == 0 {
            return "Today".to_string();
        }
        let first = window.0.with_timezone(&tz).date_naive();
        // Multi-day windows end at the next midnight, exclusive.
        let last = (window.1 - TimeDelta::seconds(1))
            .with_timezone(&tz)
            .date_naive();
        if first >= last {
            first.format("%a, %-d %b").to_string()
        } else if first.month() == last.month() {
            format!("{} – {}", first.format("%-d"), last.format("%-d %b"))
        } else {
            format!("{} – {}", first.format("%-d %b"), last.format("%-d %b"))
        }
    }
}

/// Direction of a chart step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Back,
    Forward,
}

/// Custom dates moved one range length back or forward; `None` if they are
/// not a valid range.
pub fn shift_custom(start: &str, end: &str, step: Step) -> Option<(NaiveDate, NaiveDate)> {
    let (start, end) = (parse_date(start)?, parse_date(end)?);
    if start > end {
        return None;
    }
    let days = chrono::Days::new(u64::try_from((end - start).num_days() + 1).ok()?);
    match step {
        Step::Back => Some((start.checked_sub_days(days)?, end.checked_sub_days(days)?)),
        Step::Forward => Some((start.checked_add_days(days)?, end.checked_add_days(days)?)),
    }
}

/// Whether readings exist before the window, i.e. stepping back shows data.
pub fn can_step_back(window_start: DateTime<Utc>, first_reading: Option<DateTime<Utc>>) -> bool {
    first_reading.is_some_and(|first| first < window_start)
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
            .window(local(17, 15)?, &schedule(), ("", ""), 0)
            .context("window")?;
        assert_eq!(window, (local(17, 6)?, local(17, 23)?));
        // Before opening it still shows today.
        let early = ChartRange::Today
            .window(local(18, 3)?, &schedule(), ("", ""), 0)
            .context("window")?;
        assert_eq!(early, (local(18, 6)?, local(18, 23)?));
        Ok(())
    }

    #[test]
    fn test_chart_days_end_at_tomorrow_midnight() -> Result<()> {
        let window = ChartRange::Days7
            .window(local(17, 15)?, &schedule(), ("", ""), 0)
            .context("window")?;
        assert_eq!(window, (local(11, 0)?, local(18, 0)?));
        Ok(())
    }

    #[test]
    fn test_chart_custom_is_end_inclusive_and_validated() -> Result<()> {
        let s = schedule();
        let now = local(17, 15)?;
        let window = ChartRange::Custom
            .window(now, &s, ("2024-06-10", "2024-06-12"), 0)
            .context("window")?;
        assert_eq!(window, (local(10, 0)?, local(13, 0)?));
        assert!(
            ChartRange::Custom
                .window(now, &s, ("2024-06-12", "2024-06-10"), 0)
                .is_none()
        );
        assert!(
            ChartRange::Custom
                .window(now, &s, ("June", "2024-06-10"), 0)
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

    /// `y-m-d h:00` gym-local.
    fn local_on(y: i32, m: u32, d: u32, h: u32) -> Result<DateTime<Utc>> {
        Ok(schedule()
            .timezone()
            .with_ymd_and_hms(y, m, d, h, 0, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    #[test]
    fn test_chart_today_steps_back_a_day_at_a_time() -> Result<()> {
        let s = schedule();
        let now = local(17, 15)?;
        // Sunday: weekend hours.
        let sunday = ChartRange::Today
            .window(now, &s, ("", ""), 1)
            .context("window")?;
        assert_eq!(sunday, (local(16, 9)?, local(16, 21)?));
        let friday = ChartRange::Today
            .window(now, &s, ("", ""), 3)
            .context("window")?;
        assert_eq!(friday, (local(14, 6)?, local(14, 23)?));
        Ok(())
    }

    #[test]
    fn test_chart_day_steps_cross_daylight_saving_changes() -> Result<()> {
        let s = schedule();
        // Monday after the spring change; one step back is the 23-hour Sunday.
        let spring = ChartRange::Today
            .window(local_on(2024, 4, 1, 12)?, &s, ("", ""), 1)
            .context("window")?;
        assert_eq!(
            spring,
            (local_on(2024, 3, 31, 9)?, local_on(2024, 3, 31, 21)?)
        );
        // Monday after the autumn change; one step back is the 25-hour Sunday.
        let autumn = ChartRange::Today
            .window(local_on(2024, 10, 28, 12)?, &s, ("", ""), 1)
            .context("window")?;
        assert_eq!(
            autumn,
            (local_on(2024, 10, 27, 9)?, local_on(2024, 10, 27, 21)?)
        );
        Ok(())
    }

    #[test]
    fn test_chart_multi_day_ranges_step_by_their_length() -> Result<()> {
        let s = schedule();
        let now = local(17, 15)?;
        let week = ChartRange::Days7
            .window(now, &s, ("", ""), 1)
            .context("window")?;
        assert_eq!(week, (local(4, 0)?, local(11, 0)?));
        let month = ChartRange::Days30
            .window(now, &s, ("", ""), 1)
            .context("window")?;
        assert_eq!(
            month,
            (local_on(2024, 4, 19, 0)?, local_on(2024, 5, 19, 0)?)
        );
        // Custom moves through its date inputs instead.
        let custom = ChartRange::Custom
            .window(now, &s, ("2024-06-10", "2024-06-12"), 4)
            .context("window")?;
        assert_eq!(custom, (local(10, 0)?, local(13, 0)?));
        Ok(())
    }

    #[test]
    fn test_shift_custom_moves_by_its_length() -> Result<()> {
        let date = |s: &str| parse_date(s).context("date");
        assert_eq!(
            shift_custom("2024-06-10", "2024-06-12", Step::Back),
            Some((date("2024-06-07")?, date("2024-06-09")?))
        );
        assert_eq!(
            shift_custom("2024-06-10", "2024-06-12", Step::Forward),
            Some((date("2024-06-13")?, date("2024-06-15")?))
        );
        assert_eq!(shift_custom("2024-06-12", "2024-06-10", Step::Back), None);
        assert_eq!(shift_custom("not a date", "2024-06-10", Step::Back), None);
        Ok(())
    }

    #[test]
    fn test_step_back_stops_at_the_first_reading() -> Result<()> {
        let window_start = local(16, 9)?;
        assert!(can_step_back(window_start, Some(local(10, 12)?)));
        assert!(!can_step_back(window_start, Some(local(16, 10)?)));
        assert!(!can_step_back(window_start, None));
        Ok(())
    }

    #[test]
    fn test_step_forward_stops_at_the_current_period() -> Result<()> {
        let now = local(17, 23)?;
        // After closing, today's window has ended, but tomorrow is not shown.
        assert!(!ChartRange::Today.can_step_forward(0, local(17, 23)?, now));
        assert!(ChartRange::Today.can_step_forward(1, local(16, 21)?, now));
        // Custom: forward while its last day is before today.
        assert!(ChartRange::Custom.can_step_forward(0, local(17, 0)?, now));
        assert!(!ChartRange::Custom.can_step_forward(0, local(18, 0)?, now));
        Ok(())
    }

    #[test]
    fn test_period_label_names_the_shown_days() -> Result<()> {
        let tz = schedule().timezone();
        let label = |range: ChartRange, offset: u32, window| range.period_label(offset, window, tz);
        assert_eq!(
            label(ChartRange::Today, 0, (local(17, 6)?, local(17, 23)?)),
            "Today"
        );
        assert_eq!(
            label(ChartRange::Today, 1, (local(16, 9)?, local(16, 21)?)),
            "Sun, 16 Jun"
        );
        assert_eq!(
            label(ChartRange::Days7, 0, (local(11, 0)?, local(18, 0)?)),
            "11 – 17 Jun"
        );
        assert_eq!(
            label(
                ChartRange::Days30,
                0,
                (local_on(2024, 5, 19, 0)?, local(18, 0)?)
            ),
            "19 May – 17 Jun"
        );
        assert_eq!(
            label(ChartRange::Custom, 0, (local(12, 0)?, local(13, 0)?)),
            "Wed, 12 Jun"
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
                let window = range.window(now, &s, ("", ""), 0);
                prop_assert!(window.is_some_and(|(a, b)| a <= now && now < b));
            }
            let today = ChartRange::Today.window(now, &s, ("", ""), 0);
            prop_assert!(today.is_some_and(|(a, b)| a < b && b - a <= TimeDelta::hours(19)));
        }
    }
}
