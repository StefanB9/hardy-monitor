use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;

use crate::config::ScheduleConfig;

/// Opening hours of the gym, evaluated in the gym's own timezone.
#[derive(Debug, Clone)]
pub struct GymSchedule {
    timezone: Tz,
    weekday_open: u32,
    weekday_close: u32,
    weekend_open: u32,
    weekend_close: u32,
}

impl GymSchedule {
    pub fn new(config: &ScheduleConfig) -> Self {
        Self {
            timezone: config.timezone,
            weekday_open: config.weekday.open_hour,
            weekday_close: config.weekday.close_hour,
            weekend_open: config.weekend.open_hour,
            weekend_close: config.weekend.close_hour,
        }
    }

    /// Timezone in which opening hours, holidays and local aggregates are
    /// defined.
    pub fn timezone(&self) -> Tz {
        self.timezone
    }

    /// Whether the gym is open at the given instant, judged by the wall clock
    /// in the gym's timezone.
    pub fn is_open(&self, time: &DateTime<Utc>) -> bool {
        let time = time.with_timezone(&self.timezone);
        let date = time.date_naive();
        let hour = time.hour();
        let minute = time.minute();

        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            (self.weekend_open..self.weekend_close).contains(&hour)
                || (hour == self.weekend_close && minute == 0)
        } else {
            (self.weekday_open..self.weekday_close).contains(&hour)
                || (hour == self.weekday_close && minute == 0)
        }
    }
}

impl Default for GymSchedule {
    fn default() -> Self {
        Self::new(&ScheduleConfig::default())
    }
}

impl GymSchedule {
    #[cfg(test)]
    pub fn new_for_test(
        weekday_open: u32,
        weekday_close: u32,
        weekend_open: u32,
        weekend_close: u32,
    ) -> Self {
        Self {
            timezone: chrono_tz::Europe::Berlin,
            weekday_open,
            weekday_close,
            weekend_open,
            weekend_close,
        }
    }

    pub fn get_open_hour(&self, date: NaiveDate) -> u32 {
        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            self.weekend_open
        } else {
            self.weekday_open
        }
    }

    pub fn get_close_hour(&self, date: NaiveDate) -> u32 {
        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            self.weekend_close
        } else {
            self.weekday_close
        }
    }

    /// The instant the gym opens on `date` (a gym-local calendar day).
    pub fn opening_time_on(&self, date: NaiveDate) -> DateTime<Utc> {
        self.local_hour_on(date, self.get_open_hour(date))
    }

    /// The first closing instant strictly after `now`: today's closing time,
    /// or tomorrow's once today's has passed.
    pub fn next_closing_after(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        let today = now.with_timezone(&self.timezone).date_naive();
        let close_today = self.local_hour_on(today, self.get_close_hour(today));
        if close_today > now {
            return close_today;
        }
        let tomorrow = today.succ_opt().unwrap_or(today);
        self.local_hour_on(tomorrow, self.get_close_hour(tomorrow))
    }

    /// `hour:00` gym-local on `date` as UTC; hour 24 means midnight after
    /// `date`.
    fn local_hour_on(&self, date: NaiveDate, hour: u32) -> DateTime<Utc> {
        let (date, hour) = if hour >= 24 {
            (date.succ_opt().unwrap_or(date), 0)
        } else {
            (date, hour)
        };
        let naive = date.and_time(NaiveTime::from_hms_opt(hour, 0, 0).unwrap_or(NaiveTime::MIN));
        // SAFETY: opening hours are whole hours outside the 02:00–03:00 DST
        // transition, so `earliest` only differs from `single` in a
        // misconfigured schedule; falling back to UTC keeps the result total.
        self.timezone
            .from_local_datetime(&naive)
            .earliest()
            .map_or_else(|| naive.and_utc(), |t| t.with_timezone(&Utc))
    }
}

pub fn is_bavarian_holiday(date: NaiveDate) -> bool {
    let (d, m) = (date.day(), date.month());
    let year = date.year();

    match (m, d) {
        (1 | 5 | 11, 1) | (1, 6) | (8, 15) | (10, 3) | (12, 25 | 26) => return true,
        _ => {}
    }

    if let Some(easter) = easter_date_cached(year) {
        let ordinal = date.ordinal();
        let easter_ordinal = easter.ordinal();

        if ordinal == easter_ordinal - 2 {
            return true;
        }
        if ordinal == easter_ordinal + 1 {
            return true;
        }
        if ordinal == easter_ordinal + 39 {
            return true;
        }
        if ordinal == easter_ordinal + 50 {
            return true;
        }
        if ordinal == easter_ordinal + 60 {
            return true;
        }
    }

    false
}

/// Thread-safe memoized wrapper around `easter_date`.
///
/// Easter computation is pure and deterministic for a given year. During ML
/// training (~1000 calls with the same year), this avoids redundant
/// recomputation.
fn easter_date_cached(year: i32) -> Option<NaiveDate> {
    static CACHE: LazyLock<Mutex<HashMap<i32, Option<NaiveDate>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard.entry(year).or_insert_with(|| easter_date(year))
}

#[allow(clippy::many_single_char_names)]
fn easter_date(year: i32) -> Option<NaiveDate> {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;

    NaiveDate::from_ymd_opt(year, month.cast_unsigned(), day.cast_unsigned())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{NaiveDate, TimeZone, Utc};
    use chrono_tz::{Europe::Berlin, Tz};

    use super::*;

    #[test]
    fn test_easter_2024() -> Result<()> {
        let easter = easter_date(2024).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2024, 3, 31).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2025() -> Result<()> {
        let easter = easter_date(2025).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2025, 4, 20).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2026() -> Result<()> {
        let easter = easter_date(2026).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2026, 4, 5).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_historical_1999() -> Result<()> {
        let easter = easter_date(1999).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(1999, 4, 4).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_early_march() -> Result<()> {
        let easter = easter_date(2008).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2008, 3, 23).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_late_april() -> Result<()> {
        let easter = easter_date(2038).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2038, 4, 25).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_fixed_holidays() -> Result<()> {
        let holidays = [
            (2024, 1, 1),
            (2024, 1, 6),
            (2024, 5, 1),
            (2024, 8, 15),
            (2024, 10, 3),
            (2024, 11, 1),
            (2024, 12, 25),
            (2024, 12, 26),
        ];

        for (y, m, d) in holidays {
            let date = NaiveDate::from_ymd_opt(y, m, d)
                .ok_or_else(|| anyhow::anyhow!("Invalid date: {y}-{m}-{d}"))?;
            assert!(is_bavarian_holiday(date));
        }
        Ok(())
    }

    #[test]
    fn test_variable_holidays_2024() -> Result<()> {
        let holidays = [
            (2024, 3, 29),
            (2024, 4, 1),
            (2024, 5, 9),
            (2024, 5, 20),
            (2024, 5, 30),
        ];

        for (y, m, d) in holidays {
            let date = NaiveDate::from_ymd_opt(y, m, d)
                .ok_or_else(|| anyhow::anyhow!("Invalid date: {y}-{m}-{d}"))?;
            assert!(is_bavarian_holiday(date));
        }
        Ok(())
    }

    #[test]
    fn test_regular_weekday_not_holiday() -> Result<()> {
        let date1 =
            NaiveDate::from_ymd_opt(2024, 2, 13).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date1));

        let date2 =
            NaiveDate::from_ymd_opt(2024, 7, 17).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date2));

        Ok(())
    }

    /// Builds a gym-local (`Europe/Berlin`) wall-clock time and returns the
    /// UTC instant, which is what callers hand to `is_open`.
    fn make_local_datetime(year: i32, month: u32, day: u32, hour: u32, min: u32) -> DateTime<Utc> {
        make_datetime_in(Berlin, year, month, day, hour, min)
    }

    fn make_datetime_in(
        tz: Tz,
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
    ) -> DateTime<Utc> {
        tz.with_ymd_and_hms(year, month, day, hour, min, 0)
            .single()
            .map_or_else(
                || unreachable!("test time must exist exactly once in {tz}"),
                |dt| dt.with_timezone(&Utc),
            )
    }

    fn make_utc(year: i32, month: u32, day: u32, hour: u32, min: u32) -> DateTime<Utc> {
        make_datetime_in(Tz::UTC, year, month, day, hour, min)
    }

    #[test]
    fn test_schedule_default_timezone_is_europe_berlin() {
        assert_eq!(GymSchedule::default().timezone(), Berlin);
    }

    #[test]
    fn test_schedule_new_uses_configured_timezone() {
        let config = ScheduleConfig {
            timezone: chrono_tz::America::New_York,
            ..ScheduleConfig::default()
        };
        assert_eq!(
            GymSchedule::new(&config).timezone(),
            chrono_tz::America::New_York
        );
    }

    #[test]
    fn test_is_open_interprets_utc_in_gym_timezone_summer() {
        let schedule = GymSchedule::default();
        // CEST = UTC+2: 04:30 UTC is 06:30 in the gym, 03:30 UTC is 05:30.
        assert!(schedule.is_open(&make_utc(2024, 7, 15, 4, 30)));
        assert!(!schedule.is_open(&make_utc(2024, 7, 15, 3, 30)));
        // 21:30 UTC is 23:30 in the gym — closed, although 21:30 itself is
        // within opening hours.
        assert!(!schedule.is_open(&make_utc(2024, 7, 15, 21, 30)));
    }

    #[test]
    fn test_is_open_interprets_utc_in_gym_timezone_winter() {
        let schedule = GymSchedule::default();
        // CET = UTC+1: 05:30 UTC is 06:30 in the gym, 04:30 UTC is 05:30.
        assert!(schedule.is_open(&make_utc(2024, 1, 15, 5, 30)));
        assert!(!schedule.is_open(&make_utc(2024, 1, 15, 4, 30)));
        assert!(!schedule.is_open(&make_utc(2024, 1, 15, 22, 30)));
    }

    #[test]
    fn test_is_open_respects_configured_timezone() {
        let config = ScheduleConfig {
            timezone: chrono_tz::America::New_York,
            ..ScheduleConfig::default()
        };
        let schedule = GymSchedule::new(&config);
        // EDT = UTC-4: 10:30 UTC is 06:30 local, 09:30 UTC is 05:30 local.
        assert!(schedule.is_open(&make_utc(2024, 7, 15, 10, 30)));
        assert!(!schedule.is_open(&make_utc(2024, 7, 15, 9, 30)));
    }

    #[test]
    fn test_opening_time_on_uses_gym_timezone() {
        let schedule = GymSchedule::default();
        let monday = NaiveDate::from_ymd_opt(2024, 6, 17).unwrap_or_default();
        let saturday = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap_or_default();
        // 06:00 / 09:00 CEST.
        assert_eq!(
            schedule.opening_time_on(monday),
            make_utc(2024, 6, 17, 4, 0)
        );
        assert_eq!(
            schedule.opening_time_on(saturday),
            make_utc(2024, 6, 15, 7, 0)
        );
    }

    #[test]
    fn test_next_closing_after_same_day() {
        let schedule = GymSchedule::default();
        // Monday 10:00 CEST → closes 23:00 CEST.
        assert_eq!(
            schedule.next_closing_after(make_utc(2024, 6, 17, 8, 0)),
            make_utc(2024, 6, 17, 21, 0)
        );
    }

    #[test]
    fn test_next_closing_after_closing_rolls_to_next_day() {
        let schedule = GymSchedule::default();
        // Monday 23:30 CEST → Tuesday 23:00 CEST.
        assert_eq!(
            schedule.next_closing_after(make_utc(2024, 6, 17, 21, 30)),
            make_utc(2024, 6, 18, 21, 0)
        );
        // Exactly at closing → the next day's closing.
        assert_eq!(
            schedule.next_closing_after(make_utc(2024, 6, 17, 21, 0)),
            make_utc(2024, 6, 18, 21, 0)
        );
    }

    #[test]
    fn test_next_closing_after_handles_midnight_close() {
        let schedule = GymSchedule::new_for_test(6, 24, 9, 24);
        // Monday 22:00 CEST, closing hour 24 → Tuesday 00:00 CEST.
        assert_eq!(
            schedule.next_closing_after(make_utc(2024, 6, 17, 20, 0)),
            make_utc(2024, 6, 17, 22, 0)
        );
    }

    #[test]
    fn test_is_open_holiday_decided_by_gym_local_date() {
        let schedule = GymSchedule::default();
        // 2024-12-24 23:30 UTC is already 2024-12-25 00:30 in the gym; by
        // 07:30 UTC (08:30 local) the holiday's weekend hours still apply.
        assert!(!schedule.is_open(&make_utc(2024, 12, 25, 7, 30)));
        assert!(schedule.is_open(&make_utc(2024, 12, 25, 8, 30)));
    }

    #[test]
    fn test_schedule_default_values() {
        let schedule = GymSchedule::default();
        assert_eq!(schedule.weekday_open, 6);
        assert_eq!(schedule.weekday_close, 23);
        assert_eq!(schedule.weekend_open, 9);
        assert_eq!(schedule.weekend_close, 21);
    }

    #[test]
    fn test_weekday_open_during_hours() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 14, 10, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_open_at_opening() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 12, 6, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_open_at_closing() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 12, 23, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_closed_before_opening() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 12, 5, 30);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_closed_after_closing() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 12, 23, 1);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_weekend_open_during_hours() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 17, 14, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekend_closed_before_opening() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 2, 18, 8, 0);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_holiday_uses_weekend_schedule() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 12, 25, 8, 0);
        assert!(!schedule.is_open(&time));
        let time = make_local_datetime(2024, 12, 25, 10, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_just_before_transition() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 3, 31, 1, 59);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_just_after_transition() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 3, 31, 3, 0);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_during_open_hours() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 3, 31, 10, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_early_morning() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 10, 27, 1, 59);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_during_open_hours() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 10, 27, 15, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_at_closing() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 10, 27, 21, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_dst_day_before_spring_forward() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 3, 30, 20, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_dst_day_after_fall_back() {
        let schedule = GymSchedule::default();
        let time = make_local_datetime(2024, 10, 28, 7, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_2025() {
        let schedule = GymSchedule::default();
        let morning_before_open = make_local_datetime(2025, 3, 30, 8, 0);
        let during_open = make_local_datetime(2025, 3, 30, 12, 0);
        let after_close = make_local_datetime(2025, 3, 30, 22, 0);

        assert!(!schedule.is_open(&morning_before_open));
        assert!(schedule.is_open(&during_open));
        assert!(!schedule.is_open(&after_close));
    }

    #[test]
    fn test_fall_back_2025() {
        let schedule = GymSchedule::default();
        let morning_before_open = make_local_datetime(2025, 10, 26, 8, 30);
        let during_open = make_local_datetime(2025, 10, 26, 14, 0);
        let at_closing = make_local_datetime(2025, 10, 26, 21, 0);

        assert!(!schedule.is_open(&morning_before_open));
        assert!(schedule.is_open(&during_open));
        assert!(schedule.is_open(&at_closing));
    }

    #[cfg(test)]
    mod proptest_tests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn easter_always_in_march_or_april(year in 1900i32..2100) {
                if let Some(easter) = easter_date(year) {
                    let month = easter.month();
                    prop_assert!(month == 3 || month == 4,
                        "Easter should be in March or April, got month {} for year {}",
                        month, year);
                }
            }

            #[test]
            fn easter_always_on_sunday(year in 1900i32..2100) {
                if let Some(easter) = easter_date(year) {
                    prop_assert_eq!(easter.weekday().num_days_from_monday(), 6,
                        "Easter should always be on Sunday for year {}", year);
                }
            }

            #[test]
            fn easter_date_is_valid(year in 1583i32..4099) {
                let result = easter_date(year);
                prop_assert!(result.is_some(),
                    "easter_date should return Some for year {}", year);
            }
        }
    }
}
