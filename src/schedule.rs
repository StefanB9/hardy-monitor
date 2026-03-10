use chrono::{DateTime, Datelike, Local, NaiveDate, Timelike};

use crate::config::ScheduleConfig;

#[derive(Debug, Clone)]
pub struct GymSchedule {
    weekday_open: u32,
    weekday_close: u32,
    weekend_open: u32,
    weekend_close: u32,
}

impl GymSchedule {
    pub fn new(config: &ScheduleConfig) -> Self {
        Self {
            weekday_open: config.weekday.open_hour,
            weekday_close: config.weekday.close_hour,
            weekend_open: config.weekend.open_hour,
            weekend_close: config.weekend.close_hour,
        }
    }

    pub fn is_open(&self, time: &DateTime<Local>) -> bool {
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
        Self {
            weekday_open: 6,
            weekday_close: 23,
            weekend_open: 9,
            weekend_close: 21,
        }
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
}

pub fn is_bavarian_holiday(date: NaiveDate) -> bool {
    let (d, m) = (date.day(), date.month());
    let year = date.year();

    match (m, d) {
        (1 | 5 | 11, 1) | (1, 6) | (8, 15) | (10, 3) | (12, 25 | 26) => return true,
        _ => {}
    }

    if let Some(easter) = easter_date(year) {
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

    NaiveDate::from_ymd_opt(year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{NaiveDate, TimeZone};

    use super::*;

    #[test]
    fn test_easter_2024() -> Result<()> {
        let easter = easter_date(2024).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(2024, 3, 31).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2025() -> Result<()> {
        let easter = easter_date(2025).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(2025, 4, 20).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2026() -> Result<()> {
        let easter = easter_date(2026).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(2026, 4, 5).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_historical_1999() -> Result<()> {
        let easter = easter_date(1999).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(1999, 4, 4).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_early_march() -> Result<()> {
        let easter = easter_date(2008).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(2008, 3, 23).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_late_april() -> Result<()> {
        let easter = easter_date(2038).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected = NaiveDate::from_ymd_opt(2038, 4, 25).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_fixed_holidays() -> Result<()>  {
        let holidays = [
            (2024, 1, 1), (2024, 1, 6), (2024, 5, 1),
            (2024, 8, 15), (2024, 10, 3), (2024, 11, 1),
            (2024, 12, 25), (2024, 12, 26)
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
            (2024, 3, 29), (2024, 4, 1), (2024, 5, 9),
            (2024, 5, 20), (2024, 5, 30)
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
        let date1 = NaiveDate::from_ymd_opt(2024, 2, 13)
            .ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date1));

        let date2 = NaiveDate::from_ymd_opt(2024, 7, 17)
            .ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date2));

        Ok(())
    }

    fn make_local_datetime(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
    ) -> DateTime<Local> {
        Local
            .with_ymd_and_hms(year, month, day, hour, min, 0)
            .unwrap()
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
