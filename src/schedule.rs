use chrono::{DateTime, Datelike, Local, NaiveDate, Timelike};

use crate::config::ScheduleConfig;

/// Gym schedule with configurable opening hours.
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

    /// Check if the gym is currently open.
    pub fn is_open(&self, time: &DateTime<Local>) -> bool {
        let date = time.date_naive();
        let hour = time.hour();
        let minute = time.minute();

        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            // Weekend or Holiday
            (self.weekend_open..self.weekend_close).contains(&hour)
                || (hour == self.weekend_close && minute == 0)
        } else {
            // Regular Weekday
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
    /// Create a custom schedule for testing purposes.
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

    /// Get the opening hour for a specific date.
    pub fn get_open_hour(&self, date: NaiveDate) -> u32 {
        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            self.weekend_open
        } else {
            self.weekday_open
        }
    }

    /// Get the closing hour for a specific date.
    pub fn get_close_hour(&self, date: NaiveDate) -> u32 {
        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            self.weekend_close
        } else {
            self.weekday_close
        }
    }
}

/// Check if a date is a Bavarian public holiday.
pub fn is_bavarian_holiday(date: NaiveDate) -> bool {
    let (d, m) = (date.day(), date.month());
    let year = date.year();

    // 1. Fixed Holidays (Bavaria)
    match (m, d) {
        (1, 1) => return true,   // New Year
        (1, 6) => return true,   // Epiphany
        (5, 1) => return true,   // Labour Day
        (8, 15) => return true,  // Assumption Day
        (10, 3) => return true,  // German Unity Day
        (11, 1) => return true,  // All Saints' Day
        (12, 25) => return true, // Christmas Day
        (12, 26) => return true, // 2nd Day of Christmas
        _ => {}
    }

    // 2. Variable Holidays (Easter based)
    // We calculate Easter Sunday for the given year to find variable holidays
    if let Some(easter) = easter_date(year) {
        let ordinal = date.ordinal();
        let easter_ordinal = easter.ordinal();

        // Good Friday: -2 days
        if ordinal == easter_ordinal - 2 {
            return true;
        }
        // Easter Monday: +1 day
        if ordinal == easter_ordinal + 1 {
            return true;
        }
        // Ascension Day: +39 days
        if ordinal == easter_ordinal + 39 {
            return true;
        }
        // Whit Monday: +50 days
        if ordinal == easter_ordinal + 50 {
            return true;
        }
        // Corpus Christi: +60 days
        if ordinal == easter_ordinal + 60 {
            return true;
        }
    }

    false
}

/// Calculate Easter date using the Anonymous Gregorian algorithm.
/// This is efficient and accurate for the Gregorian calendar (1583-4099).
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
    use chrono::{NaiveDate, TimeZone};

    use super::*;

    // ==================== Easter Date Tests ====================

    #[test]
    fn test_easter_2024() {
        // Easter 2024 is March 31
        let easter = easter_date(2024).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap());
    }

    #[test]
    fn test_easter_2025() {
        // Easter 2025 is April 20
        let easter = easter_date(2025).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(2025, 4, 20).unwrap());
    }

    #[test]
    fn test_easter_2026() {
        // Easter 2026 is April 5
        let easter = easter_date(2026).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(2026, 4, 5).unwrap());
    }

    #[test]
    fn test_easter_historical_1999() {
        // Easter 1999 was April 4
        let easter = easter_date(1999).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(1999, 4, 4).unwrap());
    }

    #[test]
    fn test_easter_edge_early_march() {
        // Easter 2008 was March 23 (early Easter)
        let easter = easter_date(2008).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(2008, 3, 23).unwrap());
    }

    #[test]
    fn test_easter_edge_late_april() {
        // Easter 2038 is April 25 (late Easter)
        let easter = easter_date(2038).unwrap();
        assert_eq!(easter, NaiveDate::from_ymd_opt(2038, 4, 25).unwrap());
    }

    // ==================== Bavarian Holiday Tests ====================

    #[test]
    fn test_fixed_holidays() {
        // New Year
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
        ));
        // Epiphany
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 1, 6).unwrap()
        ));
        // Labour Day
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 5, 1).unwrap()
        ));
        // Assumption Day
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 8, 15).unwrap()
        ));
        // German Unity Day
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 10, 3).unwrap()
        ));
        // All Saints' Day
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 11, 1).unwrap()
        ));
        // Christmas
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 12, 25).unwrap()
        ));
        // 2nd Christmas
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 12, 26).unwrap()
        ));
    }

    #[test]
    fn test_variable_holidays_2024() {
        // Easter 2024 is March 31
        // Good Friday: March 29
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 3, 29).unwrap()
        ));
        // Easter Monday: April 1
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 4, 1).unwrap()
        ));
        // Ascension Day: May 9 (Easter + 39)
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 5, 9).unwrap()
        ));
        // Whit Monday: May 20 (Easter + 50)
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 5, 20).unwrap()
        ));
        // Corpus Christi: May 30 (Easter + 60)
        assert!(is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 5, 30).unwrap()
        ));
    }

    #[test]
    fn test_regular_weekday_not_holiday() {
        // Random Tuesday in February
        assert!(!is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 2, 13).unwrap()
        ));
        // Random Wednesday in July
        assert!(!is_bavarian_holiday(
            NaiveDate::from_ymd_opt(2024, 7, 17).unwrap()
        ));
    }

    // ==================== GymSchedule Tests ====================

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
        // Wednesday at 10:00 (mid-day, should be open)
        let time = make_local_datetime(2024, 2, 14, 10, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_open_at_opening() {
        let schedule = GymSchedule::default();
        // Monday at 06:00 exactly
        let time = make_local_datetime(2024, 2, 12, 6, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_open_at_closing() {
        let schedule = GymSchedule::default();
        // Monday at 23:00 exactly (closing time, minute=0 is allowed)
        let time = make_local_datetime(2024, 2, 12, 23, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_closed_before_opening() {
        let schedule = GymSchedule::default();
        // Monday at 05:30 (before opening)
        let time = make_local_datetime(2024, 2, 12, 5, 30);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_weekday_closed_after_closing() {
        let schedule = GymSchedule::default();
        // Monday at 23:01 (after closing)
        let time = make_local_datetime(2024, 2, 12, 23, 1);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_weekend_open_during_hours() {
        let schedule = GymSchedule::default();
        // Saturday at 14:00
        let time = make_local_datetime(2024, 2, 17, 14, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_weekend_closed_before_opening() {
        let schedule = GymSchedule::default();
        // Sunday at 08:00 (before 09:00 opening)
        let time = make_local_datetime(2024, 2, 18, 8, 0);
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_holiday_uses_weekend_schedule() {
        let schedule = GymSchedule::default();
        // Christmas 2024 is Wednesday - should use weekend hours
        // At 08:00 (before weekend opening of 09:00), should be closed
        let time = make_local_datetime(2024, 12, 25, 8, 0);
        assert!(!schedule.is_open(&time));
        // At 10:00, should be open
        let time = make_local_datetime(2024, 12, 25, 10, 0);
        assert!(schedule.is_open(&time));
    }

    // ==================== DST Transition Tests ====================
    // Germany DST: Last Sunday in March (2:00→3:00) and October (3:00→2:00)

    #[test]
    fn test_spring_forward_just_before_transition() {
        let schedule = GymSchedule::default();
        // March 31, 2024 at 01:59 (just before DST spring forward)
        // This is a Sunday, so weekend schedule (9-21)
        let time = make_local_datetime(2024, 3, 31, 1, 59);
        // 01:59 is before weekend opening (9:00), so should be closed
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_just_after_transition() {
        let schedule = GymSchedule::default();
        // March 31, 2024 at 03:00 (just after DST spring forward, 2:00 doesn't exist)
        // This is a Sunday, so weekend schedule (9-21)
        let time = make_local_datetime(2024, 3, 31, 3, 0);
        // 03:00 is before weekend opening (9:00), so should be closed
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_during_open_hours() {
        let schedule = GymSchedule::default();
        // March 31, 2024 at 10:00 (after DST, during open hours)
        // Sunday with weekend schedule (9-21)
        let time = make_local_datetime(2024, 3, 31, 10, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_early_morning() {
        let schedule = GymSchedule::default();
        // October 27, 2024 at 01:59 (before the ambiguous 2:00-2:59 window)
        // This is a Sunday, so weekend schedule (9-21)
        let time = make_local_datetime(2024, 10, 27, 1, 59);
        // 01:59 is before weekend opening (9:00), so should be closed
        assert!(!schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_during_open_hours() {
        let schedule = GymSchedule::default();
        // October 27, 2024 at 15:00 (after DST fall back, during open hours)
        // Sunday with weekend schedule (9-21)
        let time = make_local_datetime(2024, 10, 27, 15, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_fall_back_at_closing() {
        let schedule = GymSchedule::default();
        // October 27, 2024 at 21:00 exactly (closing time after fall back)
        // Sunday with weekend schedule (9-21)
        let time = make_local_datetime(2024, 10, 27, 21, 0);
        // At exactly closing time with minute=0, should still be open
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_dst_day_before_spring_forward() {
        let schedule = GymSchedule::default();
        // March 30, 2024 (Saturday before spring forward)
        // Weekend schedule (9-21)
        let time = make_local_datetime(2024, 3, 30, 20, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_dst_day_after_fall_back() {
        let schedule = GymSchedule::default();
        // October 28, 2024 (Monday after fall back)
        // Weekday schedule (6-23)
        let time = make_local_datetime(2024, 10, 28, 7, 0);
        assert!(schedule.is_open(&time));
    }

    #[test]
    fn test_spring_forward_2025() {
        let schedule = GymSchedule::default();
        // March 30, 2025 is the spring forward date (Sunday)
        // Test that schedule works correctly on this day
        let morning_before_open = make_local_datetime(2025, 3, 30, 8, 0);
        let during_open = make_local_datetime(2025, 3, 30, 12, 0);
        let after_close = make_local_datetime(2025, 3, 30, 22, 0);

        assert!(!schedule.is_open(&morning_before_open)); // Before 9:00 weekend opening
        assert!(schedule.is_open(&during_open)); // During open hours
        assert!(!schedule.is_open(&after_close)); // After 21:00 weekend closing
    }

    #[test]
    fn test_fall_back_2025() {
        let schedule = GymSchedule::default();
        // October 26, 2025 is the fall back date (Sunday)
        // Test that schedule works correctly on this day
        let morning_before_open = make_local_datetime(2025, 10, 26, 8, 30);
        let during_open = make_local_datetime(2025, 10, 26, 14, 0);
        let at_closing = make_local_datetime(2025, 10, 26, 21, 0);

        assert!(!schedule.is_open(&morning_before_open)); // Before 9:00 weekend opening
        assert!(schedule.is_open(&during_open)); // During open hours
        assert!(schedule.is_open(&at_closing)); // At exactly 21:00 with minute=0
    }

    // ==================== Property-Based Tests ====================

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
                    // Sunday is weekday 6 in chrono (0=Monday)
                    prop_assert_eq!(easter.weekday().num_days_from_monday(), 6,
                        "Easter should always be on Sunday for year {}", year);
                }
            }

            #[test]
            fn easter_date_is_valid(year in 1583i32..4099) {
                // The algorithm is valid for Gregorian calendar (1583-4099)
                let result = easter_date(year);
                prop_assert!(result.is_some(),
                    "easter_date should return Some for year {}", year);
            }
        }
    }
}
