mod holidays;
mod hours;

use chrono::{DateTime, Datelike, Timelike, Utc};
use chrono_tz::Tz;
pub use holidays::is_bavarian_holiday;

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
    /// Schedule from the configured hours and timezone.
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

#[cfg(test)]
mod tests {

    use chrono::{TimeZone, Utc};
    use chrono_tz::{Europe::Berlin, Tz};

    use super::*;

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
}
