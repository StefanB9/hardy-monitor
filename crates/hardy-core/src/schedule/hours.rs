//! Opening and closing instants of a gym-local day.

use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, TimeZone, Utc};

use super::{GymSchedule, is_bavarian_holiday};

impl GymSchedule {
    /// Schedule in Europe/Berlin with the given hours.
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

    /// Opening hour (gym-local) on `date`; holidays use weekend hours.
    pub fn get_open_hour(&self, date: NaiveDate) -> u32 {
        if is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5 {
            self.weekend_open
        } else {
            self.weekday_open
        }
    }

    /// Closing hour (gym-local) on `date`; 24 means midnight.
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

    /// The latest closing instant at or before `now`.
    pub fn last_closing_at_or_before(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        // Two days back always precedes at least one closing.
        let mut closing = self.next_closing_after(now - chrono::TimeDelta::days(2));
        loop {
            let next = self.next_closing_after(closing);
            if next > now {
                return closing;
            }
            closing = next;
        }
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

#[cfg(test)]
mod tests {
    use chrono::{NaiveDate, TimeZone, Utc};
    use chrono_tz::Tz;

    use super::*;

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
    fn test_last_closing_at_or_before() {
        let schedule = GymSchedule::default();
        // Monday 10:00 CEST → Sunday 21:00 CEST (weekend hours).
        assert_eq!(
            schedule.last_closing_at_or_before(make_utc(2024, 6, 17, 8, 0)),
            make_utc(2024, 6, 16, 19, 0)
        );
        // Exactly at Monday's closing (23:00 CEST) → that closing.
        assert_eq!(
            schedule.last_closing_at_or_before(make_utc(2024, 6, 17, 21, 0)),
            make_utc(2024, 6, 17, 21, 0)
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
}
