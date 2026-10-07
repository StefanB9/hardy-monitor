//! Gym-local time helpers and weekday names.

use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use chrono_tz::Tz;

pub(super) const DAY_NAMES_LONG: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

const DAY_NAMES_SHORT: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// The first instant of `date` in `tz`, as UTC.
///
/// Falls back to UTC midnight only if `tz` has no midnight that day (a DST
/// jump at 00:00, which IANA zones like `Europe/Berlin` never do).
pub fn midnight_local_as_utc(date: NaiveDate, tz: Tz) -> DateTime<Utc> {
    tz.from_local_datetime(&date.and_time(NaiveTime::MIN))
        .earliest()
        .map_or_else(
            || date.and_time(NaiveTime::MIN).and_utc(),
            |dt| dt.with_timezone(&Utc),
        )
}

/// Full weekday name for 0 = Monday … 6 = Sunday.
pub fn weekday_name(weekday: i32) -> &'static str {
    usize::try_from(weekday)
        .ok()
        .and_then(|i| DAY_NAMES_LONG.get(i))
        .copied()
        .unwrap_or("Unknown")
}

/// Three-letter weekday name for 0 = Monday … 6 = Sunday.
pub fn weekday_short(weekday: i32) -> &'static str {
    usize::try_from(weekday)
        .ok()
        .and_then(|i| DAY_NAMES_SHORT.get(i))
        .copied()
        .unwrap_or("???")
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_weekday_name() {
        assert_eq!(weekday_name(0), "Monday");
        assert_eq!(weekday_name(1), "Tuesday");
        assert_eq!(weekday_name(2), "Wednesday");
        assert_eq!(weekday_name(3), "Thursday");
        assert_eq!(weekday_name(4), "Friday");
        assert_eq!(weekday_name(5), "Saturday");
        assert_eq!(weekday_name(6), "Sunday");
        assert_eq!(weekday_name(7), "Unknown");
    }

    #[test]
    fn test_weekday_short() {
        assert_eq!(weekday_short(0), "Mon");
        assert_eq!(weekday_short(1), "Tue");
        assert_eq!(weekday_short(2), "Wed");
        assert_eq!(weekday_short(3), "Thu");
        assert_eq!(weekday_short(4), "Fri");
        assert_eq!(weekday_short(5), "Sat");
        assert_eq!(weekday_short(6), "Sun");
        assert_eq!(weekday_short(7), "???");
    }
}
