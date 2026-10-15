//! Shared test fixtures: gym-local times and synthetic histories.

use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use hardy_core::GymSchedule;

use crate::history::History;

/// `days` after Monday 2024-06-03 at `h:mi` gym-local time.
pub(crate) fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
    GymSchedule::default()
        .timezone()
        .with_ymd_and_hms(2024, 6, 3, h, mi, 0)
        .single()
        .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
        + TimeDelta::days(days)
}

/// One reading per minute while the gym is open over `days` days, valued
/// `f(time)`.
pub(crate) fn minutely(days: i64, f: impl Fn(DateTime<Utc>) -> f64) -> History {
    let schedule = GymSchedule::default();
    let start = local(0, 0, 0);
    let mut points = Vec::new();
    for minute in 0..days * 24 * 60 {
        let t = start + TimeDelta::minutes(minute);
        if schedule.is_open(&t) {
            points.push((t, f(t)));
        }
    }
    History::new(points)
}
