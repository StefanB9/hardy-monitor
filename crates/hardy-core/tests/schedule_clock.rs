//! Integration tests for opening hours driven by `MockClock`.
#![allow(clippy::float_cmp)]
#![allow(clippy::manual_string_new)]

use anyhow::{Context, Result};
use chrono::{Duration as ChronoDuration, TimeZone, Utc};
use hardy_core::{
    Clock, MockClock,
    config::{ScheduleConfig, ScheduleHours},
    schedule::GymSchedule,
};

fn create_test_schedule(
    weekday_open: u32,
    weekday_close: u32,
    weekend_open: u32,
    weekend_close: u32,
) -> GymSchedule {
    let config = ScheduleConfig {
        timezone: chrono_tz::Europe::Berlin,
        weekday: ScheduleHours {
            open_hour: weekday_open,
            close_hour: weekday_close,
        },
        weekend: ScheduleHours {
            open_hour: weekend_open,
            close_hour: weekend_close,
        },
    };
    GymSchedule::new(&config)
}

#[test]
fn test_schedule_with_mock_clock() -> Result<()> {
    // 10:00 UTC on a Monday in June is 12:00 CEST in the gym.
    let clock = MockClock::new(
        Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0)
            .single()
            .context("valid UTC timestamp")?,
    );
    let schedule = create_test_schedule(6, 22, 8, 20);

    assert!(
        schedule.is_open(&clock.now_utc()),
        "Gym should be open at 12:00 local on Monday"
    );

    Ok(())
}

#[test]
fn test_schedule_open_close_transitions() -> Result<()> {
    // 05:00 UTC = 07:00 CEST (open), 12:00 UTC = 14:00 (open),
    // 00:00 UTC = 02:00 (closed) — independent of the host timezone.
    let clock = MockClock::new(
        Utc.with_ymd_and_hms(2024, 6, 17, 5, 0, 0)
            .single()
            .context("valid UTC timestamp")?,
    );
    let schedule = create_test_schedule(6, 22, 8, 20);

    assert!(
        schedule.is_open(&clock.now_utc()),
        "Should be open at 07:00"
    );

    clock.advance(ChronoDuration::hours(7));
    assert!(
        schedule.is_open(&clock.now_utc()),
        "Should be open at 14:00"
    );

    clock.advance(ChronoDuration::hours(12));
    assert!(
        !schedule.is_open(&clock.now_utc()),
        "Should be closed at 02:00"
    );

    Ok(())
}
