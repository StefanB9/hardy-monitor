//! Human-readable opening state ("Open · closes 23:00").

use chrono::{DateTime, Utc};
use hardy_core::schedule::GymSchedule;

/// Whether the gym is open and when that changes, in gym-local time.
pub fn opening_status(now: DateTime<Utc>, schedule: &GymSchedule) -> String {
    let tz = schedule.timezone();
    if schedule.is_open(&now) {
        let closes = schedule.next_closing_after(now).with_timezone(&tz);
        return format!("Open · closes {}", closes.format("%H:%M"));
    }
    let today = now.with_timezone(&tz).date_naive();
    let opening_today = schedule.opening_time_on(today);
    if now < opening_today {
        return format!(
            "Closed · opens {}",
            opening_today.with_timezone(&tz).format("%H:%M")
        );
    }
    match today.succ_opt() {
        Some(tomorrow) => format!(
            "Closed · opens tomorrow {}",
            schedule
                .opening_time_on(tomorrow)
                .with_timezone(&tz)
                .format("%H:%M")
        ),
        None => "Closed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    /// Monday 2024-06-17 `h:mi` gym-local, plus `days`.
    fn local(days: i64, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Ok(GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc)
            + chrono::TimeDelta::days(days))
    }

    #[test]
    fn test_opening_status_open() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(opening_status(local(0, 12, 0)?, &s), "Open · closes 23:00");
        Ok(())
    }

    #[test]
    fn test_opening_status_before_and_after_hours() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(opening_status(local(0, 5, 0)?, &s), "Closed · opens 06:00");
        assert_eq!(
            opening_status(local(0, 23, 30)?, &s),
            "Closed · opens tomorrow 06:00"
        );
        // Friday night → Saturday opens later.
        assert_eq!(
            opening_status(local(4, 23, 30)?, &s),
            "Closed · opens tomorrow 09:00"
        );
        Ok(())
    }
}
