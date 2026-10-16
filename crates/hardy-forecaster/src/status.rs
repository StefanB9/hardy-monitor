//! The status line shown between runs.

use chrono::{DateTime, DurationRound, TimeDelta, Utc};
use hardy_core::{Tz, schedule::GymSchedule};
use hardy_ml::service::ForecastService;

use crate::forecasts::should_publish;

/// The status line for `service` at `now`.
pub(crate) fn line(
    service: &ForecastService,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
) -> String {
    describe(
        service.is_running(),
        service.last_run_at(),
        service.fitted_count(),
        next_publish(schedule, now),
        schedule.timezone(),
    )
}

/// The next quarter hour after `now` at which forecasts are published.
fn next_publish(schedule: &GymSchedule, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let quarter = TimeDelta::minutes(15);
    let start = now.duration_trunc(quarter).ok()? + quarter;
    // Two days covers any closure the schedule has.
    (0..2 * 24 * 4)
        .map(|i| start + quarter * i)
        .find(|&slot| should_publish(schedule, slot))
}

fn describe(
    running: bool,
    last_run: Option<DateTime<Utc>>,
    fitted: usize,
    next: Option<DateTime<Utc>>,
    tz: Tz,
) -> String {
    let mut parts = Vec::with_capacity(4);
    if running {
        parts.push("forecast run in progress".to_string());
    }
    parts.push(if fitted == 0 {
        "no models fitted yet".to_string()
    } else {
        format!("{fitted} models fitted")
    });
    if !running {
        parts.push(last_run.map_or_else(
            || "no run yet".to_string(),
            |t| format!("last run {}", t.with_timezone(&tz).format("%a %H:%M")),
        ));
    }
    if let Some(next) = next {
        parts.push(format!(
            "next forecasts {}",
            next.with_timezone(&tz).format("%H:%M")
        ));
    }
    parts.join(" · ")
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    fn berlin(day: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        // The default schedule's time zone is Europe/Berlin.
        Ok(GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, day, h, mi, 0)
            .single()
            .context("time")?
            .with_timezone(&Utc))
    }

    #[test]
    fn test_next_publish_is_the_next_open_quarter_hour() -> Result<()> {
        let schedule = GymSchedule::default();
        // Monday 10:07 → 10:15; 10:15 itself → 10:30.
        assert_eq!(
            next_publish(&schedule, berlin(17, 10, 7)?),
            Some(berlin(17, 10, 15)?)
        );
        assert_eq!(
            next_publish(&schedule, berlin(17, 10, 15)?),
            Some(berlin(17, 10, 30)?)
        );
        // After closing: the first quarter hour after opening next morning.
        let next = next_publish(&schedule, berlin(17, 23, 30)?).context("next")?;
        assert!(
            next > berlin(18, 5, 0)? && next <= berlin(18, 7, 0)?,
            "{next}"
        );
        Ok(())
    }

    #[test]
    fn test_describe_idle_and_running() -> Result<()> {
        let tz = GymSchedule::default().timezone();
        let idle = describe(
            false,
            Some(berlin(17, 23, 36)?),
            46,
            Some(berlin(18, 10, 30)?),
            tz,
        );
        assert_eq!(
            idle,
            "46 models fitted · last run Mon 23:36 · next forecasts 10:30"
        );
        let first = describe(false, None, 0, None, tz);
        assert_eq!(first, "no models fitted yet · no run yet");
        let running = describe(true, None, 0, Some(berlin(18, 10, 30)?), tz);
        assert_eq!(
            running,
            "forecast run in progress · no models fitted yet · next forecasts 10:30"
        );
        Ok(())
    }
}
