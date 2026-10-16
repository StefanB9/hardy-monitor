//! Live forecasts: published every quarter hour while the gym is open, and
//! logged every full hour so their accuracy can be measured later.

use anyhow::{Context, Result};
use chrono::{DateTime, Timelike, Utc};
use hardy_core::{Database, schedule::GymSchedule};
use hardy_ml::service::ForecastService;

/// Whether forecasts are published for this fetch slot: every quarter hour.
pub(crate) fn is_publishing_slot(slot: DateTime<Utc>) -> bool {
    slot.minute().is_multiple_of(15)
}

/// Whether forecasts are published at `slot`: a quarter hour while the gym
/// is open.
pub(crate) fn should_publish(schedule: &GymSchedule, slot: DateTime<Utc>) -> bool {
    is_publishing_slot(slot) && schedule.is_open(&slot)
}

/// Whether forecasts are logged for this fetch slot: once per full hour.
pub(crate) fn is_logging_slot(slot: DateTime<Utc>) -> bool {
    slot.minute() == 0
}

/// Publishes every model's forecast made at `slot`; on full hours also logs
/// the default forecast (kept for good, so accuracy covers all history).
/// Returns how many log entries were stored.
pub(crate) async fn publish(
    database: &Database,
    service: &mut ForecastService,
    slot: DateTime<Utc>,
    log_hours: u32,
) -> Result<u64> {
    let live = service
        .publish(database, slot)
        .await
        .context("failed to publish forecasts")?;
    if !is_logging_slot(slot) {
        return Ok(0);
    }
    database.save_forecasts(&live.log_entries(log_hours)).await
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_is_publishing_slot_every_quarter_hour() {
        let at = |mi| Utc.with_ymd_and_hms(2024, 6, 17, 10, mi, 0).single();
        assert!(at(0).is_some_and(is_publishing_slot));
        assert!(at(15).is_some_and(is_publishing_slot));
        assert!(at(45).is_some_and(is_publishing_slot));
        assert!(!at(16).is_some_and(is_publishing_slot));
    }

    #[test]
    fn test_should_publish_only_while_open() {
        let schedule = GymSchedule::default();
        // 2024-06-17 is a Monday; Berlin is UTC+2 (open 06:00–23:00).
        let at = |h, mi| Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0).single();
        assert!(at(8, 15).is_some_and(|t| should_publish(&schedule, t)));
        assert!(!at(8, 16).is_some_and(|t| should_publish(&schedule, t)));
        // 02:00 Berlin: closed.
        assert!(!at(0, 0).is_some_and(|t| should_publish(&schedule, t)));
    }

    #[test]
    fn test_is_logging_slot_only_on_full_hours() {
        let at = |mi| Utc.with_ymd_and_hms(2024, 6, 17, 10, mi, 0).single();
        assert!(at(0).is_some_and(is_logging_slot));
        assert!(!at(1).is_some_and(is_logging_slot));
        assert!(!at(59).is_some_and(is_logging_slot));
    }
}
