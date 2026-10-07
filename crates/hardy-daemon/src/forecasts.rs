//! Hourly forecast logging, so their accuracy can be measured later.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, Timelike, Utc};
use hardy_core::{Database, GymSchedule};
use hardy_ml::{History, forecast_log::forecast_log_entries, maintenance::ModelMaintenance};

/// History the model's features need ("same time last week").
const HISTORY: TimeDelta = TimeDelta::days(8);
/// Logged forecasts are kept this long.
const RETENTION: TimeDelta = TimeDelta::days(90);

/// Whether forecasts are logged for this fetch slot: once per full hour.
pub(crate) fn is_logging_slot(slot: DateTime<Utc>) -> bool {
    slot.minute() == 0
}

/// Logs the forecasts made at `slot` and prunes old ones. Returns how many
/// were stored.
pub(crate) async fn log_forecasts(
    database: &Database,
    schedule: &GymSchedule,
    models: &ModelMaintenance,
    slot: DateTime<Utc>,
    max_hours_ahead: u32,
) -> Result<u64> {
    let logs = database
        .get_history_range(slot - HISTORY, slot)
        .await
        .context("failed to load history for forecasting")?;
    let history = History::from_logs(&logs);
    let entries = forecast_log_entries(
        models.current_model(),
        &history,
        schedule,
        slot,
        max_hours_ahead,
    );
    let stored = database.save_forecasts(&entries).await?;
    database.delete_forecasts_before(slot - RETENTION).await?;
    Ok(stored)
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_is_logging_slot_only_on_full_hours() {
        let at = |mi| Utc.with_ymd_and_hms(2024, 6, 17, 10, mi, 0).single();
        assert!(at(0).is_some_and(is_logging_slot));
        assert!(!at(1).is_some_and(is_logging_slot));
        assert!(!at(59).is_some_and(is_logging_slot));
    }
}
