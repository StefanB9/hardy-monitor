//! Work besides fetching: the nightly repair and the schema check.
//! Forecasting runs in `hardy-forecaster`.

use std::time::Duration;

use chrono::{DateTime, Utc};
use hardy_core::{
    Database, GymSchedule, alert::AlertService, db::SchemaStatus, health::HealthEvent,
    repair::NightlyRepair,
};

use crate::Worker;

/// Longest the nightly repair may hold up the fetch loop; it continues on a
/// later tick (after the retry delay) if this runs out.
const REPAIR_BUDGET: Duration = Duration::from_secs(30);

/// Fetch cycles between schema version checks (about ten minutes).
const SCHEMA_CHECK_ITERATIONS: u64 = 10;

/// Repairs the day that just closed (and any missed days). Time-limited;
/// it saves progress per day, so the caller may drop it on shutdown. The
/// database is read only when `nightly`'s remembered state says a repair
/// is due.
pub(crate) async fn run_repair(worker: &Worker<'_>, nightly: &mut NightlyRepair) {
    match nightly
        .run(worker.database, worker.schedule, Utc::now(), REPAIR_BUDGET)
        .await
    {
        Ok(Some(((first, last), summary))) => tracing::info!(
            %first,
            %last,
            gaps_filled = summary.gaps_filled,
            records_deleted = summary.records_deleted,
            records_smoothed = summary.records_smoothed,
            boundary_entries_added = summary.boundary_entries_added,
            "nightly data repair done"
        ),
        Ok(None) => {}
        Err(e) => tracing::warn!(error = %format!("{e:#}"), "nightly data repair failed"),
    }
}

/// Whether fetch cycle `iteration` checks the schema: every tenth while
/// the gym is open. A newer schema matters only to the fetches, and one
/// that arrived overnight is still reported within ten minutes of opening.
pub(crate) fn schema_check_due(iteration: u64, schedule: &GymSchedule, now: DateTime<Utc>) -> bool {
    iteration.is_multiple_of(SCHEMA_CHECK_ITERATIONS) && schedule.is_open(&now)
}

/// Reports once if a newer build migrated the database while this daemon
/// runs. It keeps running: additive migrations may still work, and real
/// breakage shows up as a health outage.
pub(crate) async fn check_schema(database: &Database, alerts: &AlertService, reported: &mut bool) {
    match database.schema_status().await {
        Ok(SchemaStatus::DbNewer { db, app }) if !*reported => {
            tracing::error!(
                db,
                app,
                "database schema is newer than this daemon; update it"
            );
            alerts
                .publish_health(&HealthEvent::SchemaAhead { db, app })
                .await;
            *reported = true;
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "could not check the database schema"),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{TimeZone, Utc};

    use super::*;

    #[test]
    fn test_schema_check_due_every_tenth_cycle_while_open() -> Result<()> {
        let schedule = GymSchedule::default();
        // Monday 2024-06-17; CEST, so the gym is open 04:00–21:00 UTC.
        let open = Utc
            .with_ymd_and_hms(2024, 6, 17, 10, 0, 0)
            .single()
            .context("time")?;
        let closed = Utc
            .with_ymd_and_hms(2024, 6, 17, 23, 0, 0)
            .single()
            .context("time")?;

        assert!(schema_check_due(10, &schedule, open));
        assert!(schema_check_due(20, &schedule, open));
        assert!(!schema_check_due(11, &schedule, open));
        assert!(!schema_check_due(10, &schedule, closed));
        Ok(())
    }
}
