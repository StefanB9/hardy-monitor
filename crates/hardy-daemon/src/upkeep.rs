//! Work besides fetching: the nightly repair and the schema check.
//! Forecasting runs in `hardy-forecaster`.

use std::time::Duration;

use hardy_core::{Database, alert::AlertService, db::SchemaStatus, health::HealthEvent, repair};

use crate::Worker;

/// Longest the nightly repair may hold up the fetch loop; it continues on a
/// later tick (after the retry delay) if this runs out.
const REPAIR_BUDGET: Duration = Duration::from_secs(30);

/// Repairs the day that just closed (and any missed days). Time-limited;
/// it saves progress per day, so the caller may drop it on shutdown.
pub(crate) async fn run_repair(worker: &Worker<'_>) {
    match repair::run_nightly_repair(
        worker.database,
        worker.schedule,
        chrono::Utc::now(),
        REPAIR_BUDGET,
    )
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
