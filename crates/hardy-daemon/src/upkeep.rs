//! Work besides fetching: forecast logging, nightly repair and training, and
//! the schema check.

use hardy_core::{Database, alert::AlertService, db::SchemaStatus, health::HealthEvent, repair};
use hardy_ml::maintenance::ModelMaintenance;

use crate::{Worker, forecasts};

/// Failures are logged only: accuracy tracking must not disturb collection.
pub(crate) async fn log_forecasts(
    worker: &Worker<'_>,
    models: &ModelMaintenance,
    slot: chrono::DateTime<chrono::Utc>,
) {
    match forecasts::log_forecasts(
        worker.database,
        worker.schedule,
        models,
        slot,
        worker.forecast_hours,
    )
    .await
    {
        Ok(stored) => tracing::debug!(%slot, stored, "logged forecasts"),
        Err(e) => tracing::warn!(error = %format!("{e:#}"), "failed to log forecasts"),
    }
}

async fn run_repair(worker: &Worker<'_>) {
    match repair::run_nightly_repair(worker.database, worker.schedule, chrono::Utc::now()).await {
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

/// Repairs the day that just closed (and any missed days), then lets model
/// maintenance start or collect training — in this order, so the nightly
/// retrain trains on repaired data. The repair is not raced against
/// shutdown: each day is repaired in small committed steps and a rerun is
/// harmless, but progress is only recorded once the range is done.
pub(crate) async fn nightly_upkeep(worker: &Worker<'_>, models: &mut ModelMaintenance) {
    run_repair(worker).await;
    if let Err(e) = models.tick(worker.database, chrono::Utc::now()).await {
        tracing::warn!(error = %e, "model maintenance failed");
    }
}
