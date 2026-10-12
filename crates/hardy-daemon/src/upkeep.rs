//! Work besides fetching: live forecasts, nightly repair and the forecast
//! run, and the schema check.

use std::time::Duration;

use hardy_core::{Database, alert::AlertService, db::SchemaStatus, health::HealthEvent, repair};
use hardy_ml::service::ForecastService;

use crate::{Worker, forecasts};

/// Longest the nightly repair may hold up the fetch loop; it continues on a
/// later tick (after the retry delay) if this runs out.
const REPAIR_BUDGET: Duration = Duration::from_secs(30);

/// Failures are logged only: forecasting must not disturb collection.
pub(crate) async fn publish_forecasts(
    worker: &Worker<'_>,
    service: &mut ForecastService,
    slot: chrono::DateTime<chrono::Utc>,
) {
    match forecasts::publish(worker.database, service, slot, worker.forecast_hours).await {
        Ok(logged) => tracing::debug!(%slot, logged, "published forecasts"),
        Err(e) => tracing::warn!(error = %format!("{e:#}"), "failed to publish forecasts"),
    }
}

async fn run_repair(worker: &Worker<'_>) {
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

/// Repairs the day that just closed (and any missed days), then lets the
/// forecast service start or collect its nightly run — in this order, so
/// the models are fitted on repaired data. The repair is time-limited and
/// saves progress per day, so the caller may drop this on shutdown.
pub(crate) async fn nightly_upkeep(worker: &Worker<'_>, service: &mut ForecastService) {
    run_repair(worker).await;
    if let Err(e) = service.tick(worker.database, chrono::Utc::now()).await {
        tracing::warn!(error = %format!("{e:#}"), "forecast run failed");
    }
}
