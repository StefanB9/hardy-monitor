//! Connecting to the database at startup: retried with backoff, reported to
//! the phone when it takes too long, and refused for a newer schema.

use std::{future::Future, pin::Pin};

use anyhow::{Context, Result, bail};
use hardy_core::{
    AppError, RetryPolicy,
    alert::AlertService,
    config::DatabaseConfig,
    db::{Database, Migrations},
    health::{HealthEvent, HealthMonitor},
    schedule::GymSchedule,
};

/// Connects and migrates. `Ok(None)` means shutdown was requested first.
pub(crate) async fn connect_database(
    config: &DatabaseConfig,
    policy: &RetryPolicy,
    schedule: &GymSchedule,
    alerts: &AlertService,
    health: &mut HealthMonitor,
    mut shutdown: Pin<&mut impl Future<Output = ()>>,
) -> Result<Option<Database>> {
    let mut attempt = 1;
    loop {
        let result = tokio::select! {
            result = Database::connect(config, Migrations::Apply) => result,
            () = &mut shutdown => return Ok(None),
        };
        let now = chrono::Utc::now();
        let error = match result {
            Ok(database) => {
                if let Some(event) = health.success(now) {
                    alerts.publish_health(&event).await;
                }
                return Ok(Some(database));
            }
            Err(e) => e,
        };

        match AppError::from_anyhow_sqlx(&error, "connect_database") {
            AppError::SchemaTooNew { db, app } => {
                tracing::error!(db, app, "database schema is newer than this daemon");
                alerts
                    .publish_health(&HealthEvent::SchemaAhead { db, app })
                    .await;
                return Err(error).context("refusing to run against a newer database schema");
            }
            app_error if app_error.is_retryable() => {
                // Closed hours neither start nor extend an outage.
                if schedule.is_open(&now)
                    && let Some(event) = health.failure(now, &app_error.to_string())
                {
                    alerts.publish_health(&event).await;
                }
                let delay = policy.delay_before_retry(attempt);
                tracing::warn!(attempt, error = %app_error, ?delay, "database unavailable, retrying");
                attempt = attempt.saturating_add(1);
                tokio::select! {
                    () = tokio::time::sleep(delay) => {}
                    () = &mut shutdown => return Ok(None),
                }
            }
            app_error => bail!(app_error),
        }
    }
}
