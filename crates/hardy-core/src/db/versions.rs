//! Cheap "has anything new been written" checks, so the GUI reloads the
//! published forecasts and the forecast suite only when they changed.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// Changes whenever the stored forecasts of the asked models change: each
/// publication has a new `made_at`, and a model dropping out changes the
/// count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveForecastsVersion {
    made_at: Option<DateTime<Utc>>,
    rows: i64,
}

/// Changes whenever anything the GUI shows of the suite changes: run
/// requests and attempts, model states, benchmark runs and the selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteVersion {
    retrain_requested_at: Option<DateTime<Utc>>,
    last_attempt_at: Option<DateTime<Utc>>,
    last_error: Option<String>,
    models_updated_at: Option<DateTime<Utc>>,
    models: i64,
    latest_run: Option<i64>,
    selected_at: Option<DateTime<Utc>>,
    selections: i64,
}

impl Database {
    /// The version of the stored forecasts of `models`.
    #[tracing::instrument(skip_all, fields(db.operation = "live_forecasts_version"))]
    pub async fn live_forecasts_version(&self, models: &[String]) -> Result<LiveForecastsVersion> {
        sqlx::query_as!(
            LiveForecastsVersion,
            r#"
            SELECT MAX(made_at) AS made_at, COUNT(*) AS "rows!"
            FROM live_forecasts
            WHERE model = ANY($1)
            "#,
            models
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to check the published forecasts")
    }

    /// The version of the forecast suite's state.
    #[tracing::instrument(skip_all, fields(db.operation = "suite_version"))]
    pub async fn suite_version(&self) -> Result<SuiteVersion> {
        sqlx::query_as!(
            SuiteVersion,
            r#"
            SELECT
                s.retrain_requested_at,
                s.last_attempt_at,
                s.last_error,
                (SELECT MAX(updated_at) FROM forecast_models) AS models_updated_at,
                (SELECT COUNT(*) FROM forecast_models) AS "models!",
                (SELECT MAX(id) FROM benchmark_runs) AS latest_run,
                (SELECT MAX(selected_at) FROM forecast_selection) AS selected_at,
                (SELECT COUNT(*) FROM forecast_selection) AS "selections!"
            FROM ml_state s
            WHERE s.id = 1
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to check the forecast suite")
    }
}
