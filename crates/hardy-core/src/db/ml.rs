//! Coordination of the nightly forecasting run: requests from the GUI and
//! the outcome of the last attempt.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// Training coordination state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlState {
    /// Set by the GUI's "Train model" button; cleared when training starts.
    pub retrain_requested_at: Option<DateTime<Utc>>,
    pub last_attempt_at: Option<DateTime<Utc>>,
    /// Why the last attempt produced no model; `None` after success.
    pub last_error: Option<String>,
}

impl Database {
    /// Training coordination state (retrain requests, last attempt).
    #[tracing::instrument(skip_all, fields(db.operation = "get_ml_state"))]
    pub async fn get_ml_state(&self) -> Result<MlState> {
        let row = sqlx::query!(
            "SELECT retrain_requested_at, last_attempt_at, last_error FROM ml_state WHERE id = 1"
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to load ML state")?;
        Ok(MlState {
            retrain_requested_at: row.retrain_requested_at,
            last_attempt_at: row.last_attempt_at,
            last_error: row.last_error,
        })
    }

    /// Asks the daemon to retrain soon.
    #[tracing::instrument(skip_all, fields(db.operation = "request_retrain"))]
    pub async fn request_retrain(&self, now: DateTime<Utc>) -> Result<()> {
        sqlx::query!(
            "UPDATE ml_state SET retrain_requested_at = $1 WHERE id = 1",
            now
        )
        .execute(&self.pool)
        .await
        .context("Failed to request retraining")?;
        Ok(())
    }

    /// Records that training starts now; consumes requests made up to now.
    #[tracing::instrument(skip_all, fields(db.operation = "start_training_attempt"))]
    pub async fn start_training_attempt(&self, now: DateTime<Utc>) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE ml_state
            SET last_attempt_at = $1,
                retrain_requested_at = CASE
                    WHEN retrain_requested_at <= $1 THEN NULL
                    ELSE retrain_requested_at
                END
            WHERE id = 1
            "#,
            now
        )
        .execute(&self.pool)
        .await
        .context("Failed to record training start")?;
        Ok(())
    }

    /// Records the outcome of the current attempt (`None` = success).
    #[tracing::instrument(skip_all, fields(db.operation = "finish_training_attempt"))]
    pub async fn finish_training_attempt(&self, error: Option<&str>) -> Result<()> {
        sqlx::query!("UPDATE ml_state SET last_error = $1 WHERE id = 1", error)
            .execute(&self.pool)
            .await
            .context("Failed to record training result")?;
        Ok(())
    }
}
