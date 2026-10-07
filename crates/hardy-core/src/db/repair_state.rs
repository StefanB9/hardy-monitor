//! Progress of the nightly automatic repair.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};

use super::Database;

/// Where the automatic repair stands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairState {
    /// Last gym-local day repaired; `None` before the first run.
    pub repaired_through: Option<NaiveDate>,
    pub last_attempt_at: Option<DateTime<Utc>>,
    /// Why the last attempt failed; `None` after success.
    pub last_error: Option<String>,
}

impl Database {
    /// Reads the repair progress.
    #[tracing::instrument(skip_all, fields(db.operation = "get_repair_state"))]
    pub async fn get_repair_state(&self) -> Result<RepairState> {
        sqlx::query_as!(
            RepairState,
            "SELECT repaired_through, last_attempt_at, last_error FROM repair_state WHERE id = 1"
        )
        .fetch_one(&self.pool)
        .await
        .context("failed to read repair state")
    }

    /// Records that every day up to `through` is repaired.
    #[tracing::instrument(skip(self), fields(db.operation = "record_repair_success"))]
    pub async fn record_repair_success(&self, through: NaiveDate, at: DateTime<Utc>) -> Result<()> {
        sqlx::query!(
            "UPDATE repair_state SET repaired_through = $1, last_attempt_at = $2, last_error = \
             NULL WHERE id = 1",
            through,
            at
        )
        .execute(&self.pool)
        .await
        .context("failed to record repair progress")?;
        Ok(())
    }

    /// Records a failed attempt, keeping the progress made so far.
    #[tracing::instrument(skip(self), fields(db.operation = "record_repair_failure"))]
    pub async fn record_repair_failure(&self, at: DateTime<Utc>, error: &str) -> Result<()> {
        sqlx::query!(
            "UPDATE repair_state SET last_attempt_at = $1, last_error = $2 WHERE id = 1",
            at,
            error
        )
        .execute(&self.pool)
        .await
        .context("failed to record repair failure")?;
        Ok(())
    }
}
