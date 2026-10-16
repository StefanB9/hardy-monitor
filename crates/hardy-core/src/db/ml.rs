//! Coordination of the nightly forecasting run: requests from the GUI and
//! the outcome of the last attempt.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, Utc};

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

/// The running forecaster's claim on the database: only one forecaster may
/// run at a time. It is a row, not an advisory lock, so it works through
/// connection poolers (which share server sessions between clients).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecasterLease {
    holder: String,
}

impl ForecasterLease {
    /// Who holds it.
    pub fn holder(&self) -> &str {
        &self.holder
    }
}

/// `delta` as fractional seconds for an SQL interval.
fn interval_secs(delta: TimeDelta) -> f64 {
    #[allow(clippy::cast_precision_loss)] // milliseconds of a lease timeout
    let secs = delta.num_milliseconds() as f64 / 1000.0;
    secs
}

impl Database {
    /// Claims the forecaster lease for `holder`: free, already `holder`'s,
    /// or not renewed for longer than `stale_after` (its holder crashed).
    /// `None` while another forecaster holds it. Database time only, so the
    /// machines' clocks do not matter.
    #[tracing::instrument(skip(self), fields(db.operation = "claim_forecaster_lease"))]
    pub async fn claim_forecaster_lease(
        &self,
        holder: &str,
        stale_after: TimeDelta,
    ) -> Result<Option<ForecasterLease>> {
        let claimed = sqlx::query_scalar!(
            r#"
            INSERT INTO forecaster_lease (id, holder, renewed_at)
            VALUES (1, $1, clock_timestamp())
            ON CONFLICT (id) DO UPDATE
                SET holder = EXCLUDED.holder, renewed_at = EXCLUDED.renewed_at
                WHERE forecaster_lease.holder = EXCLUDED.holder
                   OR forecaster_lease.renewed_at
                      < clock_timestamp() - make_interval(secs => $2)
            RETURNING holder
            "#,
            holder,
            interval_secs(stale_after)
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to claim the forecaster lease")?;
        Ok(claimed.map(|holder| ForecasterLease { holder }))
    }

    /// Renews `lease`; `false` if another forecaster has taken it over.
    #[tracing::instrument(skip_all, fields(db.operation = "renew_forecaster_lease"))]
    pub async fn renew_forecaster_lease(&self, lease: &ForecasterLease) -> Result<bool> {
        let result = sqlx::query!(
            "UPDATE forecaster_lease SET renewed_at = clock_timestamp() WHERE id = 1 AND holder = \
             $1",
            lease.holder
        )
        .execute(&self.pool)
        .await
        .context("Failed to renew the forecaster lease")?;
        Ok(result.rows_affected() == 1)
    }

    /// Gives `lease` up, unless another forecaster has taken it over.
    #[tracing::instrument(skip_all, fields(db.operation = "release_forecaster_lease"))]
    pub async fn release_forecaster_lease(&self, lease: ForecasterLease) -> Result<()> {
        sqlx::query!(
            "DELETE FROM forecaster_lease WHERE id = 1 AND holder = $1",
            lease.holder
        )
        .execute(&self.pool)
        .await
        .context("Failed to release the forecaster lease")?;
        Ok(())
    }

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
