//! The hour-ahead default forecast of every past slot, for the history
//! chart: backfilled once over all history, then extended nightly.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// The default forecast of one past slot.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastHistoryEntry {
    /// End of the slot.
    pub target: DateTime<Utc>,
    /// The full hour it was made at: the last one at least an hour before.
    pub made_at: DateTime<Utc>,
    /// Key of the model the value came from.
    pub model: String,
    pub value: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

impl Database {
    /// Stores past forecasts; slots already stored keep their forecast.
    /// Returns how many were added.
    #[tracing::instrument(skip_all, fields(db.operation = "save_forecast_history", count = entries.len()))]
    pub async fn save_forecast_history(&self, entries: &[ForecastHistoryEntry]) -> Result<u64> {
        let n = entries.len();
        let mut targets = Vec::with_capacity(n);
        let mut made = Vec::with_capacity(n);
        let mut models: Vec<String> = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        let mut lows: Vec<Option<f64>> = Vec::with_capacity(n);
        let mut highs: Vec<Option<f64>> = Vec::with_capacity(n);
        for e in entries {
            targets.push(e.target);
            made.push(e.made_at);
            models.push(e.model.clone());
            values.push(e.value);
            lows.push(e.low);
            highs.push(e.high);
        }
        let result = sqlx::query!(
            r#"
            INSERT INTO forecast_history (target, made_at, model, value, low, high)
            SELECT * FROM UNNEST(
                $1::timestamptz[], $2::timestamptz[], $3::text[], $4::float8[],
                $5::float8[], $6::float8[]
            )
            ON CONFLICT (target) DO NOTHING
            "#,
            &targets,
            &made,
            &models,
            &values,
            &lows as &[Option<f64>],
            &highs as &[Option<f64>]
        )
        .execute(&self.pool)
        .await
        .context("failed to store the forecast history")?;
        Ok(result.rows_affected())
    }

    /// Past forecasts for slots in `[since, until)`, oldest first.
    #[tracing::instrument(skip(self), fields(db.operation = "forecast_history"))]
    pub async fn forecast_history(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Result<Vec<ForecastHistoryEntry>> {
        sqlx::query_as!(
            ForecastHistoryEntry,
            r#"
            SELECT target, made_at, model, value, low, high
            FROM forecast_history
            WHERE target >= $1 AND target < $2
            ORDER BY target
            "#,
            since,
            until
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to read the forecast history")
    }

    /// Whether any past forecast is stored (the backfill has run).
    #[tracing::instrument(skip(self), fields(db.operation = "has_forecast_history"))]
    pub async fn has_forecast_history(&self) -> Result<bool> {
        sqlx::query_scalar!(r#"SELECT EXISTS (SELECT 1 FROM forecast_history) AS "exists!""#)
            .fetch_one(&self.pool)
            .await
            .context("failed to look up the forecast history")
    }
}
