//! The hour-ahead default forecast of every past slot, for the history
//! chart: backfilled once over all history, then extended nightly.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use chrono_tz::Tz;

use super::Database;
use crate::accuracy::Pair;

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

/// A past default forecast scored against the slot's reading.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredForecast {
    pub target: DateTime<Utc>,
    pub made_at: DateTime<Utc>,
    /// Key of the model the value came from.
    pub model: String,
    pub pair: Pair,
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

    /// Past default forecasts for slots in `[since, until)` that have a
    /// measured or smoothed reading, with plain averages and last week's
    /// reading for the same gym-local slot; oldest first.
    #[tracing::instrument(skip(self), fields(db.operation = "scored_forecast_history"))]
    pub async fn scored_forecast_history(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
        tz: Tz,
    ) -> Result<Vec<ScoredForecast>> {
        let rows = sqlx::query!(
            r#"
            SELECT h.target, h.made_at, h.model, h.value, h.low, h.high,
                   o.percentage AS actual,
                   (SELECT AVG(p.percentage) FROM occupancy_logs p
                    WHERE p.source IN ('measured', 'smoothed')
                      AND p.timestamp IN (
                          SELECT ((h.target AT TIME ZONE $3) - w * INTERVAL '7 days')
                                 AT TIME ZONE $3
                          FROM generate_series(1, 4) AS w)) AS baseline,
                   (SELECT p.percentage FROM occupancy_logs p
                    WHERE p.source IN ('measured', 'smoothed')
                      AND p.timestamp = ((h.target AT TIME ZONE $3) - INTERVAL '7 days')
                                        AT TIME ZONE $3) AS last_week
            FROM forecast_history h
            JOIN occupancy_logs o
              ON o.timestamp = h.target AND o.source IN ('measured', 'smoothed')
            WHERE h.target >= $1 AND h.target < $2
            ORDER BY h.target
            "#,
            since,
            until,
            tz.name()
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to score the forecast history")?;
        Ok(rows
            .into_iter()
            .map(|r| ScoredForecast {
                target: r.target,
                made_at: r.made_at,
                model: r.model,
                pair: Pair {
                    actual: r.actual,
                    forecast: r.value,
                    band: r.low.zip(r.high),
                    baseline: r.baseline,
                    last_week: r.last_week,
                },
            })
            .collect())
    }
}
