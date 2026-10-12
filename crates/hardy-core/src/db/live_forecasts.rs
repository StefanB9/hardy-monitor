//! The newest forecast of every model, published by the daemon for the GUI.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// One forecast value.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveForecast {
    /// Model key, or `selected` for the default forecast.
    pub model: String,
    /// End of the forecast slot.
    pub target: DateTime<Utc>,
    pub made_at: DateTime<Utc>,
    pub value: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
    /// For the default forecast: the model the value came from.
    pub source: Option<String>,
}

impl Database {
    /// Replaces all stored forecasts with `rows` (a model without a
    /// forecast now has none stored). Returns how many rows were stored.
    #[tracing::instrument(skip_all, fields(db.operation = "replace_live_forecasts", count = rows.len()))]
    pub async fn replace_live_forecasts(&self, rows: &[LiveForecast]) -> Result<u64> {
        let n = rows.len();
        let mut models: Vec<String> = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        let mut made = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        let mut lows: Vec<Option<f64>> = Vec::with_capacity(n);
        let mut highs: Vec<Option<f64>> = Vec::with_capacity(n);
        let mut sources: Vec<Option<String>> = Vec::with_capacity(n);
        for r in rows {
            models.push(r.model.clone());
            targets.push(r.target);
            made.push(r.made_at);
            values.push(r.value);
            lows.push(r.low);
            highs.push(r.high);
            sources.push(r.source.clone());
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin transaction")?;
        sqlx::query!("DELETE FROM live_forecasts")
            .execute(&mut *tx)
            .await
            .context("failed to clear old live forecasts")?;
        let result = sqlx::query!(
            r#"
            INSERT INTO live_forecasts (model, target, made_at, value, low, high, source)
            SELECT * FROM UNNEST(
                $1::text[], $2::timestamptz[], $3::timestamptz[], $4::float8[],
                $5::float8[], $6::float8[], $7::text[]
            )
            "#,
            &models,
            &targets,
            &made,
            &values,
            &lows as &[Option<f64>],
            &highs as &[Option<f64>],
            &sources as &[Option<String>]
        )
        .execute(&mut *tx)
        .await
        .context("failed to store live forecasts")?;
        tx.commit()
            .await
            .context("failed to commit live forecasts")?;
        Ok(result.rows_affected())
    }

    /// The stored forecasts of `models`, by model and target.
    #[tracing::instrument(skip(self), fields(db.operation = "live_forecasts"))]
    pub async fn live_forecasts(&self, models: &[String]) -> Result<Vec<LiveForecast>> {
        let rows = sqlx::query_as!(
            LiveForecast,
            r#"
            SELECT model, target, made_at, value, low, high, source
            FROM live_forecasts
            WHERE model = ANY($1::text[])
            ORDER BY model, target
            "#,
            models
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load live forecasts")?;
        Ok(rows)
    }
}
