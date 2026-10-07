//! Forecasts the daemon made, and how accurate they turned out.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use chrono_tz::Tz;

use super::Database;

/// One stored forecast.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastLogEntry {
    pub made_at: DateTime<Utc>,
    pub target: DateTime<Utc>,
    pub horizon_hours: u32,
    /// Model that produced `predicted`; `None` for plain averages.
    pub model_id: Option<i64>,
    pub predicted: f64,
    pub low: f64,
    pub high: f64,
    /// Plain slot-average forecast for the same target.
    pub baseline: f64,
}

/// Accuracy of the forecasts for one gym-local day and horizon, over the
/// forecasts whose target has a reading.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonAccuracy {
    pub day: NaiveDate,
    pub horizon_hours: u32,
    /// Forecasts compared with a reading.
    pub scored: i64,
    /// Of those, made by a model (the rest fell back to averages).
    pub from_model: i64,
    /// Mean absolute error of what was forecast.
    pub forecast_mae: f64,
    /// Mean absolute error of plain averages on the same targets.
    pub baseline_mae: f64,
}

impl Database {
    /// Stores forecasts; ones already stored for the same time and horizon
    /// are kept. Returns how many were added.
    #[tracing::instrument(skip_all, fields(db.operation = "save_forecasts", count = entries.len()))]
    pub async fn save_forecasts(&self, entries: &[ForecastLogEntry]) -> Result<u64> {
        let n = entries.len();
        let (mut made, mut target, mut horizon) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let (mut model, mut predicted, mut low, mut high, mut baseline) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for e in entries {
            made.push(e.made_at);
            target.push(e.target);
            horizon.push(i16::try_from(e.horizon_hours).context("horizon out of range")?);
            model.push(e.model_id);
            predicted.push(e.predicted);
            low.push(e.low);
            high.push(e.high);
            baseline.push(e.baseline);
        }
        let result = sqlx::query!(
            r#"
            INSERT INTO forecast_log
                (made_at, target, horizon_hours, model_id, predicted, low, high, baseline)
            SELECT * FROM UNNEST(
                $1::timestamptz[], $2::timestamptz[], $3::int2[], $4::int8[],
                $5::float8[], $6::float8[], $7::float8[], $8::float8[]
            )
            ON CONFLICT (made_at, horizon_hours) DO NOTHING
            "#,
            &made,
            &target,
            &horizon,
            &model as &[Option<i64>],
            &predicted,
            &low,
            &high,
            &baseline
        )
        .execute(&self.pool)
        .await
        .context("failed to store forecasts")?;
        Ok(result.rows_affected())
    }

    /// Deletes forecasts made before `cutoff`.
    #[tracing::instrument(skip(self), fields(db.operation = "delete_forecasts_before"))]
    pub async fn delete_forecasts_before(&self, cutoff: DateTime<Utc>) -> Result<u64> {
        let result = sqlx::query!("DELETE FROM forecast_log WHERE made_at < $1", cutoff)
            .execute(&self.pool)
            .await
            .context("failed to delete old forecasts")?;
        Ok(result.rows_affected())
    }

    /// Accuracy per gym-local target day and horizon for targets in
    /// `[since, until)`, compared with measured or smoothed readings.
    #[tracing::instrument(skip(self), fields(db.operation = "forecast_accuracy"))]
    pub async fn forecast_accuracy(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
        tz: Tz,
    ) -> Result<Vec<HorizonAccuracy>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                (f.target AT TIME ZONE $3)::date AS "day!",
                f.horizon_hours AS "horizon_hours!",
                COUNT(*) AS "scored!",
                COUNT(f.model_id) AS "from_model!",
                AVG(ABS(f.predicted - o.percentage)) AS "forecast_mae!",
                AVG(ABS(f.baseline - o.percentage)) AS "baseline_mae!"
            FROM forecast_log f
            JOIN occupancy_logs o
              ON o.timestamp = f.target AND o.source IN ('measured', 'smoothed')
            WHERE f.target >= $1 AND f.target < $2
            GROUP BY 1, 2
            ORDER BY 1, 2
            "#,
            since,
            until,
            tz.name()
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to score forecasts")?;
        rows.into_iter()
            .map(|r| {
                Ok(HorizonAccuracy {
                    day: r.day,
                    horizon_hours: u32::try_from(r.horizon_hours)
                        .context("negative horizon in forecast_log")?,
                    scored: r.scored,
                    from_model: r.from_model,
                    forecast_mae: r.forecast_mae,
                    baseline_mae: r.baseline_mae,
                })
            })
            .collect()
    }
}
