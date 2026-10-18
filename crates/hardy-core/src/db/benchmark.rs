//! Nightly benchmark results and the model selected per horizon bucket.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};

use super::Database;

/// One model's part in a benchmark run.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkModel {
    pub model: String,
    /// Origins it forecast from.
    pub origins: u32,
    /// Backtest days it could be fitted on.
    pub fitted_days: u32,
    /// Forecast from enough origins to be selected.
    pub eligible: bool,
    pub failure: Option<String>,
}

/// One model's score in one horizon bucket (0 = all horizons, 1 = next
/// hour, 2 = 1–3 h, 3 = 3–24 h).
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkScore {
    pub model: String,
    pub bucket: i16,
    pub mae: f64,
    pub rmse: f64,
    pub mase: Option<f64>,
    pub coverage: Option<f64>,
    pub pinball: Option<f64>,
    pub errors: u32,
    /// Conformal interval offsets (low, high) for live forecasts.
    pub interval: Option<(f64, f64)>,
}

/// A benchmark run with all its results.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRun {
    /// Assigned when stored; ignored when saving.
    pub id: i64,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub first_day: Option<NaiveDate>,
    pub last_day: Option<NaiveDate>,
    pub days: u32,
    pub origins: u32,
    pub models: Vec<BenchmarkModel>,
    pub scores: Vec<BenchmarkScore>,
}

fn int(value: u32, what: &str) -> Result<i32> {
    i32::try_from(value).with_context(|| format!("{what} out of range"))
}

fn count(value: i32, what: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("negative {what}"))
}

/// Stores each model's benchmark row of run `id`.
async fn insert_models(
    conn: &mut sqlx::PgConnection,
    id: i64,
    models: &[BenchmarkModel],
) -> Result<()> {
    let model_origins = models
        .iter()
        .map(|m| int(m.origins, "origins"))
        .collect::<Result<Vec<_>>>()?;
    let model_fitted_days = models
        .iter()
        .map(|m| int(m.fitted_days, "fitted days"))
        .collect::<Result<Vec<_>>>()?;
    sqlx::query!(
        r#"
        INSERT INTO benchmark_models (run_id, model, origins, fitted_days, eligible, failure)
        SELECT $1, * FROM UNNEST($2::text[], $3::int4[], $4::int4[], $5::bool[], $6::text[])
        "#,
        id,
        &models.iter().map(|m| m.model.clone()).collect::<Vec<_>>(),
        &model_origins,
        &model_fitted_days,
        &models.iter().map(|m| m.eligible).collect::<Vec<_>>(),
        &models.iter().map(|m| m.failure.clone()).collect::<Vec<_>>() as &[Option<String>]
    )
    .execute(&mut *conn)
    .await
    .context("failed to store the models' benchmark")?;
    Ok(())
}

/// Stores each model's scores per horizon bucket of run `id`.
async fn insert_scores(
    conn: &mut sqlx::PgConnection,
    id: i64,
    scores: &[BenchmarkScore],
) -> Result<()> {
    let score_errors = scores
        .iter()
        .map(|s| int(s.errors, "errors"))
        .collect::<Result<Vec<_>>>()?;
    sqlx::query!(
        r#"
        INSERT INTO benchmark_scores
            (run_id, model, bucket, mae, rmse, mase, coverage, pinball, errors,
             interval_low, interval_high)
        SELECT $1, * FROM UNNEST(
            $2::text[], $3::int2[], $4::float8[], $5::float8[], $6::float8[], $7::float8[],
            $8::float8[], $9::int4[], $10::float8[], $11::float8[]
        )
        "#,
        id,
        &scores.iter().map(|s| s.model.clone()).collect::<Vec<_>>(),
        &scores.iter().map(|s| s.bucket).collect::<Vec<_>>(),
        &scores.iter().map(|s| s.mae).collect::<Vec<_>>(),
        &scores.iter().map(|s| s.rmse).collect::<Vec<_>>(),
        &scores.iter().map(|s| s.mase).collect::<Vec<_>>() as &[Option<f64>],
        &scores.iter().map(|s| s.coverage).collect::<Vec<_>>() as &[Option<f64>],
        &scores.iter().map(|s| s.pinball).collect::<Vec<_>>() as &[Option<f64>],
        &score_errors,
        &scores
            .iter()
            .map(|s| s.interval.map(|(low, _)| low))
            .collect::<Vec<_>>() as &[Option<f64>],
        &scores
            .iter()
            .map(|s| s.interval.map(|(_, high)| high))
            .collect::<Vec<_>>() as &[Option<f64>]
    )
    .execute(&mut *conn)
    .await
    .context("failed to store the benchmark scores")?;
    Ok(())
}

/// Replaces the selection with `selection`, made from run `id`.
async fn replace_selection(
    conn: &mut sqlx::PgConnection,
    id: i64,
    selection: &[(i16, String)],
    selected_at: DateTime<Utc>,
) -> Result<()> {
    sqlx::query!("DELETE FROM forecast_selection")
        .execute(&mut *conn)
        .await
        .context("failed to clear the selection")?;
    sqlx::query!(
        r#"
        INSERT INTO forecast_selection (bucket, model, run_id, selected_at)
        SELECT bucket, model, $3, $4 FROM UNNEST($1::int2[], $2::text[]) AS s(bucket, model)
        "#,
        &selection
            .iter()
            .map(|(bucket, _)| *bucket)
            .collect::<Vec<_>>(),
        &selection
            .iter()
            .map(|(_, model)| model.clone())
            .collect::<Vec<_>>(),
        id,
        selected_at
    )
    .execute(&mut *conn)
    .await
    .context("failed to store the selection")?;
    Ok(())
}

impl Database {
    /// Stores a run and the selection made from it, then deletes all but
    /// the newest `keep` runs, in one transaction. Returns the run's id.
    #[tracing::instrument(skip_all, fields(
        db.operation = "save_benchmark",
        models = run.models.len(),
        scores = run.scores.len(),
    ))]
    pub async fn save_benchmark(
        &self,
        run: &BenchmarkRun,
        selection: &[(i16, String)],
        keep: usize,
    ) -> Result<i64> {
        let keep = i64::try_from(keep).context("keep count out of range")?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin transaction")?;
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO benchmark_runs (started_at, finished_at, first_day, last_day, days, origins)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            "#,
            run.started_at,
            run.finished_at,
            run.first_day,
            run.last_day,
            int(run.days, "days")?,
            int(run.origins, "origins")?
        )
        .fetch_one(&mut *tx)
        .await
        .context("failed to store benchmark run")?;

        // One statement per table (UNNEST) instead of one per row: the
        // database may be a network round trip away.
        insert_models(&mut tx, id, &run.models).await?;
        insert_scores(&mut tx, id, &run.scores).await?;
        replace_selection(&mut tx, id, selection, run.finished_at).await?;

        sqlx::query!(
            r#"
            DELETE FROM benchmark_runs
            WHERE id NOT IN (SELECT id FROM benchmark_runs ORDER BY id DESC LIMIT $1)
            "#,
            keep
        )
        .execute(&mut *tx)
        .await
        .context("failed to prune old benchmark runs")?;
        tx.commit().await.context("failed to commit benchmark")?;
        Ok(id)
    }

    /// The newest benchmark run with its results.
    #[tracing::instrument(skip(self), fields(db.operation = "latest_benchmark"))]
    pub async fn latest_benchmark(&self) -> Result<Option<BenchmarkRun>> {
        let Some(run) = sqlx::query!(
            r#"
            SELECT id, started_at, finished_at, first_day, last_day, days, origins
            FROM benchmark_runs
            ORDER BY id DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.pool)
        .await
        .context("failed to load the latest benchmark")?
        else {
            return Ok(None);
        };

        let models = sqlx::query!(
            r#"
            SELECT model, origins, fitted_days, eligible, failure
            FROM benchmark_models
            WHERE run_id = $1
            ORDER BY model
            "#,
            run.id
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load benchmarked models")?
        .into_iter()
        .map(|r| {
            Ok(BenchmarkModel {
                origins: count(r.origins, "origins")?,
                fitted_days: count(r.fitted_days, "fitted days")?,
                model: r.model,
                eligible: r.eligible,
                failure: r.failure,
            })
        })
        .collect::<Result<Vec<_>>>()?;

        let scores = sqlx::query!(
            r#"
            SELECT model, bucket, mae, rmse, mase, coverage, pinball, errors,
                   interval_low, interval_high
            FROM benchmark_scores
            WHERE run_id = $1
            ORDER BY model, bucket
            "#,
            run.id
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load benchmark scores")?
        .into_iter()
        .map(|r| {
            Ok(BenchmarkScore {
                errors: count(r.errors, "errors")?,
                model: r.model,
                bucket: r.bucket,
                mae: r.mae,
                rmse: r.rmse,
                mase: r.mase,
                coverage: r.coverage,
                pinball: r.pinball,
                interval: r.interval_low.zip(r.interval_high),
            })
        })
        .collect::<Result<Vec<_>>>()?;

        Ok(Some(BenchmarkRun {
            id: run.id,
            started_at: run.started_at,
            finished_at: run.finished_at,
            first_day: run.first_day,
            last_day: run.last_day,
            days: count(run.days, "days")?,
            origins: count(run.origins, "origins")?,
            models,
            scores,
        }))
    }

    /// The selected model per horizon bucket, by bucket.
    #[tracing::instrument(skip(self), fields(db.operation = "forecast_selection"))]
    pub async fn forecast_selection(&self) -> Result<Vec<(i16, String)>> {
        let rows = sqlx::query!("SELECT bucket, model FROM forecast_selection ORDER BY bucket")
            .fetch_all(&self.pool)
            .await
            .context("failed to load the forecast selection")?;
        Ok(rows.into_iter().map(|r| (r.bucket, r.model)).collect())
    }
}
