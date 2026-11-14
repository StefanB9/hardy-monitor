//! Every stored benchmark run: each model's scores and the selection, night
//! by night.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::{BenchmarkScore, Database, benchmark::count};

/// One nightly benchmark run's scores and selection.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkNight {
    pub run_id: i64,
    pub finished_at: DateTime<Utc>,
    /// The model selected per horizon bucket, by bucket.
    pub selection: Vec<(i16, String)>,
    /// Every model's scores, by model and bucket.
    pub scores: Vec<BenchmarkScore>,
}

impl Database {
    /// The benchmark runs finished at or after `since`, oldest first.
    #[tracing::instrument(skip(self), fields(db.operation = "benchmark_history"))]
    pub async fn benchmark_history(&self, since: DateTime<Utc>) -> Result<Vec<BenchmarkNight>> {
        let runs = sqlx::query!(
            "SELECT id, finished_at FROM benchmark_runs WHERE finished_at >= $1 ORDER BY id",
            since
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load the benchmark runs")?;
        let mut nights: BTreeMap<i64, BenchmarkNight> = runs
            .into_iter()
            .map(|r| {
                (
                    r.id,
                    BenchmarkNight {
                        run_id: r.id,
                        finished_at: r.finished_at,
                        selection: Vec::new(),
                        scores: Vec::new(),
                    },
                )
            })
            .collect();

        let selection = sqlx::query!(
            r#"
            SELECT s.run_id, s.bucket, s.model
            FROM benchmark_selection s
            JOIN benchmark_runs r ON r.id = s.run_id
            WHERE r.finished_at >= $1
            ORDER BY s.run_id, s.bucket
            "#,
            since
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load the benchmark selections")?;
        for row in selection {
            if let Some(night) = nights.get_mut(&row.run_id) {
                night.selection.push((row.bucket, row.model));
            }
        }

        let scores = sqlx::query!(
            r#"
            SELECT s.run_id, s.model, s.bucket, s.mae, s.rmse, s.mase, s.coverage, s.pinball,
                   s.errors, s.interval_low, s.interval_high
            FROM benchmark_scores s
            JOIN benchmark_runs r ON r.id = s.run_id
            WHERE r.finished_at >= $1
            ORDER BY s.run_id, s.model, s.bucket
            "#,
            since
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load the benchmark score history")?;
        for row in scores {
            if let Some(night) = nights.get_mut(&row.run_id) {
                night.scores.push(BenchmarkScore {
                    errors: count(row.errors, "errors")?,
                    model: row.model,
                    bucket: row.bucket,
                    mae: row.mae,
                    rmse: row.rmse,
                    mase: row.mase,
                    coverage: row.coverage,
                    pinball: row.pinball,
                    interval: row.interval_low.zip(row.interval_high),
                });
            }
        }
        Ok(nights.into_values().collect())
    }
}
