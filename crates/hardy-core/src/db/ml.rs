//! Storage for trained forecasting models and training coordination.
//!
//! Core stores model artifacts as opaque bytes; encoding them is
//! `hardy-ml`'s concern.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// A model to store.
#[derive(Debug, Clone, Copy)]
pub struct NewModel<'a> {
    pub trained_at: DateTime<Utc>,
    pub feature_version: u32,
    pub algorithm: &'a str,
    pub training_samples: usize,
    pub holdout_mae: f64,
    pub baseline_mae: f64,
    pub tuned: bool,
    pub bytes: &'a [u8],
}

/// Metadata of a stored model (without the model bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: i64,
    pub trained_at: DateTime<Utc>,
    pub feature_version: u32,
    pub algorithm: String,
    pub training_samples: usize,
    pub holdout_mae: f64,
    pub baseline_mae: f64,
    pub tuned: bool,
}

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
    /// Stores a model and deletes all but the newest `keep` models of the
    /// same feature version, in one transaction. Returns the new id.
    #[tracing::instrument(skip_all, fields(
        db.operation = "save_model",
        feature_version = model.feature_version,
        bytes = model.bytes.len(),
    ))]
    pub async fn save_model(&self, model: &NewModel<'_>, keep: usize) -> Result<i64> {
        let feature_version =
            i32::try_from(model.feature_version).context("feature version out of range")?;
        let samples =
            i32::try_from(model.training_samples).context("training sample count out of range")?;
        let keep = i64::try_from(keep).context("keep count out of range")?;

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin transaction")?;
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO ml_models
                (trained_at, feature_version, algorithm, training_samples,
                 holdout_mae, baseline_mae, tuned, model)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            "#,
            model.trained_at,
            feature_version,
            model.algorithm,
            samples,
            model.holdout_mae,
            model.baseline_mae,
            model.tuned,
            model.bytes
        )
        .fetch_one(&mut *tx)
        .await
        .context("Failed to insert model")?;

        sqlx::query!(
            r#"
            DELETE FROM ml_models
            WHERE feature_version = $1
              AND id NOT IN (
                  SELECT id FROM ml_models
                  WHERE feature_version = $1
                  ORDER BY trained_at DESC, id DESC
                  LIMIT $2
              )
            "#,
            feature_version,
            keep
        )
        .execute(&mut *tx)
        .await
        .context("Failed to prune old models")?;

        tx.commit().await.context("Failed to commit model")?;
        Ok(id)
    }

    /// Metadata of the newest model with this feature version.
    #[tracing::instrument(skip_all, fields(db.operation = "latest_model_info", feature_version))]
    pub async fn latest_model_info(&self, feature_version: u32) -> Result<Option<ModelInfo>> {
        let feature_version =
            i32::try_from(feature_version).context("feature version out of range")?;
        let row = sqlx::query!(
            r#"
            SELECT id, trained_at, feature_version, algorithm, training_samples,
                   holdout_mae, baseline_mae, tuned
            FROM ml_models
            WHERE feature_version = $1
            ORDER BY trained_at DESC, id DESC
            LIMIT 1
            "#,
            feature_version
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to load model info")?;

        row.map(|r| {
            Ok(ModelInfo {
                id: r.id,
                trained_at: r.trained_at,
                feature_version: u32::try_from(r.feature_version)
                    .context("stored feature version is negative")?,
                algorithm: r.algorithm,
                training_samples: usize::try_from(r.training_samples)
                    .context("stored sample count is negative")?,
                holdout_mae: r.holdout_mae,
                baseline_mae: r.baseline_mae,
                tuned: r.tuned,
            })
        })
        .transpose()
    }

    /// The stored bytes of a model.
    #[tracing::instrument(skip_all, fields(db.operation = "load_model", id))]
    pub async fn load_model(&self, id: i64) -> Result<Vec<u8>> {
        sqlx::query_scalar!("SELECT model FROM ml_models WHERE id = $1", id)
            .fetch_one(&self.pool)
            .await
            .with_context(|| format!("Failed to load model {id}"))
    }

    /// Ids of all stored models, newest first.
    #[tracing::instrument(skip_all, fields(db.operation = "model_ids"))]
    pub async fn model_ids(&self) -> Result<Vec<i64>> {
        sqlx::query_scalar!("SELECT id FROM ml_models ORDER BY trained_at DESC, id DESC")
            .fetch_all(&self.pool)
            .await
            .context("Failed to list models")
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
