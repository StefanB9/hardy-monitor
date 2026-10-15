//! The forecasting suite's model states: each model's newest fit (or why
//! there is none). Core stores the fitted state as opaque bytes; encoding it
//! is `hardy-ml`'s concern.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use super::Database;

/// Whether a model has a usable fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    /// Fitted on the newest data.
    Fitted,
    /// Not enough history yet.
    Waiting,
    /// Fitting or forecasting failed.
    Failed,
}

impl FitStatus {
    /// Stored text.
    pub fn as_str(self) -> &'static str {
        match self {
            FitStatus::Fitted => "fitted",
            FitStatus::Waiting => "waiting",
            FitStatus::Failed => "failed",
        }
    }

    /// The status stored as `text`.
    pub fn parse(text: &str) -> Option<Self> {
        [FitStatus::Fitted, FitStatus::Waiting, FitStatus::Failed]
            .into_iter()
            .find(|s| s.as_str() == text)
    }
}

/// A model's new state.
#[derive(Debug, Clone, Copy)]
pub struct ModelUpdate<'a> {
    pub model: &'a str,
    pub status: FitStatus,
    pub message: Option<&'a str>,
    pub fitted_at: Option<DateTime<Utc>>,
    /// Readings the fit used.
    pub data_from: Option<DateTime<Utc>>,
    pub data_until: Option<DateTime<Utc>>,
    /// The fitted state as (format version, bytes).
    pub artifact: Option<(u32, &'a [u8])>,
}

/// A stored model state, without the fitted bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRecord {
    pub model: String,
    pub status: FitStatus,
    pub message: Option<String>,
    pub fitted_at: Option<DateTime<Utc>>,
    pub data_from: Option<DateTime<Utc>>,
    pub data_until: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

impl Database {
    /// Stores the models' new states (replacing earlier ones) in one
    /// transaction.
    #[tracing::instrument(skip_all, fields(db.operation = "save_model_states", count = updates.len()))]
    pub async fn save_model_states(
        &self,
        updates: &[ModelUpdate<'_>],
        now: DateTime<Utc>,
    ) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin transaction")?;
        for update in updates {
            let (version, bytes) = match update.artifact {
                Some((version, bytes)) => (
                    Some(i32::try_from(version).context("artifact version out of range")?),
                    Some(bytes),
                ),
                None => (None, None),
            };
            sqlx::query!(
                r#"
                INSERT INTO forecast_models
                    (model, status, message, fitted_at, data_from, data_until,
                     artifact_version, artifact, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (model) DO UPDATE SET
                    status = EXCLUDED.status,
                    message = EXCLUDED.message,
                    fitted_at = EXCLUDED.fitted_at,
                    data_from = EXCLUDED.data_from,
                    data_until = EXCLUDED.data_until,
                    artifact_version = EXCLUDED.artifact_version,
                    artifact = EXCLUDED.artifact,
                    updated_at = EXCLUDED.updated_at
                "#,
                update.model,
                update.status.as_str(),
                update.message,
                update.fitted_at,
                update.data_from,
                update.data_until,
                version,
                bytes,
                now
            )
            .execute(&mut *tx)
            .await
            .with_context(|| format!("failed to store the state of model {}", update.model))?;
        }
        tx.commit().await.context("failed to commit model states")?;
        Ok(())
    }

    /// Every stored model state, by model key.
    #[tracing::instrument(skip(self), fields(db.operation = "model_records"))]
    pub async fn model_records(&self) -> Result<Vec<ModelRecord>> {
        let rows = sqlx::query!(
            r#"
            SELECT model, status, message, fitted_at, data_from, data_until, updated_at
            FROM forecast_models
            ORDER BY model
            "#
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load model states")?;
        rows.into_iter()
            .map(|r| {
                Ok(ModelRecord {
                    status: FitStatus::parse(&r.status)
                        .with_context(|| format!("unknown status {}", r.status))?,
                    model: r.model,
                    message: r.message,
                    fitted_at: r.fitted_at,
                    data_from: r.data_from,
                    data_until: r.data_until,
                    updated_at: r.updated_at,
                })
            })
            .collect()
    }

    /// Fitted states stored in format `version`, as (model key, bytes).
    #[tracing::instrument(skip(self), fields(db.operation = "model_artifacts"))]
    pub async fn model_artifacts(&self, version: u32) -> Result<Vec<(String, Vec<u8>)>> {
        let version = i32::try_from(version).context("artifact version out of range")?;
        let rows = sqlx::query!(
            r#"
            SELECT model, artifact AS "artifact!"
            FROM forecast_models
            WHERE artifact IS NOT NULL AND artifact_version = $1
            ORDER BY model
            "#,
            version
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load fitted models")?;
        Ok(rows.into_iter().map(|r| (r.model, r.artifact)).collect())
    }
}
