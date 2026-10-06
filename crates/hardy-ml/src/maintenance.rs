//! Keeps a current model in the database: decides when to retrain, trains in
//! the background and stores the result. Driven by the daemon each tick.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule, MlConfig,
    db::{Database, NewModel},
};
use tokio::task::JoinHandle;

use crate::{
    error::MlError,
    features::FEATURE_VERSION,
    history::History,
    model::{Algorithm, RfParams},
    retrain::{RetrainReason, retrain_due, tuning_for},
    training::{ModelArtifact, TrainOptions, train},
};

/// How many models per feature version are kept in the database.
pub const MODELS_KEPT: usize = 3;

/// Result of one training run.
#[derive(Debug)]
pub enum TrainingOutcome {
    Stored {
        id: i64,
        artifact: Box<ModelArtifact>,
    },
    /// Training ran but produced no acceptable model (recorded in
    /// `ml_state.last_error`).
    Rejected(MlError),
}

/// Background training coordinator.
#[derive(Debug)]
pub struct ModelMaintenance {
    config: MlConfig,
    schedule: GymSchedule,
    previous_params: Option<RfParams>,
    running: Option<JoinHandle<Result<TrainingOutcome>>>,
}

impl ModelMaintenance {
    pub fn new(config: MlConfig, schedule: GymSchedule) -> Self {
        Self {
            config,
            schedule,
            previous_params: None,
            running: None,
        }
    }

    /// Reads the newest stored model's hyperparameters so nightly runs can
    /// reuse them.
    #[tracing::instrument(skip_all)]
    pub async fn load_previous(&mut self, db: &Database) -> Result<()> {
        if let Some(info) = db.latest_model_info(FEATURE_VERSION).await? {
            let bytes = db.load_model(info.id).await?;
            match ModelArtifact::from_bytes(&bytes) {
                Ok(artifact) => self.previous_params = artifact.model.rf_params(),
                Err(e) => tracing::warn!(error = %e, id = info.id, "stored model unreadable"),
            }
        }
        Ok(())
    }

    /// Whether a training run is in progress.
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Collects a finished run and starts a new one if one is due. Never
    /// blocks on training.
    #[tracing::instrument(skip_all)]
    pub async fn tick(&mut self, db: &Database, now: DateTime<Utc>) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        if let Some(handle) = self.running.take_if(|h| h.is_finished()) {
            let outcome = handle.await.context("training task panicked")?;
            self.record(outcome);
        }
        if self.running.is_some() {
            return Ok(());
        }

        let latest = db.latest_model_info(FEATURE_VERSION).await?;
        let state = db.get_ml_state().await?;
        let Some(reason) = retrain_due(now, &self.schedule, latest.map(|m| m.trained_at), &state)
        else {
            return Ok(());
        };
        tracing::info!(?reason, "starting model training");

        let db = db.clone();
        let config = self.config.clone();
        let schedule = self.schedule.clone();
        let previous = self.previous_params;
        self.running = Some(tokio::spawn(async move {
            run_training(&db, &config, &schedule, reason, previous, now).await
        }));
        Ok(())
    }

    /// Runs one training to completion (used at startup and in tests).
    pub async fn train_now(
        &mut self,
        db: &Database,
        reason: RetrainReason,
        now: DateTime<Utc>,
    ) -> Result<TrainingOutcome> {
        let outcome = run_training(
            db,
            &self.config,
            &self.schedule,
            reason,
            self.previous_params,
            now,
        )
        .await?;
        if let TrainingOutcome::Stored { artifact, .. } = &outcome {
            self.previous_params = artifact.model.rf_params();
        }
        Ok(outcome)
    }

    fn record(&mut self, outcome: Result<TrainingOutcome>) {
        match outcome {
            Ok(TrainingOutcome::Stored { id, artifact }) => {
                tracing::info!(
                    id,
                    holdout_mae = artifact.metrics.holdout_mae,
                    baseline_mae = artifact.metrics.baseline_mae,
                    tuned = artifact.tuned,
                    "new model stored"
                );
                self.previous_params = artifact.model.rf_params();
            }
            Ok(TrainingOutcome::Rejected(e)) => {
                tracing::warn!(error = %e, "training produced no model");
            }
            Err(e) => tracing::error!(error = %e, "training failed"),
        }
    }
}

#[tracing::instrument(skip(db, config, schedule, previous))]
async fn run_training(
    db: &Database,
    config: &MlConfig,
    schedule: &GymSchedule,
    reason: RetrainReason,
    previous: Option<RfParams>,
    now: DateTime<Utc>,
) -> Result<TrainingOutcome> {
    db.start_training_attempt(now).await?;

    let result = train_and_store(db, config, schedule, reason, previous, now).await;
    let error = match &result {
        Ok(TrainingOutcome::Stored { .. }) => None,
        Ok(TrainingOutcome::Rejected(e)) => Some(e.to_string()),
        Err(e) => Some(format!("{e:#}")),
    };
    db.finish_training_attempt(error.as_deref()).await?;
    result
}

async fn train_and_store(
    db: &Database,
    config: &MlConfig,
    schedule: &GymSchedule,
    reason: RetrainReason,
    previous: Option<RfParams>,
    now: DateTime<Utc>,
) -> Result<TrainingOutcome> {
    let logs = db
        .get_history_range(now - TimeDelta::days(config.training_window_days), now)
        .await?;
    let history = History::from_logs(&logs);
    let options = TrainOptions {
        algorithm: Algorithm::from(&config.algorithm),
        tuning: tuning_for(reason, schedule, previous),
        max_hours_ahead: u32::try_from(config.prediction_horizon_hours)
            .context("prediction_horizon_hours out of range")?,
        min_samples: config.min_samples_for_training,
        holdout_days: config.holdout_days,
    };

    // SAFETY: training is CPU-bound for seconds to minutes; running it on the
    // blocking pool keeps the daemon's fetch loop responsive.
    let schedule_for_training = schedule.clone();
    let trained =
        tokio::task::spawn_blocking(move || train(&history, &schedule_for_training, &options, now))
            .await
            .context("training task panicked")?;

    let artifact = match trained {
        Ok(artifact) => artifact,
        Err(e) => return Ok(TrainingOutcome::Rejected(e)),
    };
    let bytes = artifact.to_bytes()?;
    let id = db
        .save_model(
            &NewModel {
                trained_at: artifact.trained_at,
                feature_version: artifact.feature_version,
                algorithm: artifact.model.algorithm().name(),
                training_samples: artifact.metrics.training_samples,
                holdout_mae: artifact.metrics.holdout_mae,
                baseline_mae: artifact.metrics.baseline_mae,
                tuned: artifact.tuned,
                bytes: &bytes,
            },
            MODELS_KEPT,
        )
        .await?;
    Ok(TrainingOutcome::Stored {
        id,
        artifact: Box::new(artifact),
    })
}
