//! Runs the forecasting suite for the forecaster: the nightly fit and benchmark
//! in the background, and live forecasts as new readings arrive.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule, MlConfig,
    analytics::midnight_local_as_utc,
    db::{BenchmarkModel, BenchmarkRun, BenchmarkScore, Database, FitStatus, ModelUpdate},
};
use tokio::task::JoinHandle;

use crate::{
    benchmark::{Bucket, Ensemble, Interval, Selection, Settings},
    error::MlError,
    history::History,
    live::{HorizonIntervals, LiveForecasts, forecast_all},
    nightly::{self, FitResult, NightlyOutcome},
    progress::{Progress, Silent},
    retrain::{RetrainReason, retrain_due},
    suite::{ARTIFACT_VERSION, AnyModel, ModelId},
};

/// Benchmark runs kept in the database.
pub const RUNS_KEPT: usize = 14;
/// Readings kept for live forecasts: the longest look-back of any model
/// (eight weeks for the recency-weighted mean) plus a day.
const LIVE_HISTORY: TimeDelta = TimeDelta::days(57);

/// A nightly run in progress.
struct Running {
    started_at: DateTime<Utc>,
    handle: JoinHandle<Result<NightlyOutcome>>,
}

/// The suite as the forecaster runs it.
pub struct ForecastService {
    config: MlConfig,
    schedule: GymSchedule,
    models: Vec<ModelId>,
    settings: Settings,
    fitted: HashMap<ModelId, AnyModel>,
    ensembles: HashMap<ModelId, Ensemble<ModelId>>,
    intervals: HashMap<ModelId, HorizonIntervals>,
    selection: Selection<ModelId>,
    /// When the last run finished.
    last_run_at: Option<DateTime<Utc>>,
    /// Models with a stored state (fitted, waiting or failed).
    known: HashSet<ModelId>,
    /// Why a run is due at once regardless of the schedule: the last one
    /// was interrupted, or some models have never been run.
    run_at_once: Option<RetrainReason>,
    running: Option<Running>,
    history: History,
    /// Told about the nightly run's stages.
    progress: Arc<dyn Progress>,
}

impl ForecastService {
    /// A service for every model of the suite.
    pub fn new(config: &MlConfig, schedule: GymSchedule) -> Self {
        Self::with_models(config, schedule, ModelId::ALL)
    }

    /// A service for `models` only.
    pub fn with_models(config: &MlConfig, schedule: GymSchedule, models: &[ModelId]) -> Self {
        let days = |d: i64| TimeDelta::days(d.max(1));
        Self {
            settings: Settings {
                days: u32::try_from(config.benchmark_days).unwrap_or(28),
                training_window: days(config.training_window_days),
                ..Settings::default()
            },
            config: config.clone(),
            schedule,
            models: models.to_vec(),
            fitted: HashMap::new(),
            ensembles: HashMap::new(),
            intervals: HashMap::new(),
            selection: Vec::new(),
            last_run_at: None,
            known: HashSet::new(),
            run_at_once: None,
            running: None,
            history: History::default(),
            progress: Arc::new(Silent),
        }
    }

    /// Reports the nightly run's stages to `progress`.
    #[must_use]
    pub fn with_progress(mut self, progress: Arc<dyn Progress>) -> Self {
        self.progress = progress;
        self
    }

    /// Loads stored fits, intervals and the selection from the last run.
    #[tracing::instrument(skip_all)]
    pub async fn load(&mut self, db: &Database) -> Result<()> {
        for record in db.model_records().await? {
            if let Some(id) = ModelId::from_key(&record.model) {
                self.known.insert(id);
            }
        }
        for (key, bytes) in db.model_artifacts(ARTIFACT_VERSION).await? {
            let Some(id) = ModelId::from_key(&key) else {
                continue;
            };
            let loaded = if id.combination().is_some() {
                Ensemble::from_bytes(&bytes).map(|e| {
                    self.ensembles.insert(id, e);
                })
            } else {
                AnyModel::from_bytes(id, &bytes).map(|model| {
                    self.fitted.insert(id, model);
                })
            };
            match loaded {
                Ok(()) => {}
                Err(e) => {
                    // Refitted by the next run.
                    tracing::warn!(model = key, error = %e, "stored model unreadable");
                    self.known.remove(&id);
                }
            }
        }
        if let Some(run) = db.latest_benchmark().await? {
            self.last_run_at = Some(run.finished_at);
            self.intervals = intervals_from_scores(&run.scores);
        }
        let interrupted = db
            .get_ml_state()
            .await?
            .last_attempt_at
            .is_some_and(|attempt| self.last_run_at.is_none_or(|run| attempt > run));
        let new_models = self.models.iter().any(|id| !self.known.contains(id));
        self.run_at_once = if interrupted {
            Some(RetrainReason::Interrupted)
        } else if new_models && self.last_run_at.is_some() {
            Some(RetrainReason::NewModels)
        } else {
            None
        };
        self.selection = db
            .forecast_selection()
            .await?
            .into_iter()
            .filter_map(|(code, key)| Some((Bucket::from_code(code)?, ModelId::from_key(&key)?)))
            .collect();
        tracing::info!(
            fitted = self.fitted.len(),
            selected = self.selection.len(),
            "forecast models loaded"
        );
        Ok(())
    }

    /// When the last complete run finished.
    pub fn last_run_at(&self) -> Option<DateTime<Utc>> {
        self.last_run_at
    }

    /// Models with a usable fit.
    pub fn fitted_count(&self) -> usize {
        self.fitted.len()
    }

    /// Whether a nightly run is in progress.
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Collects a finished run (storing its results and publishing fresh
    /// forecasts) and starts a new one if due. Never waits for a run.
    #[tracing::instrument(skip_all)]
    pub async fn tick(&mut self, db: &Database, now: DateTime<Utc>) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        if let Some(running) = self.running.take_if(|r| r.handle.is_finished()) {
            let result = running.handle.await.context("nightly run panicked")?;
            self.finish(db, running.started_at, result, now).await?;
            self.publish(db, now).await?;
        }
        if self.running.is_some() {
            return Ok(());
        }

        let complete = self.models.iter().all(|id| self.known.contains(id));
        let last_run = self.last_run_at.filter(|_| complete);
        let reason = if let Some(reason) = self.run_at_once {
            Some(reason)
        } else {
            let state = db.get_ml_state().await?;
            retrain_due(now, &self.schedule, last_run, &state)
        };
        let Some(reason) = reason else {
            return Ok(());
        };
        self.run_at_once = None;
        tracing::info!(?reason, "starting the nightly forecast run");
        db.start_training_attempt(now).await?;
        self.running = Some(Running {
            started_at: now,
            handle: self.spawn_run(db, now),
        });
        Ok(())
    }

    /// Runs the nightly work to completion and publishes forecasts.
    pub async fn run_now(&mut self, db: &Database, now: DateTime<Utc>) -> Result<()> {
        db.start_training_attempt(now).await?;
        let result = self
            .spawn_run(db, now)
            .await
            .context("nightly run panicked")?;
        self.finish(db, now, result, now).await?;
        self.publish(db, now).await?;
        Ok(())
    }

    fn spawn_run(&self, db: &Database, now: DateTime<Utc>) -> JoinHandle<Result<NightlyOutcome>> {
        let db = db.clone();
        let schedule = self.schedule.clone();
        let models = self.models.clone();
        let settings = self.settings;
        let previous = self.selection.clone();
        let progress = Arc::clone(&self.progress);
        let reach = settings.training_window + TimeDelta::days(i64::from(settings.days) + 1);
        tokio::spawn(async move {
            let logs = db
                .get_history_range(now - reach, now)
                .await
                .context("failed to load history for the nightly run")?;
            let history = History::from_logs(&logs);
            tokio::task::spawn_blocking(move || {
                nightly::run_reporting(
                    &models,
                    &history,
                    &schedule,
                    now,
                    &settings,
                    &previous,
                    progress.as_ref(),
                )
            })
            .await
            .context("nightly computation panicked")
        })
    }

    /// Stores a run's results and adopts its models.
    async fn finish(
        &mut self,
        db: &Database,
        started_at: DateTime<Utc>,
        result: Result<NightlyOutcome>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "nightly forecast run failed");
                db.finish_training_attempt(Some(&format!("{e:#}"))).await?;
                return Ok(());
            }
        };

        let states = stored_states(&outcome, &self.schedule);
        let updates: Vec<ModelUpdate<'_>> = states
            .iter()
            .map(|state| ModelUpdate {
                model: state.id.key(),
                status: state.status,
                message: state.message.as_deref(),
                fitted_at: (state.status == FitStatus::Fitted).then_some(now),
                data_from: state.span.map(|(from, _)| from),
                data_until: state.span.map(|(_, until)| until),
                artifact: state.bytes.as_deref().map(|b| (ARTIFACT_VERSION, b)),
            })
            .collect();
        db.save_model_states(&updates, now).await?;

        let run = benchmark_row(&outcome, &self.settings, started_at, now);
        let selection: Vec<(i16, String)> = outcome
            .selection
            .iter()
            .map(|(bucket, id)| (bucket.code(), id.key().to_string()))
            .collect();
        db.save_benchmark(&run, &selection, RUNS_KEPT).await?;
        db.finish_training_attempt(None).await?;

        self.intervals = intervals_from_scores(&run.scores);
        self.selection = outcome.selection;
        self.known = outcome
            .fits
            .iter()
            .map(|(id, _)| *id)
            .chain(outcome.benchmark.ensembles.iter().map(|(id, _)| *id))
            .collect();
        self.ensembles = outcome
            .benchmark
            .ensembles
            .into_iter()
            .filter_map(|(id, ensemble)| Some((id, ensemble?)))
            .collect();
        self.fitted = outcome
            .fits
            .into_iter()
            .filter_map(|(id, fit)| match fit {
                FitResult::Fitted { model, .. } => Some((id, model)),
                _ => None,
            })
            .collect();
        self.last_run_at = Some(now);
        tracing::info!(
            fitted = self.fitted.len(),
            ensembles = self.ensembles.len(),
            selection = ?self.selection,
            "nightly forecast run stored"
        );
        Ok(())
    }

    /// Publishes every fitted model's forecast as of `now` and the default
    /// forecast; returns them (e.g. for the accuracy log).
    #[tracing::instrument(skip_all, fields(%now))]
    pub async fn publish(&mut self, db: &Database, now: DateTime<Utc>) -> Result<LiveForecasts> {
        self.refresh_history(db, now).await?;
        let live = forecast_all(
            &self.fitted,
            &self.ensembles,
            &self.intervals,
            &self.selection,
            &self.history,
            &self.schedule,
            now,
        );
        for (id, error) in &live.failures {
            tracing::warn!(model = id.key(), %error, "live forecast failed");
        }
        db.replace_live_forecasts(&live.rows()).await?;
        Ok(live)
    }

    /// Fetches readings newer than those held and drops ones no model needs.
    async fn refresh_history(&mut self, db: &Database, now: DateTime<Utc>) -> Result<()> {
        let start = self
            .history
            .last_time()
            .map_or(now - LIVE_HISTORY, |t| t + TimeDelta::seconds(1));
        let logs = db
            .get_history_range(start, now)
            .await
            .context("failed to load recent readings")?;
        let newer = History::from_logs(&logs);
        if self.history.is_empty() {
            self.history = newer;
        } else {
            self.history.extend(newer.view().iter());
        }
        if self
            .history
            .first_time()
            .is_some_and(|first| first < now - LIVE_HISTORY - TimeDelta::days(1))
        {
            self.history = self.history.range(now - LIVE_HISTORY, now);
        }
        Ok(())
    }
}

/// Why a model waits, in whole days: "needs 7 days of history, has 3".
/// A model's state after a run, as stored.
struct StoredState {
    id: ModelId,
    status: FitStatus,
    message: Option<String>,
    /// The data the fit used (fitted models only).
    span: Option<(DateTime<Utc>, DateTime<Utc>)>,
    bytes: Option<Vec<u8>>,
}

/// The state of every forecaster and ensemble after `outcome`; a fit that
/// cannot be serialised counts as failed.
fn stored_states(outcome: &NightlyOutcome, schedule: &GymSchedule) -> Vec<StoredState> {
    let forecasters = outcome.fits.iter().map(|(id, fit)| match fit {
        FitResult::Fitted {
            model,
            data_from,
            data_until,
        } => fitted_state(*id, model.to_bytes(), (*data_from, *data_until)),
        FitResult::Waiting {
            have_hours,
            need_hours,
        } => StoredState {
            id: *id,
            status: FitStatus::Waiting,
            message: Some(waiting_message(*have_hours, *need_hours)),
            span: None,
            bytes: None,
        },
        FitResult::Failed(e) => StoredState {
            id: *id,
            status: FitStatus::Failed,
            message: Some(e.to_string()),
            span: None,
            bytes: None,
        },
    });
    let span = backtest_span(outcome, schedule);
    let ensembles =
        outcome
            .benchmark
            .ensembles
            .iter()
            .map(|(id, ensemble)| match (ensemble, span) {
                (Some(ensemble), Some(span)) => fitted_state(*id, ensemble.to_bytes(), span),
                _ => StoredState {
                    id: *id,
                    status: FitStatus::Waiting,
                    message: Some(ENSEMBLE_WAITING.to_string()),
                    span: None,
                    bytes: None,
                },
            });
    forecasters.chain(ensembles).collect()
}

/// A fitted model's state, or a failed one if it could not be serialised.
fn fitted_state(
    id: ModelId,
    bytes: Result<Vec<u8>, MlError>,
    span: (DateTime<Utc>, DateTime<Utc>),
) -> StoredState {
    match bytes {
        Ok(bytes) => StoredState {
            id,
            status: FitStatus::Fitted,
            message: None,
            span: Some(span),
            bytes: Some(bytes),
        },
        Err(e) => StoredState {
            id,
            status: FitStatus::Failed,
            message: Some(format!("could not be stored: {e}")),
            span: None,
            bytes: None,
        },
    }
}

/// Why an ensemble without members waits.
const ENSEMBLE_WAITING: &str = "waits for a backtest day of its members";

/// From the first backtest day's midnight to the midnight after the last:
/// the forecasts an ensemble's weights were fitted on.
fn backtest_span(
    outcome: &NightlyOutcome,
    schedule: &GymSchedule,
) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let tz = schedule.timezone();
    let first = outcome.benchmark.days.first()?;
    let last = outcome.benchmark.days.last()?.succ_opt()?;
    Some((
        midnight_local_as_utc(*first, tz),
        midnight_local_as_utc(last, tz),
    ))
}

fn waiting_message(have_hours: i64, need_hours: i64) -> String {
    let need = (need_hours + 23) / 24;
    let unit = if need == 1 { "day" } else { "days" };
    format!("needs {need} {unit} of history, has {}", have_hours / 24)
}

/// Interval offsets per model from stored scores.
fn intervals_from_scores(scores: &[BenchmarkScore]) -> HashMap<ModelId, HorizonIntervals> {
    let mut by_model: HashMap<ModelId, HashMap<Bucket, Interval>> = HashMap::new();
    for score in scores {
        if let (Some(id), Some(bucket), Some((low, high))) = (
            ModelId::from_key(&score.model),
            Bucket::from_code(score.bucket),
            score.interval,
        ) {
            by_model
                .entry(id)
                .or_default()
                .insert(bucket, Interval { low, high });
        }
    }
    by_model
        .into_iter()
        .map(|(id, buckets)| (id, HorizonIntervals::new(|b| buckets.get(&b).copied())))
        .collect()
}

/// The benchmark as stored.
fn benchmark_row(
    outcome: &NightlyOutcome,
    settings: &Settings,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
) -> BenchmarkRun {
    let benchmark = &outcome.benchmark;
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    BenchmarkRun {
        id: 0,
        started_at,
        finished_at,
        first_day: benchmark.days.first().copied(),
        last_day: benchmark.days.last().copied(),
        days: count(benchmark.days.len()),
        origins: count(benchmark.origins),
        models: benchmark
            .reports
            .iter()
            .map(|r| BenchmarkModel {
                model: r.model.key().to_string(),
                origins: count(r.origins),
                fitted_days: count(r.fitted_days),
                eligible: r.is_eligible(settings),
                failure: r.failure.as_ref().map(ToString::to_string),
            })
            .collect(),
        scores: benchmark
            .reports
            .iter()
            .flat_map(|r| {
                Bucket::ALL.into_iter().filter_map(move |bucket| {
                    let score = r.score(bucket)?;
                    Some(BenchmarkScore {
                        model: r.model.key().to_string(),
                        bucket: bucket.code(),
                        mae: score.mae,
                        rmse: score.rmse,
                        mase: score.mase,
                        coverage: score.coverage,
                        pinball: score.pinball,
                        errors: count(score.errors),
                        interval: r.interval(bucket).map(|i| (i.low, i.high)),
                    })
                })
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waiting_message_rounds_to_days() {
        assert_eq!(waiting_message(70, 168), "needs 7 days of history, has 2");
        assert_eq!(waiting_message(0, 24), "needs 1 day of history, has 0");
    }

    #[test]
    fn test_intervals_from_scores_skip_unknown_models_and_overall() {
        let score = |model: &str, bucket: i16| BenchmarkScore {
            model: model.to_string(),
            bucket,
            mae: 1.0,
            rmse: 1.0,
            mase: None,
            coverage: None,
            pinball: None,
            errors: 1,
            interval: Some((-1.0, f64::from(bucket))),
        };
        let intervals =
            intervals_from_scores(&[score("naive", 1), score("naive", 3), score("gone", 1)]);
        assert_eq!(intervals.len(), 1);
        let naive = intervals.get(&ModelId::Naive).copied().unwrap_or_default();
        assert_eq!(naive.for_step(1).map(|i| i.high), Some(1.0));
        assert_eq!(naive.for_step(5), None);
        assert_eq!(naive.for_step(20).map(|i| i.high), Some(3.0));
    }
}
