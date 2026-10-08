//! Training: build samples, optionally tune, check against the slot-average
//! baseline on a recent holdout, then refit on all data.

mod intervals;
mod tuning;

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::GymSchedule;
pub use intervals::HorizonIntervals;
use intervals::horizon_statistics;
use serde::{Deserialize, Serialize};
use tuning::grid_search;

use crate::{
    baseline::Baseline,
    error::MlError,
    evaluation,
    features::{FEATURE_VERSION, FeatureRow},
    history::History,
    model::{Algorithm, Model, RfParams},
    profile::SlotProfile,
    samples::{Sample, SampleWindow, build_samples},
};

/// Fewer holdout samples than this cannot judge a model.
pub const MIN_HOLDOUT_SAMPLES: usize = 50;

/// How hyperparameters are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tuning {
    /// Use these parameters (nightly).
    Fixed(RfParams),
    /// Search [`RfParams::weekly_grid`] (weekly).
    GridSearch,
}

/// What to train.
#[derive(Debug, Clone, Copy)]
pub struct TrainOptions {
    pub algorithm: Algorithm,
    pub tuning: Tuning,
    pub max_hours_ahead: u32,
    /// Minimum training samples (before the holdout).
    pub min_samples: usize,
    pub holdout_days: i64,
}

/// How the model did on the holdout, versus the slot-average baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub holdout_mae: f64,
    pub baseline_mae: f64,
    /// Index `h - 1`.
    pub mae_by_horizon: Vec<f64>,
    pub holdout_samples: usize,
    /// Samples the final model was fitted on.
    pub training_samples: usize,
}

/// A trained model with everything needed to forecast.
#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub feature_version: u32,
    pub trained_at: DateTime<Utc>,
    /// Newest reading used for training.
    pub data_until: DateTime<Utc>,
    pub max_hours_ahead: u32,
    /// Whether hyperparameters came from a grid search.
    pub tuned: bool,
    pub model: Model,
    pub profile: SlotProfile,
    pub intervals: HorizonIntervals,
    pub metrics: TrainingMetrics,
}

/// Trains a model on `history`, or explains why not.
#[tracing::instrument(skip_all, fields(readings = history.len(), tuning = ?options.tuning))]
pub fn train(
    history: &History,
    schedule: &GymSchedule,
    options: &TrainOptions,
    now: DateTime<Utc>,
) -> Result<ModelArtifact, MlError> {
    train_inner(history, schedule, options, now, Gate::Enforce)
}

/// Like [`train`] but keeps a model even if the baseline is better, for
/// tests that need an artifact of a given algorithm whatever its quality.
#[cfg(test)]
pub(crate) fn train_ungated(
    history: &History,
    schedule: &GymSchedule,
    options: &TrainOptions,
    now: DateTime<Utc>,
) -> Result<ModelArtifact, MlError> {
    train_inner(history, schedule, options, now, Gate::Skip)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gate {
    Enforce,
    #[cfg_attr(not(test), allow(dead_code))]
    Skip,
}

fn train_inner(
    history: &History,
    schedule: &GymSchedule,
    options: &TrainOptions,
    now: DateTime<Utc>,
    gate: Gate,
) -> Result<ModelArtifact, MlError> {
    let (Some(first), Some(last)) = (history.first_time(), history.last_time()) else {
        return Err(MlError::InsufficientData {
            found: 0,
            required: options.min_samples,
        });
    };
    let tz = schedule.timezone();
    let cutoff = last - TimeDelta::days(options.holdout_days);
    let after_last = last + TimeDelta::seconds(1);

    // Holdout check: everything (incl. the slot profile) learned before the
    // cutoff, scored on the days after it.
    let earlier = history.before(cutoff);
    let earlier_profile = SlotProfile::from_history(earlier.view(), tz);
    let train_samples = build_samples(
        &earlier,
        &earlier_profile,
        schedule,
        SampleWindow {
            anchors_from: first,
            anchors_until: cutoff,
            targets_until: cutoff,
        },
        options.max_hours_ahead,
    );
    if train_samples.len() < options.min_samples {
        return Err(MlError::InsufficientData {
            found: train_samples.len(),
            required: options.min_samples,
        });
    }
    let holdout = build_samples(
        history,
        &earlier_profile,
        schedule,
        SampleWindow {
            anchors_from: cutoff,
            anchors_until: after_last,
            targets_until: after_last,
        },
        options.max_hours_ahead,
    );
    if holdout.len() < MIN_HOLDOUT_SAMPLES {
        return Err(MlError::InsufficientData {
            found: holdout.len(),
            required: MIN_HOLDOUT_SAMPLES,
        });
    }

    let (params, tuned) = match options.tuning {
        Tuning::Fixed(params) => (params, false),
        Tuning::GridSearch if options.algorithm == Algorithm::RandomForest => {
            (grid_search(&train_samples), true)
        }
        Tuning::GridSearch => (RfParams::DEFAULT, false),
    };

    let candidate = fit(options.algorithm, params, &train_samples)?;
    let predictions = candidate.predict_batch(&rows(&holdout))?;
    let actual: Vec<f64> = holdout.iter().map(|s| s.target).collect();
    // Judged against the non-ML forecast, fitted on the same earlier data;
    // `predict` reads only readings up to each anchor.
    let earlier_baseline = Baseline::fit(&earlier, schedule, options.max_hours_ahead);
    let baseline: Vec<f64> = holdout
        .iter()
        .map(|s| earlier_baseline.predict(history, s.anchor, s.hours_ahead))
        .collect();
    let holdout_mae = evaluation::mae(&predictions, &actual).unwrap_or(f64::INFINITY);
    let baseline_mae = evaluation::mae(&baseline, &actual).unwrap_or(f64::INFINITY);
    tracing::info!(holdout_mae, baseline_mae, tuned, "holdout evaluation");
    if gate == Gate::Enforce && (!holdout_mae.is_finite() || holdout_mae >= baseline_mae) {
        return Err(MlError::QualityGate {
            model_mae: holdout_mae,
            baseline_mae,
        });
    }
    let (intervals, mae_by_horizon) =
        horizon_statistics(&holdout, &predictions, options.max_hours_ahead);

    let window = SampleWindow {
        anchors_from: first,
        anchors_until: after_last,
        targets_until: after_last,
    };
    let (profile, model, training_samples) = fit_final(history, schedule, options, params, window)?;

    Ok(ModelArtifact {
        feature_version: FEATURE_VERSION,
        trained_at: now,
        data_until: last,
        max_hours_ahead: options.max_hours_ahead,
        tuned,
        model,
        profile,
        intervals,
        metrics: TrainingMetrics {
            holdout_mae,
            baseline_mae,
            mae_by_horizon,
            holdout_samples: holdout.len(),
            training_samples,
        },
    })
}

/// The final model: all data in `window`, profile from all data.
fn fit_final(
    history: &History,
    schedule: &GymSchedule,
    options: &TrainOptions,
    params: RfParams,
    window: SampleWindow,
) -> Result<(SlotProfile, Model, usize), MlError> {
    let profile = SlotProfile::from_history(history.view(), schedule.timezone());
    let all_samples = build_samples(history, &profile, schedule, window, options.max_hours_ahead);
    let model = fit(options.algorithm, params, &all_samples)?;
    Ok((profile, model, all_samples.len()))
}

pub(super) fn rows(samples: &[Sample]) -> Vec<FeatureRow> {
    samples.iter().map(|s| s.features).collect()
}

pub(super) fn fit(
    algorithm: Algorithm,
    params: RfParams,
    samples: &[Sample],
) -> Result<Model, MlError> {
    let targets: Vec<f64> = samples.iter().map(|s| s.target).collect();
    Model::fit(algorithm, params, &rows(samples), &targets)
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;
    use chrono::TimeZone;

    use super::*;

    pub(crate) fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 3, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    /// A learnable synthetic series: a daily curve plus a per-day deviation
    /// that grows through the day, sampled every 5 minutes during opening
    /// hours.
    pub(crate) fn learnable_history(days: i64) -> History {
        let schedule = GymSchedule::default();
        let mut points = Vec::new();
        for d in 0..days {
            #[allow(clippy::cast_precision_loss)]
            let level = ((d * 37) % 23) as f64 - 11.0;
            for step in 0..(24 * 12) {
                let t = local(d, 0, 0) + TimeDelta::minutes(step * 5);
                if !schedule.is_open(&t) {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let hour = (step as f64) / 12.0;
                let curve = 35.0 + 20.0 * ((hour - 6.0) / 17.0 * std::f64::consts::PI).sin();
                // The day's deviation grows through the day: the model learns
                // that from the time of day and the recent trend; the
                // baseline's carry-over (at most 1) cannot follow it.
                let drift = level * (hour - 5.0) / 6.0;
                points.push((t, (curve + drift).clamp(0.0, 100.0)));
            }
        }
        History::new(points)
    }

    pub(crate) fn options(tuning: Tuning) -> TrainOptions {
        TrainOptions {
            algorithm: Algorithm::RandomForest,
            tuning,
            max_hours_ahead: 6,
            min_samples: 200,
            holdout_days: 7,
        }
    }

    #[test]
    fn test_train_beats_baseline_on_learnable_data() -> Result<()> {
        let history = learnable_history(28);
        let fast = RfParams::new(30, 8, 10)?;
        let artifact = train(
            &history,
            &GymSchedule::default(),
            &options(Tuning::Fixed(fast)),
            local(28, 0, 0),
        )?;
        let m = &artifact.metrics;
        assert!(
            m.holdout_mae < m.baseline_mae,
            "model {} vs baseline {}",
            m.holdout_mae,
            m.baseline_mae
        );
        assert_eq!(m.mae_by_horizon.len(), 6);
        assert_eq!(artifact.feature_version, FEATURE_VERSION);
        assert!(!artifact.tuned);
        let (low, high) = artifact.intervals.offsets(1);
        assert!(low <= high);
        Ok(())
    }

    #[test]
    fn test_train_rejects_model_worse_than_baseline() -> Result<()> {
        // Every day identical and constant within each hour: the slot average
        // is exact, the model can at best tie, so the gate must reject it.
        let schedule = GymSchedule::default();
        let mut points = Vec::new();
        for d in 0..21 {
            for step in 0..(24 * 12) {
                let t = local(d, 0, 0) + TimeDelta::minutes(step * 5);
                if schedule.is_open(&t) {
                    #[allow(clippy::cast_precision_loss)]
                    points.push((t, 20.0 + (step / 12) as f64));
                }
            }
        }
        let result = train(
            &History::new(points),
            &schedule,
            &options(Tuning::Fixed(RfParams::new(20, 8, 10)?)),
            local(21, 0, 0),
        );
        assert!(
            matches!(result, Err(MlError::QualityGate { .. })),
            "{result:?}"
        );
        Ok(())
    }

    #[test]
    fn test_train_gate_judges_against_the_quarter_hour_baseline() -> Result<()> {
        // Every day identical, quiet in the first half of each hour and busy
        // in the second: hourly averages miss by 20 points, the baseline's
        // quarter-hour slots are (almost) exact.
        let schedule = GymSchedule::default();
        let mut points = Vec::new();
        for d in 0..21 {
            for step in 0..(24 * 12) {
                let t = local(d, 0, 0) + TimeDelta::minutes(step * 5);
                if schedule.is_open(&t) {
                    points.push((t, if step % 12 < 6 { 20.0 } else { 60.0 }));
                }
            }
        }
        let result = train(
            &History::new(points),
            &schedule,
            &options(Tuning::Fixed(RfParams::new(20, 8, 10)?)),
            local(21, 0, 0),
        );
        let baseline_mae = match result {
            Ok(artifact) => artifact.metrics.baseline_mae,
            Err(MlError::QualityGate { baseline_mae, .. }) => baseline_mae,
            Err(e) => return Err(e.into()),
        };
        assert!(baseline_mae < 1.0, "baseline error {baseline_mae}");
        Ok(())
    }

    #[test]
    fn test_train_requires_enough_data() {
        let history = learnable_history(2);
        let result = train(
            &history,
            &GymSchedule::default(),
            &options(Tuning::Fixed(RfParams::DEFAULT)),
            local(2, 0, 0),
        );
        assert!(matches!(result, Err(MlError::InsufficientData { .. })));
        assert!(matches!(
            train(
                &History::default(),
                &GymSchedule::default(),
                &options(Tuning::GridSearch),
                local(0, 0, 0)
            ),
            Err(MlError::InsufficientData { found: 0, .. })
        ));
    }
}
