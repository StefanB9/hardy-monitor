//! Training: build samples, optionally tune, check against the slot-average
//! baseline on a recent holdout, then refit on all data.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::GymSchedule;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
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
/// Folds for the weekly grid search.
const TUNING_FOLDS: usize = 3;
/// Gap between a fold's training anchors and its validation anchors; longer
/// than any horizon so no training target overlaps validation.
const TUNING_GAP: TimeDelta = TimeDelta::hours(24);

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

/// 10th / 90th percentile of out-of-sample residuals (actual − predicted),
/// per horizon; index `h - 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonIntervals {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl HorizonIntervals {
    /// `(lower, upper)` residual offsets for a horizon; `(0, 0)` if unknown.
    pub fn offsets(&self, hours_ahead: u32) -> (f64, f64) {
        let i = (hours_ahead as usize).saturating_sub(1);
        (
            self.lower.get(i).copied().unwrap_or(0.0),
            self.upper.get(i).copied().unwrap_or(0.0),
        )
    }
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
    let baseline: Vec<f64> = holdout
        .iter()
        .map(|s| s.features.target_slot_mean())
        .collect();
    let holdout_mae = evaluation::mae(&predictions, &actual).unwrap_or(f64::INFINITY);
    let baseline_mae = evaluation::mae(&baseline, &actual).unwrap_or(f64::INFINITY);
    tracing::info!(holdout_mae, baseline_mae, tuned, "holdout evaluation");
    if !holdout_mae.is_finite() || holdout_mae >= baseline_mae {
        return Err(MlError::QualityGate {
            model_mae: holdout_mae,
            baseline_mae,
        });
    }
    let (intervals, mae_by_horizon) =
        horizon_statistics(&holdout, &predictions, options.max_hours_ahead);

    // Final model: all data, profile from all data.
    let profile = SlotProfile::from_history(history.view(), tz);
    let all_samples = build_samples(
        history,
        &profile,
        schedule,
        SampleWindow {
            anchors_from: first,
            anchors_until: after_last,
            targets_until: after_last,
        },
        options.max_hours_ahead,
    );
    let model = fit(options.algorithm, params, &all_samples)?;

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
            training_samples: all_samples.len(),
        },
    })
}

fn rows(samples: &[Sample]) -> Vec<FeatureRow> {
    samples.iter().map(|s| s.features).collect()
}

fn fit(algorithm: Algorithm, params: RfParams, samples: &[Sample]) -> Result<Model, MlError> {
    let targets: Vec<f64> = samples.iter().map(|s| s.target).collect();
    Model::fit(algorithm, params, &rows(samples), &targets)
}

/// Picks the grid entry with the lowest mean validation MAE over
/// time-ordered folds.
fn grid_search(samples: &[Sample]) -> RfParams {
    let folds = time_folds(samples);
    if folds.is_empty() {
        return RfParams::DEFAULT;
    }
    let scored: Vec<(RfParams, f64)> = RfParams::weekly_grid()
        .into_par_iter()
        .map(|params| {
            let maes: Vec<f64> = folds
                .iter()
                .filter_map(|(train, validation)| {
                    let model = fit(Algorithm::RandomForest, params, train).ok()?;
                    let predicted = model.predict_batch(&rows(validation)).ok()?;
                    let actual: Vec<f64> = validation.iter().map(|s| s.target).collect();
                    evaluation::mae(&predicted, &actual)
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let mean = if maes.len() == folds.len() {
                maes.iter().sum::<f64>() / maes.len() as f64
            } else {
                f64::INFINITY
            };
            (params, mean)
        })
        .collect();
    let best = scored
        .iter()
        .filter(|(_, mae)| mae.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(params, mae)| {
            tracing::info!(?params, cv_mae = mae, "grid search winner");
            *params
        });
    best.unwrap_or(RfParams::DEFAULT)
}

/// Expanding-window folds: validate on each of the last `TUNING_FOLDS`
/// chunks, train on anchors at least [`TUNING_GAP`] before it.
fn time_folds(samples: &[Sample]) -> Vec<(Vec<Sample>, Vec<Sample>)> {
    let chunk = samples.len() / (TUNING_FOLDS + 1);
    if chunk == 0 {
        return Vec::new();
    }
    (1..=TUNING_FOLDS)
        .filter_map(|k| {
            let validation = &samples[k * chunk..((k + 1) * chunk).min(samples.len())];
            let validation_start = validation.first()?.anchor;
            let train: Vec<Sample> = samples
                .iter()
                .take_while(|s| s.anchor < validation_start - TUNING_GAP)
                .copied()
                .collect();
            (!train.is_empty()).then(|| (train, validation.to_vec()))
        })
        .collect()
}

fn horizon_statistics(
    holdout: &[Sample],
    predictions: &[f64],
    max_hours_ahead: u32,
) -> (HorizonIntervals, Vec<f64>) {
    let horizons = max_hours_ahead as usize;
    let mut residuals: Vec<Vec<f64>> = vec![Vec::new(); horizons];
    for (sample, predicted) in holdout.iter().zip(predictions) {
        if let Some(bucket) = residuals.get_mut((sample.hours_ahead as usize).saturating_sub(1)) {
            bucket.push(sample.target - predicted);
        }
    }
    let mut lower = Vec::with_capacity(horizons);
    let mut upper = Vec::with_capacity(horizons);
    let mut mae_by_horizon = Vec::with_capacity(horizons);
    for mut bucket in residuals {
        bucket.sort_by(f64::total_cmp);
        lower.push(quantile(&bucket, 0.10));
        upper.push(quantile(&bucket, 0.90));
        #[allow(clippy::cast_precision_loss)]
        let mae = if bucket.is_empty() {
            f64::NAN
        } else {
            bucket.iter().map(|r| r.abs()).sum::<f64>() / bucket.len() as f64
        };
        mae_by_horizon.push(mae);
    }
    (HorizonIntervals { lower, upper }, mae_by_horizon)
}

/// Linear-interpolated quantile of sorted data; 0 for empty input.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n => {
            #[allow(clippy::cast_precision_loss)]
            let position = q.clamp(0.0, 1.0) * (n - 1) as f64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let below = position.floor() as usize;
            let above = (below + 1).min(n - 1);
            #[allow(clippy::cast_precision_loss)]
            let weight = position - below as f64;
            sorted[below] * (1.0 - weight) + sorted[above] * weight
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    pub(crate) fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 3, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    /// A learnable synthetic series: a daily curve whose level differs per
    /// day and persists through the day, sampled every 5 minutes during
    /// opening hours. The slot average cannot know a day's level; the
    /// current reading can.
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
                points.push((t, (curve + level).clamp(0.0, 100.0)));
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

    #[test]
    fn test_time_folds_keep_gap_between_train_and_validation() -> Result<()> {
        let history = learnable_history(21);
        let schedule = GymSchedule::default();
        let profile = SlotProfile::from_history(history.view(), schedule.timezone());
        let samples = build_samples(
            &history,
            &profile,
            &schedule,
            SampleWindow {
                anchors_from: local(0, 0, 0),
                anchors_until: local(21, 0, 0),
                targets_until: local(21, 0, 0),
            },
            6,
        );
        let folds = time_folds(&samples);
        assert_eq!(folds.len(), TUNING_FOLDS);
        for (train, validation) in &folds {
            let last_train = train.last().context("train")?.anchor;
            let first_validation = validation.first().context("validation")?.anchor;
            assert!(first_validation - last_train > TUNING_GAP);
        }
        Ok(())
    }

    #[test]
    fn test_quantile() {
        assert_relative_eq!(quantile(&[], 0.5), 0.0);
        assert_relative_eq!(quantile(&[3.0], 0.9), 3.0);
        assert_relative_eq!(quantile(&[0.0, 10.0], 0.1), 1.0);
        assert_relative_eq!(quantile(&[0.0, 5.0, 10.0], 0.5), 5.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn quantile_is_bounded_and_monotonic(
            mut values in prop::collection::vec(-50.0f64..50.0, 1..100),
            q1 in 0.0f64..1.0,
            q2 in 0.0f64..1.0,
        ) {
            values.sort_by(f64::total_cmp);
            let (lo, hi) = (q1.min(q2), q1.max(q2));
            let a = quantile(&values, lo);
            let b = quantile(&values, hi);
            prop_assert!(a <= b + 1e-12);
            prop_assert!(a >= values[0] - 1e-12);
            prop_assert!(b <= values[values.len() - 1] + 1e-12);
        }
    }
}
