//! Hyperparameter search on time-ordered folds.

use chrono::TimeDelta;
use rayon::prelude::*;

/// Folds for the weekly grid search.
use super::{fit, rows};
use crate::{
    evaluation,
    model::{Algorithm, RfParams},
    samples::Sample,
};

pub(super) const TUNING_FOLDS: usize = 3;

/// Gap between a fold's training anchors and its validation anchors; longer
/// than any horizon so no training target overlaps validation.
pub(super) const TUNING_GAP: TimeDelta = TimeDelta::hours(24);

/// Picks the grid entry with the lowest mean validation MAE over
/// time-ordered folds.
pub(super) fn grid_search(samples: &[Sample]) -> RfParams {
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
pub(super) fn time_folds(samples: &[Sample]) -> Vec<(Vec<Sample>, Vec<Sample>)> {
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

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        profile::SlotProfile,
        samples::{SampleWindow, build_samples},
        training::tests::{learnable_history, local},
    };

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
}
