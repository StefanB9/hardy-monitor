//! The nightly run: benchmark every model on the last weeks, refit each one
//! on the newest data and choose the default model per horizon.

use chrono::{DateTime, Utc};
use hardy_core::GymSchedule;
use rayon::prelude::*;

use crate::{
    benchmark::{self, Benchmark, Combination, Defaults, Selection, Settings},
    error::MlError,
    grid::{Grid, SLOT},
    history::History,
    live::{COMPARISON, FALLBACKS},
    progress::{Progress, Silent, Stage},
    suite::{self, AnyModel, ModelId, Observed, safety::guarded},
};

/// How a model's final fit went.
#[derive(Debug)]
pub enum FitResult {
    /// Fitted on the readings between the two times.
    Fitted {
        model: AnyModel,
        data_from: DateTime<Utc>,
        data_until: DateTime<Utc>,
    },
    /// Not enough history yet.
    Waiting { have_hours: i64, need_hours: i64 },
    /// Fitting failed.
    Failed(MlError),
}

/// Everything one nightly run produced.
#[derive(Debug)]
pub struct NightlyOutcome {
    pub benchmark: Benchmark<ModelId>,
    pub fits: Vec<(ModelId, FitResult)>,
    pub selection: Selection<ModelId>,
}

/// Runs the benchmark of `models` (forecasters and ensembles of them), fits
/// each forecaster on the newest data and selects the default per horizon
/// (keeping `previous` winners on ties). The ensembles' fits are in the
/// benchmark.
pub fn run(
    models: &[ModelId],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    previous: &Selection<ModelId>,
) -> NightlyOutcome {
    run_reporting(models, history, schedule, now, settings, previous, &Silent)
}

/// [`run`], reporting its stages to `progress`.
pub fn run_reporting(
    models: &[ModelId],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    previous: &Selection<ModelId>,
    progress: &dyn Progress,
) -> NightlyOutcome {
    let forecasters: Vec<ModelId> = models
        .iter()
        .copied()
        .filter(|id| id.combination().is_none())
        .collect();
    let ensembles: Vec<(ModelId, Combination)> = models
        .iter()
        .filter_map(|&id| id.combination().map(|c| (id, c)))
        .collect();
    let benchmark = benchmark::run_reporting(
        &forecasters,
        AnyModel::new,
        &ensembles,
        history,
        schedule,
        now,
        settings,
        &Defaults {
            fallbacks: FALLBACKS.to_vec(),
            comparison: Some(COMPARISON),
        },
        progress,
    );
    let selection = benchmark::select(&benchmark.reports, previous, settings);
    let fits = fit_reporting(models, history, schedule, now, settings, progress);
    NightlyOutcome {
        benchmark,
        fits,
        selection,
    }
}

/// Fits every forecaster in `models` on the readings in the training window
/// before `now` (ensembles are skipped).
pub fn fit_all(
    models: &[ModelId],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
) -> Vec<(ModelId, FitResult)> {
    fit_reporting(models, history, schedule, now, settings, &Silent)
}

fn fit_reporting(
    models: &[ModelId],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    progress: &dyn Progress,
) -> Vec<(ModelId, FitResult)> {
    let forecasters = models
        .iter()
        .filter(|id| id.combination().is_none())
        .count();
    progress.start(Stage::Fit, forecasters);
    let window = history.range(now - settings.training_window, now);
    let grid = history.first_time().map_or_else(Grid::default, |first| {
        Grid::build(history, schedule, first - SLOT, now)
    });
    let data = Observed::new(&grid, &window, schedule, now);
    let fits = models
        .par_iter()
        .filter_map(|&id| {
            let mut model = AnyModel::new(id)?;
            let result = match guarded(|| suite::fit(&mut model, &data)) {
                Ok(()) => match (window.first_time(), window.last_time()) {
                    (Some(data_from), Some(data_until)) => FitResult::Fitted {
                        model,
                        data_from,
                        data_until,
                    },
                    _ => FitResult::Failed(MlError::InsufficientData {
                        found: 0,
                        required: 1,
                    }),
                },
                Err(MlError::NotEnoughHistory {
                    have_hours,
                    need_hours,
                }) => FitResult::Waiting {
                    have_hours,
                    need_hours,
                },
                Err(e) => FitResult::Failed(e),
            };
            progress.advance(Stage::Fit);
            Some((id, result))
        })
        .collect();
    progress.finish(Stage::Fit);
    fits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        benchmark::Bucket,
        progress::{Stage, counting::Counting},
        test_support::{local, minutely},
    };

    #[test]
    fn test_nightly_run_benchmarks_fits_and_selects() {
        let history = minutely(10, |t| {
            use chrono::Timelike;
            20.0 + f64::from(
                crate::grid::slot_end_of(t)
                    .with_timezone(&GymSchedule::default().timezone())
                    .hour(),
            )
        });
        let models = [
            ModelId::Naive,
            ModelId::DailySeasonalNaive,
            ModelId::WeeklySeasonalNaive,
        ];
        let outcome = run(
            &models,
            &history,
            &GymSchedule::default(),
            local(10, 0, 0),
            &Settings::default(),
            &Vec::new(),
        );

        assert_eq!(outcome.benchmark.reports.len(), 3);
        // Each weekday repeats exactly (weekends have other hours): the
        // winner of every bucket forecast without error.
        assert_eq!(outcome.selection.len(), Bucket::HORIZONS.len());
        for (bucket, winner) in &outcome.selection {
            assert_ne!(*winner, ModelId::Naive);
            let report = outcome
                .benchmark
                .reports
                .iter()
                .find(|r| r.model == *winner);
            let mae = report.and_then(|r| r.score(*bucket)).map(|s| s.mae);
            assert!(
                mae.is_some_and(|m| m < 1e-9),
                "{winner:?} in {bucket:?}: {mae:?}"
            );
        }
        let status = |id| outcome.fits.iter().find(|(m, _)| *m == id).map(|(_, r)| r);
        assert!(matches!(
            status(ModelId::Naive),
            Some(FitResult::Fitted { .. })
        ));
        // Ten days of data, but the weekly model only needs seven.
        assert!(matches!(
            status(ModelId::WeeklySeasonalNaive),
            Some(FitResult::Fitted { .. })
        ));
    }

    #[test]
    fn test_nightly_run_fits_ensembles_but_not_as_models() {
        let history = minutely(4, |_| 30.0);
        let models = [
            ModelId::Naive,
            ModelId::MovingAverage,
            ModelId::StackedEnsemble,
        ];
        let outcome = run(
            &models,
            &history,
            &GymSchedule::default(),
            local(4, 0, 0),
            &Settings::default(),
            &Vec::new(),
        );
        assert_eq!(outcome.fits.len(), 2);
        assert_eq!(outcome.benchmark.reports.len(), 3);
        assert_eq!(outcome.benchmark.reports[2].model, ModelId::StackedEnsemble);
        assert!(matches!(
            outcome.benchmark.ensembles.as_slice(),
            [(ModelId::StackedEnsemble, Some(_))]
        ));
    }

    #[test]
    fn test_nightly_run_reports_every_stage() {
        let history = minutely(10, |_| 30.0);
        let models = [ModelId::Naive, ModelId::MovingAverage, ModelId::Mlp];
        let progress = Counting::default();
        let _ = run_reporting(
            &models,
            &history,
            &GymSchedule::default(),
            local(10, 0, 0),
            &Settings::default(),
            &Vec::new(),
            &progress,
        );
        // Ten backtest days; the network refits weekly: days 1 and 8 (day
        // 0 has no history before it).
        assert_eq!(progress.get(Stage::Backtest), Some((10, 10, true)));
        assert_eq!(progress.get(Stage::SharedFits), Some((2, 2, true)));
        assert_eq!(progress.get(Stage::Fit), Some((3, 3, true)));
    }

    #[test]
    fn test_models_without_enough_history_wait() {
        let history = minutely(2, |_| 30.0);
        let fits = fit_all(
            &[ModelId::Naive, ModelId::WeeklySeasonalMean],
            &history,
            &GymSchedule::default(),
            local(2, 0, 0),
            &Settings::default(),
        );
        assert!(matches!(fits[0].1, FitResult::Fitted { .. }));
        assert!(matches!(
            fits[1].1,
            FitResult::Waiting {
                need_hours: 168,
                ..
            }
        ));
    }
}
