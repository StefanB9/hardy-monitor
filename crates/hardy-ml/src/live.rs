//! Live forecasts: every fitted model's forecast from the newest data with
//! its conformal interval, and the default forecast stitched together from
//! the model selected for each horizon bucket.

use std::{collections::HashMap, hash::BuildHasher};

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule,
    db::{ForecastLogEntry, LiveForecast},
};

use crate::{
    benchmark::{Bucket, Interval, Selection},
    error::MlError,
    grid::{Grid, HORIZON_STEPS, SLOT, floor_to_slot, is_open_slot},
    history::History,
    suite::{self, AnyModel, ModelId, Observed, safety::guarded},
};

/// Stored model key of the default forecast.
pub const SELECTED: &str = "selected";
/// Used in this order when a selected model has no forecast (or nothing
/// has been selected yet).
const FALLBACKS: [ModelId; 4] = [
    ModelId::ProfileDecayingOffset,
    ModelId::WeeklyRecencyMean,
    ModelId::DailySeasonalNaive,
    ModelId::Naive,
];
/// The model logged alongside the default forecast for comparison.
const COMPARISON: ModelId = ModelId::ProfileDecayingOffset;

/// A model's interval offsets per horizon bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HorizonIntervals([Option<Interval>; 3]);

impl HorizonIntervals {
    /// Intervals from a per-bucket lookup.
    pub fn new(mut interval: impl FnMut(Bucket) -> Option<Interval>) -> Self {
        Self(Bucket::HORIZONS.map(&mut interval))
    }

    /// The interval for forecast step `step`.
    pub fn for_step(&self, step: usize) -> Option<Interval> {
        let bucket = Bucket::of_step(step);
        Bucket::HORIZONS
            .iter()
            .position(|b| *b == bucket)
            .and_then(|i| self.0[i])
    }
}

/// One value of a live forecast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub step: usize,
    pub target: DateTime<Utc>,
    pub value: f64,
    pub bounds: Option<(f64, f64)>,
}

/// All live forecasts made at one origin.
#[derive(Debug, Clone)]
pub struct LiveForecasts {
    pub origin: DateTime<Utc>,
    /// Open slots only, per model.
    pub models: Vec<(ModelId, Vec<Point>)>,
    /// The default forecast and the model each value came from.
    pub selected: Vec<(Point, ModelId)>,
    /// Models whose forecast failed.
    pub failures: Vec<(ModelId, MlError)>,
}

/// Forecasts of every model in `fitted` from the newest slot boundary at or
/// before `now`.
pub fn forecast_all<S1: BuildHasher, S2: BuildHasher>(
    fitted: &HashMap<ModelId, AnyModel, S1>,
    intervals: &HashMap<ModelId, HorizonIntervals, S2>,
    selection: &Selection<ModelId>,
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
) -> LiveForecasts {
    let origin = floor_to_slot(now);
    let seen = history.until(origin);
    let grid = seen.first_time().map_or_else(Grid::default, |first| {
        Grid::build(&seen, schedule, first - SLOT, origin)
    });
    let data = Observed::new(&grid, &seen, schedule, origin);

    let mut models = Vec::with_capacity(fitted.len());
    let mut failures = Vec::new();
    for id in ModelId::ALL {
        let Some(model) = fitted.get(id) else {
            continue;
        };
        match guarded(|| suite::forecast(model, &data)) {
            Ok(values) => {
                let model_intervals = intervals.get(id).copied().unwrap_or_default();
                let points = (1..=HORIZON_STEPS)
                    .zip(values)
                    .map(|(step, value)| (step, data.grid().step_end(step), value))
                    .filter(|(_, target, _)| is_open_slot(schedule, *target))
                    .map(|(step, target, value)| Point {
                        step,
                        target,
                        value,
                        bounds: model_intervals.for_step(step).map(|i| i.around(value)),
                    })
                    .collect();
                models.push((*id, points));
            }
            Err(e) => failures.push((*id, e)),
        }
    }
    let selected = stitch(&models, selection);
    LiveForecasts {
        origin,
        models,
        selected,
        failures,
    }
}

/// For each open step, the value of the model selected for its bucket (or
/// the first fallback that has one).
fn stitch(
    models: &[(ModelId, Vec<Point>)],
    selection: &Selection<ModelId>,
) -> Vec<(Point, ModelId)> {
    let point = |id: ModelId, step: usize| {
        models
            .iter()
            .find(|(m, _)| *m == id)
            .and_then(|(_, points)| points.iter().find(|p| p.step == step))
            .copied()
    };
    // Every model forecasts the same open slots.
    let steps: Vec<usize> = models.first().map_or_else(Vec::new, |(_, points)| {
        points.iter().map(|p| p.step).collect()
    });
    steps
        .into_iter()
        .filter_map(|step| {
            let bucket = Bucket::of_step(step);
            let preferred = selection
                .iter()
                .find(|(b, _)| *b == bucket)
                .map(|(_, m)| *m);
            preferred
                .into_iter()
                .chain(selection.iter().map(|(_, m)| *m))
                .chain(FALLBACKS)
                .find_map(|id| point(id, step).map(|p| (p, id)))
        })
        .collect()
}

impl LiveForecasts {
    /// Database rows: every model's forecast and the default one.
    pub fn rows(&self) -> Vec<LiveForecast> {
        let row = |model: &str, point: &Point, source: Option<ModelId>| LiveForecast {
            model: model.to_string(),
            target: point.target,
            made_at: self.origin,
            value: point.value,
            low: point.bounds.map(|(low, _)| low),
            high: point.bounds.map(|(_, high)| high),
            source: source.map(|id| id.key().to_string()),
        };
        let total = self.models.iter().map(|(_, p)| p.len()).sum::<usize>() + self.selected.len();
        let mut rows = Vec::with_capacity(total);
        for (id, points) in &self.models {
            rows.extend(points.iter().map(|p| row(id.key(), p, None)));
        }
        rows.extend(
            self.selected
                .iter()
                .map(|(p, id)| row(SELECTED, p, Some(*id))),
        );
        rows
    }

    /// Forecast log entries for each whole hour `1..=hours` ahead whose slot
    /// is open: the default forecast, with the profile baseline for
    /// comparison.
    pub fn log_entries(&self, hours: u32) -> Vec<ForecastLogEntry> {
        let comparison: HashMap<usize, f64> = self
            .models
            .iter()
            .find(|(id, _)| *id == COMPARISON)
            .map_or_else(HashMap::new, |(_, points)| {
                points.iter().map(|p| (p.step, p.value)).collect()
            });
        (1..=hours)
            .filter_map(|h| {
                let step = usize::try_from(h).ok()? * 4;
                let (point, source) = self.selected.iter().find(|(p, _)| p.step == step)?;
                let (low, high) = point.bounds.unwrap_or((point.value, point.value));
                Some(ForecastLogEntry {
                    made_at: self.origin,
                    target: self.origin + TimeDelta::hours(i64::from(h)),
                    horizon_hours: h,
                    model: Some(source.key().to_string()),
                    predicted: point.value,
                    low,
                    high,
                    baseline: comparison.get(&step).copied().unwrap_or(point.value),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;

    use super::*;
    use crate::{
        benchmark::Settings,
        nightly::{FitResult, fit_all},
        test_support::{local, minutely},
    };

    fn fitted(
        history: &History,
        now: DateTime<Utc>,
        ids: &[ModelId],
    ) -> HashMap<ModelId, AnyModel> {
        fit_all(
            ids,
            history,
            &GymSchedule::default(),
            now,
            &Settings::default(),
        )
        .into_iter()
        .filter_map(|(id, result)| match result {
            FitResult::Fitted { model, .. } => Some((id, model)),
            _ => None,
        })
        .collect()
    }

    #[test]
    fn test_live_forecasts_use_the_selection_per_bucket() -> Result<()> {
        let history = minutely(3, |t| if t < local(2, 9, 0) { 30.0 } else { 50.0 });
        let now = local(2, 10, 7);
        let models = fitted(
            &history,
            local(2, 0, 0),
            &[ModelId::Naive, ModelId::DailySeasonalNaive],
        );
        let selection = vec![
            (Bucket::NextHour, ModelId::Naive),
            (Bucket::OneToThreeHours, ModelId::DailySeasonalNaive),
        ];
        let interval = Interval {
            low: -2.0,
            high: 3.0,
        };
        let intervals =
            HashMap::from([(ModelId::Naive, HorizonIntervals::new(|_| Some(interval)))]);
        let live = forecast_all(
            &models,
            &intervals,
            &selection,
            &history,
            &GymSchedule::default(),
            now,
        );

        assert_eq!(live.origin, local(2, 10, 0));
        assert_eq!(live.failures, Vec::new());
        let (first, source) = live.selected.first().context("a first step")?;
        assert_eq!((first.step, *source), (1, ModelId::Naive));
        assert_relative_eq!(first.value, 50.0);
        assert_eq!(first.bounds, Some((48.0, 53.0)));
        let (fifth, source) = live
            .selected
            .iter()
            .find(|(p, _)| p.step == 5)
            .context("step 5")?;
        assert_eq!(*source, ModelId::DailySeasonalNaive);
        assert_relative_eq!(fifth.value, 30.0);
        assert_eq!(fifth.bounds, None, "no interval known for that model");
        // 3–24 h was not selected: the first available fallback is used.
        let (late, source) = live
            .selected
            .iter()
            .find(|(p, _)| p.step == 13)
            .context("step 13")?;
        assert_eq!(*source, ModelId::Naive);
        assert_relative_eq!(late.value, 50.0);
        Ok(())
    }

    #[test]
    fn test_rows_and_log_entries() {
        let history = minutely(3, |_| 40.0);
        let now = local(2, 10, 0);
        let models = fitted(
            &history,
            local(2, 0, 0),
            &[ModelId::Naive, ModelId::ProfileDecayingOffset],
        );
        let live = forecast_all(
            &models,
            &HashMap::new(),
            &vec![(Bucket::NextHour, ModelId::Naive)],
            &history,
            &GymSchedule::default(),
            now,
        );
        let rows = live.rows();
        let selected: Vec<_> = rows.iter().filter(|r| r.model == SELECTED).collect();
        assert_eq!(selected.len(), live.selected.len());
        assert!(
            selected
                .iter()
                .all(|r| r.source.is_some() && r.made_at == now)
        );
        // Open until 23:00: 13 hours of open slots ahead, plus tomorrow from 6.
        assert!(rows.iter().filter(|r| r.model == "naive").count() < HORIZON_STEPS);

        let log = live.log_entries(6);
        assert_eq!(log.len(), 6);
        assert_eq!(log[0].target, local(2, 11, 0));
        assert_eq!(log[0].model.as_deref(), Some("naive"));
        assert_relative_eq!(log[0].predicted, 40.0);
        assert_relative_eq!(log[0].low, 40.0);
    }

    #[test]
    fn test_horizon_intervals_by_step() {
        let i = |low| Some(Interval { low, high: 1.0 });
        let intervals = HorizonIntervals::new(|b| match b {
            Bucket::NextHour => i(-1.0),
            Bucket::OneToThreeHours => i(-2.0),
            _ => None,
        });
        assert_eq!(intervals.for_step(4).map(|x| x.low), Some(-1.0));
        assert_eq!(intervals.for_step(5).map(|x| x.low), Some(-2.0));
        assert_eq!(intervals.for_step(13), None);
    }
}
