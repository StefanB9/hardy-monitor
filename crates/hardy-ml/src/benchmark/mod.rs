//! Rolling-origin benchmark: how well each model would have forecast the
//! last weeks, seeing only what was known at the time.
//!
//! For each backtest day, every model is fitted on the data before that
//! day's midnight, then forecasts from every full hour of the day around
//! opening hours. Each forecast is scored against the slots that followed,
//! per horizon bucket. Intervals are conformal: a model's interval on a day
//! comes from its own errors on the days before, so coverage is measured
//! honestly; the interval for live forecasts comes from all its errors.
//!
//! Ensembles are scored the same way: on each day they combine the
//! members' forecasts with weights fitted on the days before only.
//!
//! The default forecast is recorded as it would have been: on each day, the
//! models selected on the scores of the days before.

pub mod ensemble;
mod metrics;
mod past;
mod selection;

use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use ensemble::Rows;
pub use ensemble::{Combination, Ensemble};
use hardy_core::{GymSchedule, analytics::midnight_local_as_utc};
use metrics::Accumulator;
pub use metrics::{INTERVAL_TAIL, Interval, MIN_INTERVAL_ERRORS, Score};
pub use past::{Defaults, PastForecast};
use rayon::prelude::*;
pub use selection::{Selection, select};

use crate::{
    error::MlError,
    grid::{Grid, HORIZON_STEPS, SLOT, is_open_slot, same_local_time_before},
    history::History,
    progress::{Progress, Silent, Stage},
    suite::{self, Forecaster, Observed, safety::guarded},
};

/// Steps after an origin of which one must be open for the origin to count:
/// covers the hour before opening.
const ORIGIN_LOOKAHEAD_STEPS: i32 = 5;
/// History needed before the weekly lag is used for the MASE scale.
const WEEKLY_SCALE_SPAN: TimeDelta = TimeDelta::days(14);

/// Horizons scored together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bucket {
    /// Every horizon.
    Overall,
    /// Up to one hour ahead.
    NextHour,
    /// One to three hours ahead.
    OneToThreeHours,
    /// Three to 24 hours ahead.
    ThreeToDay,
}

impl Bucket {
    /// Every bucket.
    pub const ALL: [Bucket; 4] = [
        Bucket::Overall,
        Bucket::NextHour,
        Bucket::OneToThreeHours,
        Bucket::ThreeToDay,
    ];
    /// The horizon buckets (without [`Bucket::Overall`]), nearest first.
    pub const HORIZONS: [Bucket; 3] = [
        Bucket::NextHour,
        Bucket::OneToThreeHours,
        Bucket::ThreeToDay,
    ];

    /// The bucket of forecast step `step` (1 = the first slot).
    pub fn of_step(step: usize) -> Self {
        match step {
            0..=4 => Bucket::NextHour,
            5..=12 => Bucket::OneToThreeHours,
            _ => Bucket::ThreeToDay,
        }
    }

    /// Stable code for storage.
    pub fn code(self) -> i16 {
        match self {
            Bucket::Overall => 0,
            Bucket::NextHour => 1,
            Bucket::OneToThreeHours => 2,
            Bucket::ThreeToDay => 3,
        }
    }

    /// The bucket stored as `code`.
    pub fn from_code(code: i16) -> Option<Self> {
        Bucket::ALL.into_iter().find(|b| b.code() == code)
    }

    /// Display label.
    pub fn label(self) -> &'static str {
        match self {
            Bucket::Overall => "All",
            Bucket::NextHour => "Next hour",
            Bucket::OneToThreeHours => "1–3 h",
            Bucket::ThreeToDay => "3–24 h",
        }
    }

    fn index(self) -> usize {
        match self {
            Bucket::Overall => 0,
            Bucket::NextHour => 1,
            Bucket::OneToThreeHours => 2,
            Bucket::ThreeToDay => 3,
        }
    }
}

/// How the benchmark runs.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Most recent complete days with data to backtest.
    pub days: u32,
    /// Data each fit may use, back from its cut.
    pub training_window: TimeDelta,
    /// Origins a model must have forecast from to be selectable.
    pub min_origins: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            days: 28,
            training_window: TimeDelta::days(56),
            min_origins: 24,
        }
    }
}

/// One model's benchmark result.
#[derive(Debug, Clone)]
pub struct ModelReport<L> {
    pub model: L,
    /// Origins the model forecast from.
    pub origins: usize,
    /// Backtest days on which it could be fitted.
    pub fitted_days: usize,
    /// The last failure (fit or forecast), if any; missing history is not a
    /// failure.
    pub failure: Option<MlError>,
    scores: [Option<Score>; 4],
    intervals: [Option<Interval>; 4],
}

impl<L> ModelReport<L> {
    /// Score in `bucket`, if anything was scored there.
    pub fn score(&self, bucket: Bucket) -> Option<&Score> {
        self.scores[bucket.index()].as_ref()
    }

    /// Interval offsets for forecasts in `bucket`, from all its errors.
    pub fn interval(&self, bucket: Bucket) -> Option<Interval> {
        self.intervals[bucket.index()]
    }

    /// Whether the model forecast from enough origins to be selected.
    pub fn is_eligible(&self, settings: &Settings) -> bool {
        self.origins >= settings.min_origins
    }
}

/// The benchmark of all models.
#[derive(Debug, Clone)]
pub struct Benchmark<L> {
    /// The backtested days, oldest first.
    pub days: Vec<NaiveDate>,
    /// Origins forecast from (by any model).
    pub origins: usize,
    /// The models' reports, then the ensembles'.
    pub reports: Vec<ModelReport<L>>,
    /// Each ensemble fitted on every backtest day (`None` without members).
    pub ensembles: Vec<(L, Option<Ensemble<L>>)>,
    /// The default forecast of every scored slot, as it would have been
    /// made: oldest day first.
    pub past: Vec<PastForecast<L>>,
}

/// One model on one backtest day.
#[derive(Debug, Default, Clone, PartialEq)]
struct ModelDay {
    fitted: bool,
    failure: Option<MlError>,
    origins: usize,
    /// The model's forecast for each of the day's scored steps.
    values: Vec<Option<f64>>,
}

/// A scored step of a backtest day.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    /// Index of its origin within the day.
    origin: usize,
    /// Slots after the origin (1 = the first).
    horizon: usize,
    bucket: Bucket,
    actual: f64,
}

/// Every model on one backtest day.
#[derive(Debug, Clone, PartialEq)]
struct DayResult {
    /// Origins forecast from by any model.
    origins: usize,
    /// When each of those origins was.
    origin_times: Vec<DateTime<Utc>>,
    steps: Vec<Step>,
    models: Vec<ModelDay>,
}

/// Runs the benchmark of the models `labels` (built by `make`; `None`
/// counts as never fitted) and of the `ensembles` of them, on `history` as
/// of `now`.
///
/// Days are backtested in parallel, each with fresh models; scoring then
/// walks the days in order, so each day's intervals and ensemble weights
/// come from earlier days only.
pub fn run<L, M>(
    labels: &[L],
    make: impl Fn(L) -> Option<M> + Sync,
    ensembles: &[(L, Combination)],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
) -> Benchmark<L>
where
    L: Copy + PartialEq + Send + Sync,
    M: Forecaster + Send + Sync,
{
    run_reporting(
        labels,
        make,
        ensembles,
        history,
        schedule,
        now,
        settings,
        &Defaults::default(),
        &Silent,
    )
}

/// [`run`] with the default forecast's `defaults`, reporting the shared
/// fits and the backtest days to `progress`.
#[allow(clippy::too_many_arguments)]
pub fn run_reporting<L, M>(
    labels: &[L],
    make: impl Fn(L) -> Option<M> + Sync,
    ensembles: &[(L, Combination)],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    defaults: &Defaults<L>,
    progress: &dyn Progress,
) -> Benchmark<L>
where
    L: Copy + PartialEq + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let days = backtest_days(history, schedule, now, settings.days);
    let grid = history.first_time().map_or_else(Grid::default, |first| {
        Grid::build(history, schedule, first - SLOT, now)
    });
    let scale = naive_scale(&grid, schedule, history);
    let mut results = backtest_days_of(
        &days, labels, &make, history, &grid, schedule, now, settings, progress,
    );
    let combinations: Vec<Combination> = ensembles.iter().map(|&(_, c)| c).collect();
    let fitted = add_ensembles(&mut results, labels.len(), &combinations);

    let all_labels: Vec<L> = labels
        .iter()
        .copied()
        .chain(ensembles.iter().map(|&(l, _)| l))
        .collect();
    let walked = walk(results, &all_labels, scale, settings, defaults);
    let ensembles = ensembles
        .iter()
        .zip(fitted)
        .map(|(&(label, _), ensemble)| {
            (
                label,
                ensemble.and_then(|e| e.try_map(|m| labels.get(m).copied())),
            )
        })
        .collect();
    Benchmark {
        days,
        origins: walked.origins,
        reports: walked.reports,
        ensembles,
        past: walked.past,
    }
}

/// What walking the backtest days in order produced.
struct Walked<L> {
    origins: usize,
    reports: Vec<ModelReport<L>>,
    past: Vec<PastForecast<L>>,
}

/// Scores every model (indexed like `labels`) over `results`, oldest day
/// first: each day's intervals, and the selection behind its default
/// forecast, come from the days before it only.
fn walk<L: Copy + PartialEq>(
    results: Vec<DayResult>,
    labels: &[L],
    scale: Option<f64>,
    settings: &Settings,
    defaults: &Defaults<L>,
) -> Walked<L> {
    let mut origins_total = 0;
    let mut reports: Vec<ModelReport<L>> = labels
        .iter()
        .map(|&model| ModelReport {
            model,
            origins: 0,
            fitted_days: 0,
            failure: None,
            scores: [None; 4],
            intervals: [None; 4],
        })
        .collect();
    let mut accumulators = vec![[Accumulator::default(); 4]; labels.len()];
    let mut errors: Vec<[Vec<f64>; 4]> = labels
        .iter()
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();
    let mut past = Vec::new();
    let mut selection: Selection<L> = Vec::new();
    for day in results {
        let intervals: Vec<[Option<Interval>; 4]> = errors
            .iter()
            .map(|by_bucket| std::array::from_fn(|b| Interval::from_errors(&by_bucket[b])))
            .collect();
        let so_far: Vec<ModelReport<L>> = reports
            .iter()
            .zip(&accumulators)
            .map(|(report, accumulator)| ModelReport {
                scores: std::array::from_fn(|b| accumulator[b].score(None)),
                failure: None,
                ..*report
            })
            .collect();
        selection = select(&so_far, &selection, settings);
        past::day_defaults(&day, labels, &selection, &intervals, defaults, &mut past);

        origins_total += day.origins;
        for (i, model_day) in day.models.into_iter().enumerate() {
            let report = &mut reports[i];
            report.origins += model_day.origins;
            report.fitted_days += usize::from(model_day.fitted);
            if model_day.failure.is_some() {
                report.failure = model_day.failure;
            }
            let scored = || {
                day.steps
                    .iter()
                    .zip(&model_day.values)
                    .filter_map(|(step, value)| value.map(|v| (step.bucket, step.actual, v)))
            };
            for (bucket, actual, forecast) in scored() {
                let interval = intervals[i][bucket.index()];
                for b in [bucket, Bucket::Overall] {
                    accumulators[i][b.index()].add(actual, forecast, interval);
                }
            }
            for (bucket, actual, forecast) in scored() {
                errors[i][bucket.index()].push(actual - forecast);
            }
        }
    }
    for ((report, accumulator), errors) in reports.iter_mut().zip(&accumulators).zip(&errors) {
        report.scores = std::array::from_fn(|i| accumulator[i].score(scale));
        report.intervals = std::array::from_fn(|i| {
            (i != Bucket::Overall.index())
                .then(|| Interval::from_errors(&errors[i]))
                .flatten()
        });
    }
    Walked {
        origins: origins_total,
        reports,
        past,
    }
}

/// Backtests every model on every day of `days`, shared fits first.
#[allow(clippy::too_many_arguments)]
fn backtest_days_of<L, M>(
    days: &[NaiveDate],
    labels: &[L],
    make: &(impl Fn(L) -> Option<M> + Sync),
    history: &History,
    grid: &Grid,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    progress: &dyn Progress,
) -> Vec<DayResult>
where
    L: Copy + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let (fits, plan) = shared_fits(
        days, labels, make, history, grid, schedule, settings, progress,
    );
    progress.start(Stage::Backtest, days.len());
    let results = days
        .par_iter()
        .enumerate()
        .map(|(i, &day)| {
            let shared: Vec<DayFit<'_, M>> = plan
                .iter()
                .map(|days| match days {
                    None => DayFit::Fresh,
                    Some(d) => d[i]
                        .and_then(|f| fits.get(f))
                        .map_or(DayFit::Unfitted, DayFit::Shared),
                })
                .collect();
            let result = backtest_day(
                day, labels, make, &shared, history, grid, schedule, now, settings,
            );
            progress.advance(Stage::Backtest);
            result
        })
        .collect();
    progress.finish(Stage::Backtest);
    results
}

/// Appends each ensemble's forecasts to every day of `results` (whose first
/// `members` models are the members), each day combined with weights
/// fitted on the days before it; returns each ensemble fitted on all days.
fn add_ensembles(
    results: &mut [DayResult],
    members: usize,
    combinations: &[Combination],
) -> Vec<Option<Ensemble<usize>>> {
    let indices: Vec<usize> = (0..members).collect();
    let mut rows: [Rows; 3] = std::array::from_fn(|_| Rows::new(members));
    let fit_all = |rows: &[Rows; 3]| -> Vec<Option<Ensemble<usize>>> {
        combinations
            .iter()
            .map(|&c| Ensemble::fit(c, &indices, rows))
            .collect()
    };
    let mut member_values = vec![None; members];
    for day in results.iter_mut() {
        let fitted = fit_all(&rows);
        for ensemble in &fitted {
            let mut model_day = ModelDay {
                fitted: ensemble.is_some(),
                values: Vec::with_capacity(day.steps.len()),
                ..ModelDay::default()
            };
            let mut last_origin = None;
            for (s, step) in day.steps.iter().enumerate() {
                let value = ensemble.as_ref().and_then(|e| {
                    e.combine(step.bucket, |m| {
                        day.models
                            .get(m)
                            .and_then(|d| d.values.get(s).copied().flatten())
                    })
                });
                if value.is_some() && last_origin != Some(step.origin) {
                    model_day.origins += 1;
                    last_origin = Some(step.origin);
                }
                model_day.values.push(value);
            }
            day.models.push(model_day);
        }
        for (s, step) in day.steps.iter().enumerate() {
            for (slot, model) in member_values.iter_mut().zip(&day.models) {
                *slot = model.values.get(s).copied().flatten();
            }
            if let Some(i) = Bucket::HORIZONS.iter().position(|&b| b == step.bucket) {
                rows[i].push(step.actual, &member_values);
            }
        }
    }
    fit_all(&rows)
}

/// A model fitted once and used on several backtest days, with the outcome
/// of its fit.
struct SharedFit<M> {
    model: Option<M>,
    outcome: ModelDay,
}

/// How a model gets its fit for a backtest day.
enum DayFit<'a, M> {
    /// Fitted afresh on the data before the day.
    Fresh,
    /// A fit made on an earlier day (or this one) serves it.
    Shared(&'a SharedFit<M>),
    /// Refits less often than daily and has no fit for the day yet.
    Unfitted,
}

/// Fits the models that refit less often than daily (see
/// [`Forecaster::refit_days`]) once per run of days they serve. Returns
/// the fits and, per label, `None` for a daily model or the fit (index)
/// serving each day.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn shared_fits<L, M>(
    days: &[NaiveDate],
    labels: &[L],
    make: &(impl Fn(L) -> Option<M> + Sync),
    history: &History,
    grid: &Grid,
    schedule: &GymSchedule,
    settings: &Settings,
    progress: &dyn Progress,
) -> (Vec<SharedFit<M>>, Vec<Option<Vec<Option<usize>>>>)
where
    L: Copy + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let fit_history = |day: NaiveDate| {
        let cut = midnight_local_as_utc(day, schedule.timezone());
        (cut, history.range(cut - settings.training_window, cut))
    };
    let mut jobs: Vec<(L, NaiveDate)> = Vec::new();
    let plan = labels
        .iter()
        .map(|&label| {
            let probe = make(label)?;
            let every = probe.refit_days();
            if every <= 1 {
                return None;
            }
            let mut anchor: Option<(NaiveDate, usize)> = None;
            let serving = days
                .iter()
                .map(|&day| {
                    let stale = anchor.is_none_or(|(a, _)| {
                        (day - a).num_days() >= i64::try_from(every).unwrap_or(i64::MAX)
                    });
                    if stale {
                        let (cut, seen) = fit_history(day);
                        if suite::can_fit(&probe, &Observed::new(grid, &seen, schedule, cut)) {
                            anchor = Some((day, jobs.len()));
                            jobs.push((label, day));
                        } else {
                            anchor = None;
                        }
                    }
                    anchor.map(|(_, job)| job)
                })
                .collect();
            Some(serving)
        })
        .collect();
    progress.start(Stage::SharedFits, jobs.len());
    let fits = jobs
        .par_iter()
        .map(|&(label, day)| {
            let (cut, seen) = fit_history(day);
            let mut model = make(label);
            let outcome = fit_for_day(&mut model, &Observed::new(grid, &seen, schedule, cut));
            progress.advance(Stage::SharedFits);
            SharedFit { model, outcome }
        })
        .collect();
    progress.finish(Stage::SharedFits);
    (fits, plan)
}

/// Fits `model` (if any) on `data`; the outcome as a day's starting record.
fn fit_for_day<M: Forecaster>(model: &mut Option<M>, data: &Observed<'_>) -> ModelDay {
    match model {
        None => ModelDay::default(),
        Some(model) => match guarded(|| suite::fit(model, data)) {
            Ok(()) => ModelDay {
                fitted: true,
                ..ModelDay::default()
            },
            Err(MlError::NotEnoughHistory { .. }) => ModelDay::default(),
            Err(e) => ModelDay {
                failure: Some(e),
                ..ModelDay::default()
            },
        },
    }
}

/// Fits fresh models on the data before `day` (or takes the `shared` fit
/// serving it) and forecasts from each of its origins.
#[allow(clippy::too_many_arguments)]
fn backtest_day<L, M>(
    day: NaiveDate,
    labels: &[L],
    make: &(impl Fn(L) -> Option<M> + Sync),
    shared: &[DayFit<'_, M>],
    history: &History,
    grid: &Grid,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
) -> DayResult
where
    L: Copy + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let cut = midnight_local_as_utc(day, schedule.timezone());
    let fit_history = history.range(cut - settings.training_window, cut);
    let fit_data = Observed::new(grid, &fit_history, schedule, cut);
    let mut owned: Vec<Option<M>> = labels
        .iter()
        .zip(shared)
        .map(|(&l, s)| matches!(s, DayFit::Fresh).then(|| make(l)).flatten())
        .collect();
    let mut days: Vec<ModelDay> = owned
        .par_iter_mut()
        .zip(shared)
        .map(|(model, s)| match s {
            DayFit::Fresh => fit_for_day(model, &fit_data),
            DayFit::Unfitted => ModelDay::default(),
            DayFit::Shared(fit) => ModelDay {
                fitted: fit.outcome.fitted,
                failure: fit.outcome.failure.clone(),
                ..ModelDay::default()
            },
        })
        .collect();
    let models: Vec<Option<&M>> = owned
        .iter()
        .zip(shared)
        .map(|(model, s)| match s {
            DayFit::Fresh => model.as_ref(),
            DayFit::Shared(fit) => fit.model.as_ref(),
            DayFit::Unfitted => None,
        })
        .collect();

    let mut origins = 0;
    let mut origin_times = Vec::new();
    let mut steps = Vec::new();
    for origin in day_origins(day, schedule) {
        if origin >= now {
            break;
        }
        let seen = history.until(origin);
        let data = Observed::new(grid, &seen, schedule, origin);
        let forecasts: Vec<Option<Result<Vec<f64>, MlError>>> = models
            .par_iter()
            .zip(&days)
            .map(|(model, d)| {
                model
                    .filter(|_| d.fitted)
                    .map(|m| guarded(|| suite::forecast(m, &data)))
            })
            .collect();
        if !forecasts.iter().any(|f| matches!(f, Some(Ok(_)))) {
            for (model_day, forecast) in days.iter_mut().zip(forecasts) {
                if let Some(Err(e)) = forecast {
                    model_day.failure = Some(e);
                }
            }
            continue;
        }
        let targets: Vec<(usize, f64)> = scored_steps(&data, grid, now).collect();
        steps.extend(targets.iter().map(|&(step, actual)| Step {
            origin: origins,
            horizon: step,
            bucket: Bucket::of_step(step),
            actual,
        }));
        origins += 1;
        origin_times.push(origin);
        for (model_day, forecast) in days.iter_mut().zip(forecasts) {
            match forecast {
                Some(Ok(values)) => {
                    model_day.origins += 1;
                    model_day.values.extend(
                        targets
                            .iter()
                            .map(|&(step, _)| values.get(step - 1).copied()),
                    );
                }
                Some(Err(e)) => {
                    model_day.failure = Some(e);
                    model_day.values.extend(targets.iter().map(|_| None));
                }
                None => model_day.values.extend(targets.iter().map(|_| None)),
            }
        }
    }
    DayResult {
        origins,
        origin_times,
        steps,
        models: days,
    }
}

/// (step, actual) for each step after `data`'s origin whose slot is open,
/// over by `now` and has a reading.
fn scored_steps<'a>(
    data: &'a Observed<'_>,
    grid: &'a Grid,
    now: DateTime<Utc>,
) -> impl Iterator<Item = (usize, f64)> + 'a {
    (1..=HORIZON_STEPS)
        .map(move |step| (step, data.grid().step_end(step)))
        .take_while(move |&(_, end)| end <= now)
        .filter(move |&(_, end)| is_open_slot(data.schedule(), end))
        .filter_map(move |(step, end)| grid.value_at(end).map(|actual| (step, actual)))
}

/// The newest `count` complete gym-local days that have readings, oldest
/// first.
fn backtest_days(
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    count: u32,
) -> Vec<NaiveDate> {
    let tz = schedule.timezone();
    let mut days: Vec<NaiveDate> = Vec::new();
    for (t, _) in history.view().iter() {
        let day = t.with_timezone(&tz).date_naive();
        if days.last() != Some(&day) {
            days.push(day);
        }
    }
    days.retain(|day| {
        day.succ_opt()
            .is_some_and(|next| midnight_local_as_utc(next, tz) <= now)
    });
    let skip = days.len().saturating_sub(count as usize);
    days.split_off(skip)
}

/// Full gym-local hours of `day` from which the next hour or so includes an
/// open slot: opening hours plus the hour before opening.
fn day_origins(day: NaiveDate, schedule: &GymSchedule) -> Vec<DateTime<Utc>> {
    let tz = schedule.timezone();
    (0..24)
        .filter_map(|h| day.and_hms_opt(h, 0, 0))
        .filter_map(|local| tz.from_local_datetime(&local).earliest())
        .map(|t| t.with_timezone(&Utc))
        .filter(|&t| (1..=ORIGIN_LOOKAHEAD_STEPS).any(|k| is_open_slot(schedule, t + SLOT * k)))
        .collect()
}

/// Mean absolute change of open slots against the same time a week before
/// (a day before while there is less than two weeks of history): the error
/// of a seasonal naive forecast, which MASE divides by.
fn naive_scale(grid: &Grid, schedule: &GymSchedule, history: &History) -> Option<f64> {
    let tz = schedule.timezone();
    let span = match (history.first_time(), history.last_time()) {
        (Some(first), Some(last)) => last - first,
        _ => return None,
    };
    let lag_days = if span >= WEEKLY_SCALE_SPAN { 7 } else { 1 };
    let (mut sum, mut n) = (0.0, 0_u32);
    for i in 0..grid.len() {
        let end = grid.end_of(i);
        if !is_open_slot(schedule, end) {
            continue;
        }
        let Some(value) = grid.value_at(end) else {
            continue;
        };
        if let Some(earlier) =
            same_local_time_before(end, lag_days, tz).and_then(|t| grid.value_at(t))
        {
            sum += (value - earlier).abs();
            n += 1;
        }
    }
    (n > 0).then(|| sum / f64::from(n))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    };

    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{Datelike, Timelike};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::HORIZON_STEPS,
        suite::{AnyModel, ModelId},
        test_support::{local, minutely},
    };

    /// Newest data a spy was ever shown, relative to its origin.
    struct Seen {
        newest_seen: AtomicI64,
        newest_allowed: AtomicI64,
    }

    /// Records the newest reading it was shown, to catch leakage.
    struct Spy(Arc<Seen>);

    impl std::ops::Deref for Spy {
        type Target = Seen;

        fn deref(&self) -> &Seen {
            &self.0
        }
    }

    impl Spy {
        fn check(&self, data: &Observed<'_>) {
            if let Some(last) = data.history().last_time() {
                self.newest_seen.fetch_max(
                    last.timestamp() - data.origin().timestamp(),
                    Ordering::Relaxed,
                );
            }
            if let Some((end, _)) = data.grid().last_known() {
                self.newest_allowed.fetch_max(
                    end.timestamp() - data.origin().timestamp(),
                    Ordering::Relaxed,
                );
            }
        }
    }

    impl Forecaster for Spy {
        fn min_history(&self) -> TimeDelta {
            TimeDelta::zero()
        }

        fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
            self.check(data);
            Ok(())
        }

        fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
            self.check(data);
            Ok(vec![50.0; HORIZON_STEPS])
        }
    }

    /// Panics when asked to forecast.
    struct Crashing;

    impl Forecaster for Crashing {
        fn min_history(&self) -> TimeDelta {
            TimeDelta::zero()
        }

        fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
            Ok(())
        }

        fn predict(&self, _: &Observed<'_>) -> Result<Vec<f64>, MlError> {
            let empty: Vec<Vec<f64>> = std::hint::black_box(Vec::new());
            Ok(empty[std::hint::black_box(0)].clone())
        }
    }

    /// Refitted once a week; records the cut of every fit and forecasts
    /// the number of days since it.
    struct Weekly(
        Arc<std::sync::Mutex<Vec<DateTime<Utc>>>>,
        Option<DateTime<Utc>>,
    );

    impl Forecaster for Weekly {
        fn min_history(&self) -> TimeDelta {
            TimeDelta::zero()
        }

        fn refit_days(&self) -> usize {
            7
        }

        fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
            if let Ok(mut cuts) = self.0.lock() {
                cuts.push(data.origin());
            }
            self.1 = Some(data.origin());
            Ok(())
        }

        fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
            let cut = self.1.ok_or_else(|| MlError::Fit("unfitted".to_string()))?;
            #[allow(clippy::cast_precision_loss)]
            let days = (data.origin() - cut).num_days() as f64;
            Ok(vec![days; HORIZON_STEPS])
        }
    }

    #[test]
    fn test_weekly_models_are_fitted_once_per_week_of_backtest_days() -> Result<()> {
        let history = minutely(16, |_| 3.0);
        let schedule = GymSchedule::default();
        let cuts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let result = run(
            &[0],
            |_| Some(Weekly(Arc::clone(&cuts), None)),
            &[],
            &history,
            &schedule,
            local(16, 0, 0),
            &Settings::default(),
        );
        assert_eq!(result.days.len(), 16);
        let mut cuts = cuts
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .clone();
        cuts.sort();
        // Day 0 has no history to fit on, so the weeks start on day 1.
        assert_eq!(cuts, vec![local(1, 0, 0), local(8, 0, 0), local(15, 0, 0)]);
        let report = &result.reports[0];
        assert_eq!(report.fitted_days, 15);
        assert!(report.failure.is_none());
        // Forecasts are 0–6 days after the week's fit, against a flat 3: the
        // mean error over whole weeks of origins sits between 0 and 6.
        let mae = report.score(Bucket::Overall).context("scored")?.mae;
        assert!(mae > 0.5 && mae < 3.5, "MAE {mae}");
        Ok(())
    }

    fn weekly_pattern(t: DateTime<Utc>) -> f64 {
        let local = crate::grid::slot_end_of(t) - TimeDelta::minutes(1);
        let local = local.with_timezone(&GymSchedule::default().timezone());
        20.0 + f64::from(local.weekday().num_days_from_monday()) * 5.0 + f64::from(local.hour())
    }

    #[test]
    fn test_bucket_of_step_and_codes() {
        assert_eq!(Bucket::of_step(1), Bucket::NextHour);
        assert_eq!(Bucket::of_step(4), Bucket::NextHour);
        assert_eq!(Bucket::of_step(5), Bucket::OneToThreeHours);
        assert_eq!(Bucket::of_step(12), Bucket::OneToThreeHours);
        assert_eq!(Bucket::of_step(13), Bucket::ThreeToDay);
        assert_eq!(Bucket::of_step(96), Bucket::ThreeToDay);
        for b in Bucket::ALL {
            assert_eq!(Bucket::from_code(b.code()), Some(b));
            assert_ne!(b.label(), "");
        }
        assert_eq!(Bucket::from_code(9), None);
    }

    #[test]
    fn test_day_origins_cover_opening_hours_and_the_hour_before() {
        // Weekdays 6–23 by default.
        let schedule = GymSchedule::default();
        let day = local(0, 12, 0)
            .with_timezone(&schedule.timezone())
            .date_naive();
        let origins = day_origins(day, &schedule);
        assert_eq!(origins.first(), Some(&local(0, 5, 0)));
        assert_eq!(origins.last(), Some(&local(0, 22, 0)));
        assert_eq!(origins.len(), 18);
    }

    #[test]
    fn test_backtest_days_are_complete_days_with_data() {
        let history = minutely(5, |_| 30.0);
        let schedule = GymSchedule::default();
        let days = backtest_days(&history, &schedule, local(4, 12, 0), 3);
        let expected: Vec<NaiveDate> = (1..=3)
            .map(|d| {
                local(d, 12, 0)
                    .with_timezone(&schedule.timezone())
                    .date_naive()
            })
            .collect();
        assert_eq!(days, expected);
    }

    #[test]
    fn test_benchmark_never_shows_a_model_data_from_after_its_origin() {
        let history = minutely(6, |_| 30.0);
        let schedule = GymSchedule::default();
        let seen = Arc::new(Seen {
            newest_seen: AtomicI64::new(i64::MIN),
            newest_allowed: AtomicI64::new(i64::MIN),
        });
        let result = run(
            &["spy"],
            |_| Some(Spy(Arc::clone(&seen))),
            &[],
            &history,
            &schedule,
            local(6, 0, 0),
            &Settings::default(),
        );
        assert!(result.origins > 0);
        let spy = &seen;
        assert!(spy.newest_seen.load(Ordering::Relaxed) <= 0);
        assert!(spy.newest_allowed.load(Ordering::Relaxed) <= 0);
    }

    #[test]
    fn test_benchmark_scores_a_perfect_model_as_perfect() -> Result<()> {
        // Every week repeats exactly: the weekly seasonal naive is exact.
        let history = minutely(22, weekly_pattern);
        let schedule = GymSchedule::default();
        let result = run(
            &[ModelId::WeeklySeasonalNaive, ModelId::Naive],
            AnyModel::new,
            &[],
            &history,
            &schedule,
            local(22, 0, 0),
            &Settings::default(),
        );
        let weekly = &result.reports[0];
        let naive = &result.reports[1];
        let weekly_score = weekly.score(Bucket::Overall).context("weekly scored")?;
        assert_relative_eq!(weekly_score.mae, 0.0, epsilon = 1e-9);
        assert!(naive.score(Bucket::Overall).context("naive scored")?.mae > 1.0);
        // Week one cannot be fitted by the weekly model; the naive one can.
        assert!(weekly.fitted_days < naive.fitted_days);
        assert!(weekly.is_eligible(&Settings::default()));
        assert!(weekly.failure.is_none());
        // Intervals exist once enough errors were seen.
        assert!(naive.interval(Bucket::NextHour).is_some());
        assert!(naive.interval(Bucket::Overall).is_none());
        Ok(())
    }

    #[test]
    fn test_benchmark_survives_a_crashing_model() {
        let history = minutely(3, |_| 30.0);
        let schedule = GymSchedule::default();
        let result = run(
            &[0],
            |_| Some(Crashing),
            &[],
            &history,
            &schedule,
            local(3, 0, 0),
            &Settings::default(),
        );
        let report = &result.reports[0];
        assert_eq!(report.origins, 0);
        assert!(matches!(report.failure, Some(MlError::Panicked(_))));
    }

    #[test]
    fn test_coverage_uses_only_earlier_days() -> Result<()> {
        let history = minutely(10, |t| weekly_pattern(t) + 3.0 * f64::from(t.minute() % 7));
        let schedule = GymSchedule::default();
        let result = run(
            &[ModelId::Naive],
            AnyModel::new,
            &[],
            &history,
            &schedule,
            local(10, 0, 0),
            &Settings::default(),
        );
        let score = result.reports[0]
            .score(Bucket::NextHour)
            .context("scored")?;
        // Day one has no earlier errors, so not every value had an interval.
        let coverage = score.coverage.context("coverage")?;
        assert!((0.0..=1.0).contains(&coverage));
        assert!(score.mase.is_some());
        Ok(())
    }

    #[test]
    fn test_benchmark_scores_ensembles_from_earlier_days() -> Result<()> {
        // Every week repeats exactly, and the weekly model can be fitted on
        // every backtest day: the best-member ensemble picks it.
        let history = minutely(30, weekly_pattern);
        let settings = Settings {
            days: 21,
            ..Settings::default()
        };
        let models = [ModelId::WeeklySeasonalNaive, ModelId::Naive];
        let ensembles = [
            (ModelId::BestMemberEnsemble, Combination::BestMember),
            (ModelId::MeanEnsemble, Combination::Mean),
        ];
        let result = run(
            &models,
            AnyModel::new,
            &ensembles,
            &history,
            &GymSchedule::default(),
            local(30, 0, 0),
            &settings,
        );
        assert_eq!(result.reports.len(), 4);
        let best = &result.reports[2];
        assert_eq!(best.model, ModelId::BestMemberEnsemble);
        // The first day has no earlier days to fit weights on.
        assert_eq!(best.fitted_days, result.days.len() - 1);
        assert!(best.origins < result.reports[0].origins);
        let mae = best.score(Bucket::Overall).context("scored")?.mae;
        assert_relative_eq!(mae, 0.0, epsilon = 1e-9);
        assert!(best.interval(Bucket::NextHour).is_some());

        let (label, fitted) = &result.ensembles[0];
        assert_eq!(*label, ModelId::BestMemberEnsemble);
        let fitted = fitted.as_ref().context("best member fitted")?;
        for bucket in Bucket::HORIZONS {
            assert_eq!(
                fitted.members(bucket),
                &[(ModelId::WeeklySeasonalNaive, 1.0)]
            );
        }
        let mean = result.ensembles[1].1.as_ref().context("mean fitted")?;
        assert_eq!(mean.members(Bucket::NextHour).len(), 2);
        Ok(())
    }

    #[test]
    fn test_ensembles_without_members_are_not_fitted() {
        let history = minutely(3, |_| 30.0);
        let result = run(
            &[0],
            |_| Some(Crashing),
            &[(1, Combination::Mean)],
            &history,
            &GymSchedule::default(),
            local(3, 0, 0),
            &Settings::default(),
        );
        assert_eq!(result.reports[1].fitted_days, 0);
        assert_eq!(result.reports[1].origins, 0);
        assert_eq!(result.ensembles, vec![(1, None)]);
    }

    /// Backtest days with `members` members forecasting `actual + noise`.
    fn synthetic_days(seeds: &[Vec<(f64, [f64; 3])>]) -> Vec<DayResult> {
        seeds
            .iter()
            .map(|day| {
                let steps: Vec<Step> = day
                    .iter()
                    .enumerate()
                    .map(|(i, &(actual, _))| {
                        let step = 1 + (i * 7) % HORIZON_STEPS;
                        Step {
                            origin: i / 4,
                            horizon: step,
                            bucket: Bucket::of_step(step),
                            actual,
                        }
                    })
                    .collect();
                let origins = steps.last().map_or(0, |s| s.origin + 1);
                let models = (0..3)
                    .map(|m| ModelDay {
                        fitted: true,
                        origins: steps.last().map_or(0, |s| s.origin + 1),
                        values: day
                            .iter()
                            .map(|&(actual, noise)| Some(actual + noise[m]))
                            .collect(),
                        ..ModelDay::default()
                    })
                    .collect();
                DayResult {
                    origins,
                    origin_times: (0..origins)
                        .map(|o| local(0, 6, 0) + TimeDelta::hours(i64::try_from(o).unwrap_or(0)))
                        .collect(),
                    steps,
                    models,
                }
            })
            .collect()
    }

    const ALL_COMBINATIONS: [Combination; 5] = [
        Combination::Mean,
        Combination::Median,
        Combination::InverseMae,
        Combination::BestMember,
        Combination::Stacked,
    ];

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// An ensemble's forecasts on a day never depend on what happened
        /// on that day or later: changing the last day's actual values
        /// changes no ensemble forecast.
        #[test]
        fn test_ensemble_weights_never_see_the_day_they_forecast(
            seeds in prop::collection::vec(
                prop::collection::vec((0.0_f64..100.0, prop::array::uniform3(-10.0_f64..10.0)), 1..24),
                2..5,
            ),
            shift in 1.0_f64..30.0,
        ) {
            let mut days = synthetic_days(&seeds);
            let mut changed = days.clone();
            if let Some(last) = changed.last_mut() {
                for step in &mut last.steps {
                    step.actual += shift;
                }
            }
            add_ensembles(&mut days, 3, &ALL_COMBINATIONS);
            add_ensembles(&mut changed, 3, &ALL_COMBINATIONS);
            for (a, b) in days.iter().zip(&changed) {
                prop_assert_eq!(&a.models[3..], &b.models[3..]);
            }
            // The first day has nothing to fit on.
            prop_assert!(days[0].models[3..].iter().all(|m| !m.fitted && m.origins == 0));
            prop_assert!(days[0].models[3..].iter().all(|m| m.values.iter().all(Option::is_none)));
            for day in &days {
                prop_assert_eq!(day.models.len(), 3 + ALL_COMBINATIONS.len());
                for model in &day.models {
                    prop_assert_eq!(model.values.len(), day.steps.len());
                }
            }
        }

        /// The default forecast of a day never depends on what happened on
        /// that day or later: neither the selection behind it nor its
        /// interval changes when the last day's actual values do.
        #[test]
        fn test_past_defaults_never_see_the_day_they_forecast(
            seeds in prop::collection::vec(
                prop::collection::vec((0.0_f64..100.0, prop::array::uniform3(-10.0_f64..10.0)), 1..24),
                2..6,
            ),
            shift in 1.0_f64..30.0,
        ) {
            let days = synthetic_days(&seeds);
            let mut changed = days.clone();
            if let Some(last) = changed.last_mut() {
                for step in &mut last.steps {
                    step.actual += shift;
                }
            }
            let settings = Settings { min_origins: 1, ..Settings::default() };
            let defaults = Defaults { fallbacks: vec![0], comparison: Some(2) };
            let a = walk(days, &[0, 1, 2], None, &settings, &defaults);
            let b = walk(changed, &[0, 1, 2], None, &settings, &defaults);
            prop_assert_eq!(a.past, b.past);
        }
    }

    #[test]
    fn test_walk_selects_each_day_from_the_days_before() -> Result<()> {
        // Model 0 is exact on day 0 (model 1 off by 3), model 1 on days 1
        // and 2 (model 0 off by 9).
        let mut days = synthetic_days(&[
            vec![(50.0, [0.0, 3.0, 5.0]); 8],
            vec![(50.0, [9.0, 0.0, 5.0]); 8],
            vec![(50.0, [9.0, 0.0, 5.0]); 8],
        ]);
        for day in &mut days {
            for step in &mut day.steps {
                step.horizon = 2;
                step.bucket = Bucket::NextHour;
            }
        }
        let settings = Settings {
            min_origins: 1,
            ..Settings::default()
        };
        let defaults = Defaults {
            fallbacks: vec![2],
            comparison: Some(2),
        };
        let walked = walk(days, &[0, 1, 2], None, &settings, &defaults);
        let models: Vec<usize> = walked.past.iter().map(|p| p.model).collect();
        // Day 0: nothing scored yet, the fallback; day 1: model 0 (best on
        // day 0); day 2: model 1 (best over days 0 and 1 together).
        assert_eq!(models, [vec![2; 8], vec![0; 8], vec![1; 8]].concat());
        let first = walked.past.first().context("no past forecasts")?;
        assert_eq!(first.made_at, local(0, 6, 0));
        assert_eq!(first.target, local(0, 6, 30));
        assert_eq!(first.comparison, Some(55.0));
        assert_eq!(walked.origins, 6);
        assert_eq!(walked.reports[1].origins, 6);
        Ok(())
    }

    #[test]
    fn test_benchmark_records_the_default_forecast_of_each_day() -> Result<()> {
        let history = minutely(6, |_| 40.0);
        let settings = Settings {
            min_origins: 1,
            ..Settings::default()
        };
        let result = run_reporting(
            &[ModelId::Naive, ModelId::MovingAverage],
            AnyModel::new,
            &[],
            &history,
            &GymSchedule::default(),
            local(6, 0, 0),
            &settings,
            &Defaults {
                fallbacks: vec![ModelId::MovingAverage],
                comparison: Some(ModelId::Naive),
            },
            &Silent,
        );
        assert_ne!(result.past.len(), 0);
        let tz = GymSchedule::default().timezone();
        let day_of = |t: DateTime<Utc>| t.with_timezone(&tz).date_naive();
        let first_day = result.days.first().copied().context("no days")?;
        // The first backtest day has nothing before it: the fallback.
        assert!(
            result
                .past
                .iter()
                .filter(|p| day_of(p.made_at) == first_day)
                .all(|p| p.model == ModelId::MovingAverage)
        );
        for p in &result.past {
            assert_eq!(p.made_at.minute(), 0);
            assert_eq!(p.target, p.made_at + SLOT * i32::try_from(p.step)?);
            assert!(p.target <= local(6, 0, 0));
            // Before opening the day's first reading is still unknown.
            assert!((0.0..=40.0).contains(&p.value), "{p:?}");
            assert!(p.comparison.is_some());
        }
        Ok(())
    }
}
