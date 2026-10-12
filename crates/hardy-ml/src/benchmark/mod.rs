//! Rolling-origin benchmark: how well each model would have forecast the
//! last weeks, seeing only what was known at the time.
//!
//! For each backtest day, every model is fitted on the data before that
//! day's midnight, then forecasts from every full hour of the day around
//! opening hours. Each forecast is scored against the slots that followed,
//! per horizon bucket. Intervals are conformal: a model's interval on a day
//! comes from its own errors on the days before, so coverage is measured
//! honestly; the interval for live forecasts comes from all its errors.

mod metrics;
mod selection;

use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use hardy_core::{GymSchedule, analytics::midnight_local_as_utc};
use metrics::Accumulator;
pub use metrics::{INTERVAL_TAIL, Interval, MIN_INTERVAL_ERRORS, Score};
use rayon::prelude::*;
pub use selection::{Selection, select};

use crate::{
    error::MlError,
    grid::{Grid, HORIZON_STEPS, SLOT, is_open_slot, same_local_time_before},
    history::History,
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
    pub reports: Vec<ModelReport<L>>,
}

/// One model on one backtest day.
#[derive(Default)]
struct ModelDay {
    fitted: bool,
    failure: Option<MlError>,
    origins: usize,
    /// (bucket, actual, forecast) of every scored step.
    scored: Vec<(Bucket, f64, f64)>,
}

/// Every model on one backtest day.
struct DayResult {
    /// Origins forecast from by any model.
    origins: usize,
    models: Vec<ModelDay>,
}

/// Runs the benchmark of the models `labels` (built by `make`) on `history`
/// as of `now`.
///
/// Days are backtested in parallel, each with fresh models; scoring then
/// walks the days in order, so each day's intervals come from earlier days'
/// errors only.
pub fn run<L, M>(
    labels: &[L],
    make: impl Fn(L) -> M + Sync,
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
) -> Benchmark<L>
where
    L: Copy + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let days = backtest_days(history, schedule, now, settings.days);
    let grid = history.first_time().map_or_else(Grid::default, |first| {
        Grid::build(history, schedule, first - SLOT, now)
    });
    let scale = naive_scale(&grid, schedule, history);
    let results: Vec<DayResult> = days
        .par_iter()
        .map(|&day| backtest_day(day, labels, &make, history, &grid, schedule, now, settings))
        .collect();

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
    for day in results {
        origins_total += day.origins;
        for (i, model_day) in day.models.into_iter().enumerate() {
            let report = &mut reports[i];
            report.origins += model_day.origins;
            report.fitted_days += usize::from(model_day.fitted);
            if model_day.failure.is_some() {
                report.failure = model_day.failure;
            }
            let intervals: [Option<Interval>; 4] =
                std::array::from_fn(|b| Interval::from_errors(&errors[i][b]));
            for &(bucket, actual, forecast) in &model_day.scored {
                let interval = intervals[bucket.index()];
                for b in [bucket, Bucket::Overall] {
                    accumulators[i][b.index()].add(actual, forecast, interval);
                }
            }
            for (bucket, actual, forecast) in model_day.scored {
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
    Benchmark {
        days,
        origins: origins_total,
        reports,
    }
}

/// Fits fresh models on the data before `day` and forecasts from each of
/// its origins.
#[allow(clippy::too_many_arguments)]
fn backtest_day<L, M>(
    day: NaiveDate,
    labels: &[L],
    make: &(impl Fn(L) -> M + Sync),
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
    let mut models: Vec<M> = labels.iter().map(|&l| make(l)).collect();
    let mut days: Vec<ModelDay> = models
        .par_iter_mut()
        .map(|model| match guarded(|| suite::fit(model, &fit_data)) {
            Ok(()) => ModelDay {
                fitted: true,
                ..ModelDay::default()
            },
            Err(MlError::NotEnoughHistory { .. }) => ModelDay::default(),
            Err(e) => ModelDay {
                failure: Some(e),
                ..ModelDay::default()
            },
        })
        .collect();

    let mut origins = 0;
    for origin in day_origins(day, schedule) {
        if origin >= now {
            break;
        }
        let seen = history.until(origin);
        let data = Observed::new(grid, &seen, schedule, origin);
        let forecasts: Vec<Option<Result<Vec<f64>, MlError>>> = models
            .par_iter()
            .zip(&days)
            .map(|(model, d)| d.fitted.then(|| guarded(|| suite::forecast(model, &data))))
            .collect();
        let mut any = false;
        for (model_day, forecast) in days.iter_mut().zip(forecasts) {
            match forecast {
                Some(Ok(values)) => {
                    any = true;
                    model_day.origins += 1;
                    model_day.scored.extend(
                        scored_steps(&values, &data, grid, now)
                            .map(|(step, value, actual)| (Bucket::of_step(step), actual, value)),
                    );
                }
                Some(Err(e)) => model_day.failure = Some(e),
                None => {}
            }
        }
        origins += usize::from(any);
    }
    DayResult {
        origins,
        models: days,
    }
}

/// (step, forecast, actual) for each step of a forecast from `data`'s
/// origin whose slot is open, over by `now` and has a reading.
fn scored_steps<'a>(
    values: &'a [f64],
    data: &'a Observed<'_>,
    grid: &'a Grid,
    now: DateTime<Utc>,
) -> impl Iterator<Item = (usize, f64, f64)> + 'a {
    (1..=HORIZON_STEPS)
        .zip(values.iter().copied())
        .map(move |(step, value)| (step, value, data.grid().step_end(step)))
        .take_while(move |&(_, _, end)| end <= now)
        .filter(move |&(_, _, end)| is_open_slot(data.schedule(), end))
        .filter_map(move |(step, value, end)| {
            grid.value_at(end).map(|actual| (step, value, actual))
        })
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
            |_| Spy(Arc::clone(&seen)),
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
            |_| Crashing,
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
}
