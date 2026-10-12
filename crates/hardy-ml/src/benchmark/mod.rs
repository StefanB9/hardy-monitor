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

/// Per-model running state.
struct Tally {
    origins: usize,
    fitted_days: usize,
    failure: Option<MlError>,
    accumulators: [Accumulator; 4],
    /// Errors (actual − forecast) of earlier days, per bucket.
    errors: [Vec<f64>; 4],
}

impl Tally {
    fn new() -> Self {
        Self {
            origins: 0,
            fitted_days: 0,
            failure: None,
            accumulators: [Accumulator::default(); 4],
            errors: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn note(&mut self, result: Result<(), MlError>) -> bool {
        match result {
            Ok(()) => true,
            Err(MlError::NotEnoughHistory { .. }) => false,
            Err(e) => {
                self.failure = Some(e);
                false
            }
        }
    }
}

/// Runs the benchmark of `models` on `history` as of `now`.
pub fn run<L, M>(
    models: &mut [(L, M)],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
) -> Benchmark<L>
where
    L: Copy + Send + Sync,
    M: Forecaster + Send + Sync,
{
    let tz = schedule.timezone();
    let days = backtest_days(history, schedule, now, settings.days);
    let grid = history.first_time().map_or_else(Grid::default, |first| {
        Grid::build(history, schedule, first - SLOT, now)
    });
    let scale = naive_scale(&grid, schedule, history);
    let mut tallies: Vec<Tally> = models.iter().map(|_| Tally::new()).collect();
    let mut origins_total = 0;

    for &day in &days {
        let cut = midnight_local_as_utc(day, tz);
        let fit_history = history.range(cut - settings.training_window, cut);
        let fit_data = Observed::new(&grid, &fit_history, schedule, cut);
        let fitted: Vec<Result<(), MlError>> = models
            .par_iter_mut()
            .map(|(_, model)| guarded(|| suite::fit(model, &fit_data)))
            .collect();
        let fitted: Vec<bool> = tallies
            .iter_mut()
            .zip(fitted)
            .map(|(tally, result)| tally.note(result))
            .collect();
        let intervals: Vec<[Option<Interval>; 4]> = tallies
            .iter()
            .map(|tally| std::array::from_fn(|i| Interval::from_errors(&tally.errors[i])))
            .collect();
        let mut day_errors: Vec<[Vec<f64>; 4]> = models
            .iter()
            .map(|_| std::array::from_fn(|_| Vec::new()))
            .collect();

        for origin in day_origins(day, schedule) {
            if origin >= now {
                break;
            }
            let seen = history.until(origin);
            let data = Observed::new(&grid, &seen, schedule, origin);
            let forecasts: Vec<Option<Result<Vec<f64>, MlError>>> = models
                .par_iter()
                .zip(&fitted)
                .map(|((_, model), &ok)| ok.then(|| guarded(|| suite::forecast(model, &data))))
                .collect();
            let mut any = false;
            for (i, forecast) in forecasts.into_iter().enumerate() {
                let values = match forecast {
                    Some(Ok(values)) => values,
                    Some(Err(e)) => {
                        tallies[i].failure = Some(e);
                        continue;
                    }
                    None => continue,
                };
                any = true;
                tallies[i].origins += 1;
                for (step, value, actual) in scored_steps(&values, &data, &grid, now) {
                    let bucket = Bucket::of_step(step);
                    let interval = intervals[i][bucket.index()];
                    for b in [bucket, Bucket::Overall] {
                        tallies[i].accumulators[b.index()].add(actual, value, interval);
                    }
                    day_errors[i][bucket.index()].push(actual - value);
                }
            }
            origins_total += usize::from(any);
        }

        for ((tally, errors), &ok) in tallies.iter_mut().zip(day_errors).zip(&fitted) {
            tally.fitted_days += usize::from(ok);
            for (all, new) in tally.errors.iter_mut().zip(errors) {
                all.extend(new);
            }
        }
    }

    let reports = models
        .iter()
        .zip(tallies)
        .map(|((label, _), tally)| ModelReport {
            model: *label,
            origins: tally.origins,
            fitted_days: tally.fitted_days,
            scores: std::array::from_fn(|i| tally.accumulators[i].score(scale)),
            intervals: std::array::from_fn(|i| {
                (i != Bucket::Overall.index())
                    .then(|| Interval::from_errors(&tally.errors[i]))
                    .flatten()
            }),
            failure: tally.failure,
        })
        .collect();
    Benchmark {
        days,
        origins: origins_total,
        reports,
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
    use std::sync::atomic::{AtomicI64, Ordering};

    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{Datelike, Timelike};

    use super::*;
    use crate::{
        grid::HORIZON_STEPS,
        suite::{AnyModel, ModelId},
        test_support::{local, minutely},
    };

    /// Records the newest reading it was ever shown, to catch leakage.
    struct Spy {
        newest_seen: AtomicI64,
        newest_allowed: AtomicI64,
    }

    impl Spy {
        fn new() -> Self {
            Self {
                newest_seen: AtomicI64::new(i64::MIN),
                newest_allowed: AtomicI64::new(i64::MIN),
            }
        }

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
        let mut models = [("spy", Spy::new())];
        let result = run(
            &mut models,
            &history,
            &schedule,
            local(6, 0, 0),
            &Settings::default(),
        );
        assert!(result.origins > 0);
        let spy = &models[0].1;
        assert!(spy.newest_seen.load(Ordering::Relaxed) <= 0);
        assert!(spy.newest_allowed.load(Ordering::Relaxed) <= 0);
    }

    #[test]
    fn test_benchmark_scores_a_perfect_model_as_perfect() -> Result<()> {
        // Every week repeats exactly: the weekly seasonal naive is exact.
        let history = minutely(22, weekly_pattern);
        let schedule = GymSchedule::default();
        let mut models = [
            (
                ModelId::WeeklySeasonalNaive,
                AnyModel::new(ModelId::WeeklySeasonalNaive),
            ),
            (ModelId::Naive, AnyModel::new(ModelId::Naive)),
        ];
        let result = run(
            &mut models,
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
        let mut models = [(0, Crashing)];
        let result = run(
            &mut models,
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
        let mut models = [(ModelId::Naive, AnyModel::new(ModelId::Naive))];
        let result = run(
            &mut models,
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
