//! Slot-level features: what is known at an origin about the deviation from
//! the weekly profile, plus calendar facts about the target slot. Everything
//! is read through a [`GridView`] cut at the origin, so nothing after it can
//! leak in.

use std::f64::consts::TAU;

use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};
use hardy_core::GymSchedule;

use crate::{
    benchmark::Bucket,
    grid::{GridView, HORIZON_STEPS, SLOT, floor_to_slot, is_open_slot},
    suite::{
        Observed,
        profiles::{WeeklyProfile, is_weekend_like, slot_date},
    },
};

/// Features per row.
pub const FEATURES: usize = 16;

/// One feature row.
pub type Row = [f64; FEATURES];

/// Training rows of one horizon bucket.
#[derive(Debug, Default, Clone)]
pub struct Rows {
    pub(super) x: Vec<Row>,
    pub(super) y: Vec<f64>,
}

/// The deviation from `profile` of the open slot ending at `end`, if `view`
/// knows a value for it.
fn deviation(
    view: &GridView<'_>,
    profile: &WeeklyProfile,
    schedule: &GymSchedule,
    end: DateTime<Utc>,
) -> Option<f64> {
    if !is_open_slot(schedule, end) {
        return None;
    }
    view.value_at(end).map(|v| v - profile.at(end))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0_u32), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / f64::from(n))
}

/// What is known at an origin, shared by all its target slots.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Context {
    /// Newest deviation within the last hour (0 without one).
    last: f64,
    /// 1 if there is a deviation within the last hour.
    recent: f64,
    mean_1h: f64,
    mean_3h: f64,
    /// Newest deviation minus the one an hour before it.
    trend: f64,
    /// Mean deviation of the origin's day so far.
    today: f64,
}

impl Context {
    fn at(view: &GridView<'_>, profile: &WeeklyProfile, schedule: &GymSchedule) -> Self {
        let origin = view.origin();
        let back = |slots: usize| origin - SLOT * i32::try_from(slots).unwrap_or(i32::MAX);
        let recent: [Option<f64>; 12] =
            std::array::from_fn(|i| deviation(view, profile, schedule, back(i)));
        let newest = recent[..4]
            .iter()
            .enumerate()
            .find_map(|(i, d)| d.map(|d| (i, d)));
        let (last, has_recent) = newest.map_or((0.0, 0.0), |(_, d)| (d, 1.0));
        let mean_1h = mean(recent[..4].iter().flatten().copied()).unwrap_or(last);
        let mean_3h = mean(recent.iter().flatten().copied()).unwrap_or(mean_1h);
        let trend = newest
            .and_then(|(i, d)| {
                deviation(view, profile, schedule, back(i + 4)).map(|before| d - before)
            })
            .unwrap_or(0.0);
        let tz = schedule.timezone();
        let day = slot_date(origin, tz);
        let today = mean(
            (0..96)
                .map(back)
                .take_while(|end| slot_date(*end, tz) == day)
                .filter_map(|end| deviation(view, profile, schedule, end)),
        )
        .unwrap_or(0.0);
        Self {
            last,
            recent: has_recent,
            mean_1h,
            mean_3h,
            trend,
            today,
        }
    }
}

/// The features for the `step`-th slot after `view`'s origin.
fn row(
    context: &Context,
    view: &GridView<'_>,
    profile: &WeeklyProfile,
    schedule: &GymSchedule,
    step: usize,
) -> Row {
    let end = view.step_end(step);
    let tz = schedule.timezone();
    let local = (end - TimeDelta::minutes(1)).with_timezone(&tz);
    let hour = f64::from(local.hour()) + f64::from(local.minute()) / 60.0;
    let weekday = f64::from(local.weekday().num_days_from_monday());
    #[allow(clippy::cast_precision_loss)]
    let horizon = step as f64 / 4.0;
    #[allow(clippy::cast_precision_loss)]
    let to_close = (schedule.next_closing_after(end - TimeDelta::minutes(1)) - end)
        .num_minutes()
        .max(0) as f64
        / 60.0;
    let same_time =
        |days: i64| deviation(view, profile, schedule, end - TimeDelta::days(days)).unwrap_or(0.0);
    [
        horizon,
        (TAU * hour / 24.0).sin(),
        (TAU * hour / 24.0).cos(),
        (TAU * weekday / 7.0).sin(),
        (TAU * weekday / 7.0).cos(),
        f64::from(u8::from(is_weekend_like(slot_date(end, tz)))),
        to_close,
        profile.at(end),
        context.last,
        context.recent,
        context.mean_1h,
        context.mean_3h,
        context.trend,
        context.today,
        same_time(1),
        same_time(7),
    ]
}

/// Feature rows for every open slot of the horizon after `data`'s origin, as
/// (step, row).
pub(super) fn forecast_rows(data: &Observed<'_>, profile: &WeeklyProfile) -> Vec<(usize, Row)> {
    let view = data.grid();
    let context = Context::at(&view, profile, data.schedule());
    (1..=HORIZON_STEPS)
        .filter(|&step| is_open_slot(data.schedule(), view.step_end(step)))
        .map(|step| (step, row(&context, &view, profile, data.schedule(), step)))
        .collect()
}

/// Training rows per horizon bucket (order of [`Bucket::HORIZONS`]): from
/// every whole hour in `data` as an origin, each open slot after it that
/// has a value by `data`'s origin; beyond three hours only every fourth
/// step, which keeps the bucket's size in line with the others.
pub(super) fn training_rows(data: &Observed<'_>, profile: &WeeklyProfile) -> [Rows; 3] {
    let mut rows: [Rows; 3] = std::array::from_fn(|_| Rows::default());
    let all = data.grid();
    let schedule = data.schedule();
    let Some(first) = data.history().first_time() else {
        return rows;
    };
    let mut origin = floor_to_slot(first);
    while origin.minute() != 0 {
        origin += SLOT;
    }
    while origin < data.origin() {
        let view = all.until(origin);
        let context = Context::at(&view, profile, schedule);
        for step in 1..=HORIZON_STEPS {
            let end = view.step_end(step);
            if end > data.origin() {
                break;
            }
            let bucket = Bucket::of_step(step);
            if bucket == Bucket::ThreeToDay && step % 4 != 0 {
                continue;
            }
            let Some(actual) = deviation(&all, profile, schedule, end) else {
                continue;
            };
            if let Some(i) = Bucket::HORIZONS.iter().position(|&b| b == bucket) {
                rows[i]
                    .x
                    .push(row(&context, &view, profile, schedule, step));
                rows[i].y.push(actual);
            }
        }
        origin += TimeDelta::hours(1);
    }
    rows
}

#[cfg(test)]
mod tests {
    use anyhow::{Context as _, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::Grid,
        history::History,
        test_support::{local, minutely},
    };

    /// 30 every day except on day 9 from 09:00, where it is 30 + offset.
    fn history(offset: f64) -> History {
        minutely(10, move |t| {
            if t >= local(9, 9, 0) {
                30.0 + offset
            } else {
                30.0
            }
        })
    }

    fn setup(history: &History, origin: DateTime<Utc>) -> (Grid, History) {
        let schedule = GymSchedule::default();
        (
            Grid::build(history, &schedule, local(0, 0, 0), origin),
            history.until(origin),
        )
    }

    #[test]
    fn test_context_and_rows_on_a_known_series() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = history(6.0);
        let origin = local(9, 12, 0);
        let (grid, seen) = setup(&history, origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        // The profile of days 0–8 only, so it is 30 everywhere.
        let profile_history = history.until(local(9, 0, 0));
        let profile = WeeklyProfile::fit(&Observed::new(
            &grid,
            &profile_history,
            &schedule,
            local(9, 0, 0),
        ));
        let view = data.grid();
        let context = Context::at(&view, &profile, &schedule);
        assert_relative_eq!(context.last, 6.0, epsilon = 1e-9);
        assert_eq!(context.recent, 1.0);
        assert_relative_eq!(context.mean_1h, 6.0, epsilon = 1e-9);
        assert_relative_eq!(context.trend, 0.0, epsilon = 1e-9);
        // Today: 06:00–09:00 at 0, 09:00–12:00 at 6 (the 09:00 reading
        // falls in the slot ending 09:00, hence not exactly 3).
        assert_relative_eq!(context.today, 3.0, epsilon = 0.05);

        let rows = forecast_rows(&data, &profile);
        // Open until 23:00: 44 slots today, then tomorrow from 06:00.
        let (step, first) = rows.first().context("a row")?;
        assert_eq!(*step, 1);
        assert_relative_eq!(first[0], 0.25);
        assert_relative_eq!(first[7], 30.0, epsilon = 1e-9);
        // Same time yesterday and last week: deviation 0.
        assert_relative_eq!(first[14], 0.0, epsilon = 1e-9);
        assert_relative_eq!(first[15], 0.0, epsilon = 1e-9);
        assert!(rows.iter().all(|(s, _)| (1..=HORIZON_STEPS).contains(s)));
        assert_eq!(rows.iter().filter(|(s, _)| *s <= 44).count(), 44);
        Ok(())
    }

    #[test]
    fn test_training_rows_by_bucket_with_thinning() {
        let schedule = GymSchedule::default();
        let history = minutely(3, |_| 30.0);
        let origin = local(3, 0, 0);
        let (grid, seen) = setup(&history, origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let [next_hour, one_to_three, rest] = training_rows(&data, &profile);
        for rows in [&next_hour, &one_to_three, &rest] {
            assert_eq!(rows.x.len(), rows.y.len());
            assert_ne!(rows.x.len(), 0);
            assert!(
                rows.y.iter().all(|y| y.abs() < 1e-9),
                "flat data: no deviation"
            );
        }
        assert!(next_hour.x.iter().all(|r| r[0] <= 1.0));
        assert!(one_to_three.x.iter().all(|r| r[0] > 1.0 && r[0] <= 3.0));
        // Beyond three hours only whole hours ahead.
        assert!(rest.x.iter().all(|r| r[0] > 3.0 && r[0].fract() == 0.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Changing anything after the origin changes no feature.
        #[test]
        fn test_features_never_see_after_the_origin(
            later in -20.0_f64..20.0,
            minute in 0_u32..4,
            hour in 6_u32..22,
        ) {
            let schedule = GymSchedule::default();
            let origin = local(2, hour, minute * 15);
            let base = minutely(3, |_| 30.0);
            let changed = History::new(
                base.view()
                    .iter()
                    .map(|(t, v)| (t, if t > origin { v + later } else { v }))
                    .collect(),
            );
            let rows_of = |history: &History| {
                let grid = Grid::build(history, &schedule, local(0, 0, 0), local(3, 0, 0));
                let seen = history.until(origin);
                let data = Observed::new(&grid, &seen, &schedule, origin);
                let profile = WeeklyProfile::fit(&data);
                forecast_rows(&data, &profile)
            };
            prop_assert_eq!(rows_of(&base), rows_of(&changed));
        }
    }
}
