//! The windows the neural networks learn from and forecast with: the last
//! `input` slots before an origin (occupancy, its deviation from the weekly
//! profile and whether the gym was open, scaled to 0–1), what is known in
//! advance about the [`HORIZON_STEPS`] target slots (open, profile value)
//! plus the origin's calendar, and the target deviations with a mask of
//! the slots that count (open, with a value, not after the cut).

use std::f64::consts::TAU;

use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};

use crate::{
    grid::{HORIZON_STEPS, SLOT, floor_to_slot, is_open_slot},
    suite::{
        Observed,
        profiles::{WeeklyProfile, is_weekend_like, slot_date},
    },
};

/// Values per input slot: occupancy, deviation, open.
pub(super) const CHANNELS: usize = 3;
/// Calendar values of the origin: time of day and weekday (sine, cosine),
/// weekend or holiday.
const CALENDAR: usize = 5;
/// Known values per window: open and profile per target slot, then the
/// calendar.
pub(super) const COVARIATES: usize = 2 * HORIZON_STEPS + CALENDAR;
/// Occupancy is scaled from 0–100 to 0–1.
const OCCUPANCY_SCALE: f64 = 100.0;
/// Deviations (inputs and targets) are scaled from points to tens of
/// points, so a typical one is of order 1 like the other inputs.
pub(super) const DEVIATION_SCALE: f64 = 10.0;
/// Origins on the last day before the cut are held out for early stopping.
const VALIDATION: TimeDelta = TimeDelta::days(1);

/// Windows stored flat, window after window.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Windows {
    input: usize,
    /// `input × CHANNELS` per window, oldest slot first.
    pub(super) series: Vec<f32>,
    /// [`COVARIATES`] per window.
    pub(super) covariates: Vec<f32>,
    /// Target deviations, [`HORIZON_STEPS`] per window.
    pub(super) targets: Vec<f32>,
    /// 1 where a target counts, [`HORIZON_STEPS`] per window.
    pub(super) mask: Vec<f32>,
}

impl Windows {
    /// No windows yet.
    pub(super) fn empty(input: usize) -> Self {
        Self {
            input,
            series: Vec::new(),
            covariates: Vec::new(),
            targets: Vec::new(),
            mask: Vec::new(),
        }
    }

    /// Input slots per window.
    pub(super) fn input(&self) -> usize {
        self.input
    }

    /// Number of windows.
    pub(super) fn len(&self) -> usize {
        self.covariates.len() / COVARIATES
    }

    /// Removes the newest window.
    fn drop_last(&mut self) {
        let n = self.len().saturating_sub(1);
        self.series.truncate(n * self.input * CHANNELS);
        self.covariates.truncate(n * COVARIATES);
        self.targets.truncate(n * HORIZON_STEPS);
        self.mask.truncate(n * HORIZON_STEPS);
    }

    /// The windows at `indices`, in that order.
    pub(super) fn select(&self, indices: &[usize]) -> Self {
        let width = self.input * CHANNELS;
        let pick = |values: &[f32], size: usize| {
            let mut out = Vec::with_capacity(indices.len() * size);
            for &i in indices {
                out.extend_from_slice(&values[i * size..(i + 1) * size]);
            }
            out
        };
        Self {
            input: self.input,
            series: pick(&self.series, width),
            covariates: pick(&self.covariates, COVARIATES),
            targets: pick(&self.targets, HORIZON_STEPS),
            mask: pick(&self.mask, HORIZON_STEPS),
        }
    }
}

/// Every slot from `first_end` on: open, profile value and observed value
/// (NaN where unknown to `data`).
struct Slots {
    first_end: DateTime<Utc>,
    open: Vec<bool>,
    profile: Vec<f64>,
    value: Vec<f64>,
}

impl Slots {
    fn build(
        data: &Observed<'_>,
        profile: &WeeklyProfile,
        first_end: DateTime<Utc>,
        len: usize,
    ) -> Self {
        let view = data.grid();
        let mut slots = Self {
            first_end,
            open: Vec::with_capacity(len),
            profile: Vec::with_capacity(len),
            value: Vec::with_capacity(len),
        };
        for i in 0..len {
            let end = first_end + SLOT * i32::try_from(i).unwrap_or(i32::MAX);
            let open = is_open_slot(data.schedule(), end);
            slots.open.push(open);
            slots.profile.push(if open { profile.at(end) } else { 0.0 });
            slots
                .value
                .push(view.value_at(end).filter(|_| open).unwrap_or(f64::NAN));
        }
        slots
    }

    fn index(&self, end: DateTime<Utc>) -> usize {
        usize::try_from((end - self.first_end).num_minutes() / SLOT.num_minutes()).unwrap_or(0)
    }

    /// Appends the window whose newest input slot ends at `origin`; targets
    /// count up to slot index `last_known`.
    #[allow(clippy::cast_possible_truncation)]
    fn push(
        &self,
        out: &mut Windows,
        data: &Observed<'_>,
        origin: DateTime<Utc>,
        last_known: usize,
    ) {
        let at = self.index(origin);
        let scaled = |v: f64| (v / OCCUPANCY_SCALE) as f32;
        let deviation = |v: f64| (v / DEVIATION_SCALE) as f32;
        for i in at + 1 - out.input..=at {
            let (open, typical, value) = (self.open[i], self.profile[i], self.value[i]);
            let known = !value.is_nan();
            out.series.push(scaled(if known { value } else { typical }));
            out.series
                .push(deviation(if known { value - typical } else { 0.0 }));
            out.series.push(f32::from(u8::from(open)));
        }
        for i in at + 1..=at + HORIZON_STEPS {
            out.covariates.push(f32::from(u8::from(self.open[i])));
            out.covariates.push(scaled(self.profile[i]));
            let value = self.value[i];
            let counts = i <= last_known && self.open[i] && !value.is_nan();
            out.targets.push(if counts {
                deviation(value - self.profile[i])
            } else {
                0.0
            });
            out.mask.push(f32::from(u8::from(counts)));
        }
        let tz = data.schedule().timezone();
        let local = (origin + SLOT - TimeDelta::minutes(1)).with_timezone(&tz);
        let hour = f64::from(local.hour()) + f64::from(local.minute()) / 60.0;
        let weekday = f64::from(local.weekday().num_days_from_monday());
        for value in [
            (TAU * hour / 24.0).sin(),
            (TAU * hour / 24.0).cos(),
            (TAU * weekday / 7.0).sin(),
            (TAU * weekday / 7.0).cos(),
            f64::from(u8::from(is_weekend_like(slot_date(origin + SLOT, tz)))),
        ] {
            out.covariates.push(value as f32);
        }
    }
}

/// The window for a forecast from `data`'s origin (its targets are empty).
pub(super) fn forecast_window(
    data: &Observed<'_>,
    profile: &WeeklyProfile,
    input: usize,
) -> Windows {
    let origin = data.origin();
    let first_end = origin - SLOT * i32::try_from(input - 1).unwrap_or(i32::MAX);
    let slots = Slots::build(data, profile, first_end, input + HORIZON_STEPS);
    let mut out = Windows::empty(input);
    slots.push(&mut out, data, origin, 0);
    out
}

/// One window per slot origin from `lead` after the first reading up to
/// `data`'s origin (the cut), as (training, validation): the validation
/// windows are those of the last day. Windows without a target that counts
/// are left out.
pub(super) fn training_windows(
    data: &Observed<'_>,
    profile: &WeeklyProfile,
    input: usize,
    lead: TimeDelta,
) -> (Windows, Windows) {
    let (mut train, mut validation) = (Windows::empty(input), Windows::empty(input));
    let cut = data.origin();
    let Some(first) = data.history().first_time() else {
        return (train, validation);
    };
    let start = floor_to_slot(first + lead) + SLOT;
    if start >= cut {
        return (train, validation);
    }
    let first_end = start - SLOT * i32::try_from(input - 1).unwrap_or(i32::MAX);
    let len = usize::try_from((cut - first_end).num_minutes() / SLOT.num_minutes()).unwrap_or(0)
        + HORIZON_STEPS;
    let slots = Slots::build(data, profile, first_end, len);
    let last_known = slots.index(cut);
    let mut origin = start;
    while origin < cut {
        let target = if origin >= cut - VALIDATION {
            &mut validation
        } else {
            &mut train
        };
        slots.push(target, data, origin, last_known);
        let mask = &target.mask[target.mask.len() - HORIZON_STEPS..];
        if mask.iter().all(|&m| m == 0.0) {
            target.drop_last();
        }
        origin += SLOT;
    }
    (train, validation)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use hardy_core::GymSchedule;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::Grid,
        history::History,
        test_support::{local, minutely},
    };

    #[test]
    fn test_forecast_window_scales_inputs_and_knows_the_targets_calendar() {
        let schedule = GymSchedule::default();
        let history = minutely(3, |_| 40.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let origin = local(2, 12, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let window = forecast_window(&data, &profile, 96);
        assert_eq!(window.len(), 1);
        assert_eq!(window.series.len(), 96 * CHANNELS);
        assert_eq!(window.covariates.len(), COVARIATES);
        // Newest slot: open at 40 with no deviation.
        let newest = &window.series[95 * CHANNELS..];
        assert_relative_eq!(newest[0], 0.4);
        assert_relative_eq!(newest[1], 0.0);
        assert_relative_eq!(newest[2], 1.0);
        // The slot ending 01:00 the night before is closed.
        let night = &window.series[(95 - 44) * CHANNELS..(96 - 44) * CHANNELS];
        assert_eq!(night, &[0.0, 0.0, 0.0][..], "{night:?}");
        // First target slot open with profile 40, a Wednesday at noon.
        assert_relative_eq!(window.covariates[0], 1.0);
        assert_relative_eq!(window.covariates[1], 0.4);
        let calendar = &window.covariates[2 * HORIZON_STEPS..];
        // The first target slot, 12:00–12:15, read at 12:14.
        let hour = 12.0 + 14.0 / 60.0;
        #[allow(clippy::cast_possible_truncation)]
        let expected = (TAU * hour / 24.0).sin() as f32;
        assert_relative_eq!(calendar[0], expected, epsilon = 1e-6);
        assert_relative_eq!(calendar[4], 0.0);
        assert!(window.mask.iter().all(|&m| m < 0.5));
    }

    #[test]
    fn test_training_windows_mask_the_cut_and_hold_out_the_last_day() {
        let schedule = GymSchedule::default();
        // Monday again on day 7, and 20 higher.
        let history = minutely(8, |t| if t >= local(7, 0, 0) { 50.0 } else { 30.0 });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(8, 0, 0));
        let cut = local(8, 0, 0);
        let data = Observed::new(&grid, &history, &schedule, cut);
        let profile = WeeklyProfile::fit(&data);
        let (train, validation) = training_windows(&data, &profile, 96, TimeDelta::days(1));
        assert!(train.len() > 0 && validation.len() > 0);
        // Every window has a target that counts, and none after the cut:
        // the newest validation window starts within the closed night.
        for windows in [&train, &validation] {
            for mask in windows.mask.chunks(HORIZON_STEPS) {
                assert!(mask.iter().any(|&m| m > 0.5));
            }
        }
        // Monday's profile is the mean of 30 and 50: day 7 is 10 above,
        // which is 1 in tens of points.
        let last = validation.len() - 1;
        let targets = &validation.targets[last * HORIZON_STEPS..];
        let mask = &validation.mask[last * HORIZON_STEPS..];
        for (t, m) in targets.iter().zip(mask) {
            if *m > 0.5 {
                assert_relative_eq!(*t, 1.0, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_select_picks_whole_windows() {
        let schedule = GymSchedule::default();
        let history = minutely(3, |t| f64::from(t.minute()));
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(3, 0, 0));
        let profile = WeeklyProfile::fit(&data);
        let (train, _) = training_windows(&data, &profile, 8, TimeDelta::hours(6));
        let picked = train.select(&[2, 0]);
        assert_eq!(picked.len(), 2);
        assert_eq!(
            picked.covariates[..COVARIATES],
            train.covariates[2 * COVARIATES..3 * COVARIATES]
        );
        assert_eq!(picked.series[8 * CHANNELS..], train.series[..8 * CHANNELS]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Changing anything after the origin changes no input.
        #[test]
        fn test_windows_never_see_after_the_origin(
            later in -20.0_f64..20.0,
            minute in 0_u32..4,
            hour in 0_u32..24,
        ) {
            let schedule = GymSchedule::default();
            let origin = local(2, hour, minute * 15);
            let base = minutely(3, |t| 30.0 + f64::from(t.minute() % 5));
            let changed = History::new(
                base.view()
                    .iter()
                    .map(|(t, v)| (t, if t > origin { v + later } else { v }))
                    .collect(),
            );
            let window_of = |history: &History| {
                let grid = Grid::build(history, &schedule, local(0, 0, 0), local(3, 0, 0));
                let seen = history.until(origin);
                let data = Observed::new(&grid, &seen, &schedule, origin);
                let profile = WeeklyProfile::fit(&data);
                forecast_window(&data, &profile, 96)
            };
            prop_assert_eq!(window_of(&base), window_of(&changed));
        }
    }
}
