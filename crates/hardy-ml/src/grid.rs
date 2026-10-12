//! Occupancy as a regular series of 15-minute slots around the clock.
//!
//! A slot is named by its end `t` and covers `(t − 15 min, t]`: its value is
//! the mean of the readings in that span, so it is known at `t`. Slots while
//! the gym is closed are 0; open slots without a reading are missing (NaN).
//! A forecast made at `t` sees only the slots ending at or before `t`
//! ([`Grid::view_until`]), so it cannot read data from after it was made.

use chrono::{DateTime, DurationRound, TimeDelta, Utc};
use hardy_core::{GymSchedule, Tz};

use crate::history::History;

/// Length of one slot.
pub const SLOT: TimeDelta = TimeDelta::minutes(15);
/// Slots per day (a day without a DST change).
pub const SLOTS_PER_DAY: usize = 96;
/// Slots per week (a week without a DST change).
pub const SLOTS_PER_WEEK: usize = 7 * SLOTS_PER_DAY;
/// Steps every model forecasts: 24 hours of slots.
pub const HORIZON_STEPS: usize = SLOTS_PER_DAY;

/// `t` rounded down to a slot boundary.
pub fn floor_to_slot(t: DateTime<Utc>) -> DateTime<Utc> {
    t.duration_trunc(SLOT).unwrap_or(t)
}

/// End of the slot containing `t` (a boundary itself ends a slot).
pub fn slot_end_of(t: DateTime<Utc>) -> DateTime<Utc> {
    let floor = floor_to_slot(t);
    if floor == t { t } else { floor + SLOT }
}

/// Whether the slot ending at `end` is an open slot: judged at its middle,
/// so the slot starting at closing time counts as closed.
pub fn is_open_slot(schedule: &GymSchedule, end: DateTime<Utc>) -> bool {
    schedule.is_open(&(end - SLOT / 2))
}

/// The same gym-local wall-clock time `days` days before `t` (so seasonal
/// lags stay correct across DST changes); `None` if that time does not
/// exist locally.
pub fn same_local_time_before(t: DateTime<Utc>, days: u32, tz: Tz) -> Option<DateTime<Utc>> {
    let local = t.with_timezone(&tz).naive_local() - TimeDelta::days(i64::from(days));
    local
        .and_local_timezone(tz)
        .earliest()
        .map(|t| t.with_timezone(&Utc))
}

/// Index of the slot ending at `end` in a series of `len` slots from
/// `first_end`.
fn index_in(first_end: DateTime<Utc>, len: usize, end: DateTime<Utc>) -> Option<usize> {
    let offset = end - first_end;
    let slot_minutes = SLOT.num_minutes();
    if offset < TimeDelta::zero() || offset.num_seconds() % (slot_minutes * 60) != 0 {
        return None;
    }
    usize::try_from(offset.num_minutes() / slot_minutes)
        .ok()
        .filter(|&i| i < len)
}

/// Slot values from `first_end` on, one every [`SLOT`].
#[derive(Debug, Clone, Default)]
pub struct Grid {
    first_end: DateTime<Utc>,
    values: Vec<f64>,
}

impl Grid {
    /// Slots ending in `(from, until]` (both rounded down to a boundary).
    pub fn build(
        history: &History,
        schedule: &GymSchedule,
        from: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Self {
        let first_end = floor_to_slot(from) + SLOT;
        let last_end = floor_to_slot(until);
        if last_end < first_end {
            return Self {
                first_end,
                values: Vec::new(),
            };
        }
        let slots = usize::try_from((last_end - first_end).num_minutes() / SLOT.num_minutes())
            .unwrap_or(0)
            + 1;
        let mut sums = vec![(0.0_f64, 0_u32); slots];
        for (t, value) in history.view().between(from, last_end) {
            let end = slot_end_of(t);
            if let Some(index) = index_in(first_end, slots, end) {
                sums[index].0 += value;
                sums[index].1 += 1;
            }
        }
        let values = sums
            .into_iter()
            .enumerate()
            .map(|(i, (sum, count))| {
                let end = first_end + SLOT * i32::try_from(i).unwrap_or(i32::MAX);
                if !is_open_slot(schedule, end) {
                    0.0
                } else if count == 0 {
                    f64::NAN
                } else {
                    sum / f64::from(count)
                }
            })
            .collect();
        Self { first_end, values }
    }

    /// End of the first slot.
    pub fn first_end(&self) -> DateTime<Utc> {
        self.first_end
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no slots.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// End of slot `index`.
    pub fn end_of(&self, index: usize) -> DateTime<Utc> {
        let steps = i32::try_from(index).unwrap_or(i32::MAX);
        self.first_end + SLOT * steps
    }

    /// Index of the slot ending at `end`, if it is a slot of this grid.
    pub fn index_of(&self, end: DateTime<Utc>) -> Option<usize> {
        index_in(self.first_end, self.values.len(), end)
    }

    /// Value of the slot ending at `end`; `None` outside the grid or when
    /// the slot has no reading.
    pub fn value_at(&self, end: DateTime<Utc>) -> Option<f64> {
        self.index_of(end)
            .and_then(|i| self.values.get(i))
            .copied()
            .filter(|v| !v.is_nan())
    }

    /// The slots a forecast made at `origin` may use: those ending at or
    /// before it.
    pub fn view_until(&self, origin: DateTime<Utc>) -> GridView<'_> {
        let origin = floor_to_slot(origin);
        let available = if origin < self.first_end {
            0
        } else {
            usize::try_from((origin - self.first_end).num_minutes() / SLOT.num_minutes())
                .map_or(0, |i| i + 1)
                .min(self.values.len())
        };
        GridView {
            first_end: self.first_end,
            values: &self.values[..available],
            origin,
        }
    }
}

/// The slots of a [`Grid`] ending at or before `origin`.
#[derive(Debug, Clone, Copy)]
pub struct GridView<'a> {
    first_end: DateTime<Utc>,
    values: &'a [f64],
    origin: DateTime<Utc>,
}

impl<'a> GridView<'a> {
    /// The moment the forecast is made: the end of the newest slot it may
    /// use (rounded down to a slot boundary).
    pub fn origin(&self) -> DateTime<Utc> {
        self.origin
    }

    /// Slot values, oldest first; NaN where a slot has no reading.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// End of the first slot.
    pub fn first_end(&self) -> DateTime<Utc> {
        self.first_end
    }

    /// Value of the slot ending at `end`, if known to this view.
    pub fn value_at(&self, end: DateTime<Utc>) -> Option<f64> {
        index_in(self.first_end, self.values.len(), end)
            .map(|i| self.values[i])
            .filter(|v| !v.is_nan())
    }

    /// The newest slot with a value, as (end, value).
    pub fn last_known(&self) -> Option<(DateTime<Utc>, f64)> {
        let (index, value) = self
            .values
            .iter()
            .enumerate()
            .rev()
            .find(|(_, v)| !v.is_nan())?;
        let steps = i32::try_from(index).ok()?;
        Some((self.first_end + SLOT * steps, *value))
    }

    /// End of the `step`-th slot after the origin (`step` ≥ 1).
    pub fn step_end(&self, step: usize) -> DateTime<Utc> {
        let steps = i32::try_from(step).unwrap_or(i32::MAX);
        self.origin + SLOT * steps
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::test_support::{local, minutely};

    #[test]
    fn test_slot_end_of_rounds_up_except_on_boundaries() {
        let t = local(0, 10, 0);
        assert_eq!(slot_end_of(t), t);
        assert_eq!(
            slot_end_of(t + TimeDelta::seconds(1)),
            t + TimeDelta::minutes(15)
        );
        assert_eq!(
            slot_end_of(t + TimeDelta::minutes(14)),
            t + TimeDelta::minutes(15)
        );
    }

    #[test]
    fn test_is_open_slot_judges_the_middle() {
        // Default schedule: weekdays 6–23.
        let schedule = GymSchedule::default();
        assert!(!is_open_slot(&schedule, local(0, 6, 0)));
        assert!(is_open_slot(&schedule, local(0, 6, 15)));
        assert!(is_open_slot(&schedule, local(0, 23, 0)));
        assert!(!is_open_slot(&schedule, local(0, 23, 15)));
    }

    #[test]
    fn test_grid_averages_readings_per_slot_and_zeroes_closed_slots() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = History::new(vec![
            (local(0, 10, 1), 10.0),
            (local(0, 10, 10), 20.0),
            (local(0, 10, 15), 30.0),
            (local(0, 10, 16), 50.0),
            // Closed at night: ignored.
            (local(0, 2, 0), 40.0),
        ]);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));

        assert_eq!(grid.len(), 96);
        assert_eq!(grid.first_end(), local(0, 0, 15));
        assert_relative_eq!(grid.value_at(local(0, 10, 15)).context("slot")?, 20.0);
        assert_relative_eq!(grid.value_at(local(0, 10, 30)).context("slot")?, 50.0);
        assert_eq!(grid.value_at(local(0, 2, 0)), Some(0.0));
        assert_eq!(
            grid.value_at(local(0, 11, 0)),
            None,
            "open without readings"
        );
        assert_eq!(grid.value_at(local(1, 0, 15)), None, "outside the grid");
        assert_eq!(grid.value_at(local(0, 10, 20)), None, "not a boundary");
        Ok(())
    }

    #[test]
    fn test_view_until_hides_later_slots() {
        let schedule = GymSchedule::default();
        let history = minutely(1, |_| 40.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));

        let view = grid.view_until(local(0, 10, 7));
        assert_eq!(view.origin(), local(0, 10, 0));
        assert_eq!(view.values().len(), 40);
        assert_eq!(view.value_at(local(0, 10, 0)), Some(40.0));
        assert_eq!(view.value_at(local(0, 10, 15)), None);
        assert_eq!(view.last_known(), Some((local(0, 10, 0), 40.0)));
        assert_eq!(view.step_end(1), local(0, 10, 15));
        assert_eq!(grid.view_until(local(0, 0, 0)).values().len(), 0);
    }

    #[test]
    fn test_last_known_skips_missing_slots() {
        let schedule = GymSchedule::default();
        let history = History::new(vec![(local(0, 10, 5), 25.0)]);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        let view = grid.view_until(local(0, 12, 0));
        assert_eq!(view.last_known(), Some((local(0, 10, 15), 25.0)));
    }

    #[test]
    fn test_same_local_time_before_follows_the_wall_clock_across_dst() -> Result<()> {
        let tz = GymSchedule::default().timezone();
        // 2024-03-31: clocks go forward in Germany.
        let after = tz
            .with_ymd_and_hms(2024, 4, 1, 10, 0, 0)
            .single()
            .context("time")?
            .with_timezone(&Utc);
        let before = same_local_time_before(after, 7, tz).context("week before")?;
        assert_eq!(
            before.with_timezone(&tz).format("%m-%d %H:%M").to_string(),
            "03-25 10:00"
        );
        assert_eq!(after - before, TimeDelta::days(7) - TimeDelta::hours(1));
        Ok(())
    }

    use chrono::TimeZone;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// A view never exposes a slot ending after its origin.
        #[test]
        fn test_view_never_reaches_past_its_origin(minutes in 0_i64..(3 * 24 * 60)) {
            let schedule = GymSchedule::default();
            let history = History::new(vec![(local(0, 12, 0), 30.0), (local(2, 12, 0), 60.0)]);
            let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
            let origin = local(0, 0, 0) + TimeDelta::minutes(minutes);
            let view = grid.view_until(origin);
            prop_assert!(view.origin() <= origin);
            prop_assert!(origin - view.origin() < SLOT);
            let newest = view.first_end() + SLOT * i32::try_from(view.values().len()).unwrap_or(0) - SLOT;
            prop_assert!(view.values().is_empty() || newest == view.origin());
        }
    }
}
