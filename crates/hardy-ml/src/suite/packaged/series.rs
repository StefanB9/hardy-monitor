//! The series handed to the packaged models: the slot grid over the last
//! [`LOOKBACK`] before the origin.

use chrono::{DateTime, TimeDelta, Utc};

use crate::{
    grid::SLOT,
    suite::{
        Observed,
        profiles::WeeklyProfile,
        statistical::series::{Slot, slots_after},
    },
};

/// How far back before the origin the models fit.
pub(super) const LOOKBACK: TimeDelta = TimeDelta::days(56);

/// Where the lookback window starts: `LOOKBACK` before the origin, or the
/// grid's start if that is later.
fn window_start(data: &Observed<'_>) -> DateTime<Utc> {
    (data.origin() - LOOKBACK).max(data.grid().first_end() - SLOT)
}

/// Every slot of the window (end, value): closed slots 0, missing open
/// slots the profile's typical value.
pub(super) fn grid_series(
    data: &Observed<'_>,
    profile: &WeeklyProfile,
) -> (Vec<DateTime<Utc>>, Vec<f64>) {
    slots_after(data, window_start(data))
        .map(|(end, slot)| {
            let value = match slot {
                Slot::Closed => 0.0,
                Slot::Missing => profile.at(end),
                Slot::Value(v) => v,
            };
            (end, value)
        })
        .unzip()
}

/// The open slots of the window (end, value), `None` where a slot has no
/// reading.
pub(super) fn open_slots(data: &Observed<'_>) -> Vec<(DateTime<Utc>, Option<f64>)> {
    slots_after(data, window_start(data))
        .filter_map(|(end, slot)| match slot {
            Slot::Closed => None,
            Slot::Missing => Some((end, None)),
            Slot::Value(v) => Some((end, Some(v))),
        })
        .collect()
}

/// The open slots of the window as deviations from `profile` (0 where a
/// slot has no reading).
pub(super) fn deviation_series(data: &Observed<'_>, profile: &WeeklyProfile) -> Vec<f64> {
    open_slots(data)
        .into_iter()
        .map(|(end, value)| value.map_or(0.0, |v| v - profile.at(end)))
        .collect()
}

#[cfg(test)]
mod tests {
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::{Grid, SLOTS_PER_DAY, is_open_slot},
        history::History,
        test_support::{local, minutely},
    };

    /// 60 days of readings at 30, except none on day 59 from 10:00 to
    /// 12:00.
    fn history() -> History {
        let full = minutely(60, |_| 30.0);
        History::new(
            full.view()
                .iter()
                .filter(|(t, _)| *t < local(59, 10, 0) || *t >= local(59, 12, 0))
                .collect(),
        )
    }

    #[test]
    fn test_grid_series_covers_the_window_with_nights_at_zero() {
        let schedule = GymSchedule::default();
        let history = history();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(60, 0, 0));
        let origin = local(59, 13, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);

        let (ends, values) = grid_series(&data, &profile);
        assert_eq!(values.len(), 56 * SLOTS_PER_DAY);
        assert_eq!(ends.len(), values.len());
        assert_eq!(ends.last(), Some(&origin));
        assert_eq!(ends[0], origin - LOOKBACK + SLOT);
        for (end, value) in ends.iter().zip(&values) {
            if is_open_slot(&schedule, *end) {
                // Gaps are filled with the profile, which is 30 too.
                assert!((value - 30.0).abs() < 1e-9, "{end}: {value}");
            } else {
                assert_eq!(*value, 0.0, "{end}");
            }
        }
    }

    #[test]
    fn test_grid_series_starts_at_the_grid_when_history_is_short() {
        let schedule = GymSchedule::default();
        let history = minutely(3, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let origin = local(2, 12, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let (ends, _) = grid_series(&data, &profile);
        assert_eq!(ends.first(), Some(&grid.first_end()));
    }

    #[test]
    fn test_deviation_series_holds_open_slots_only() {
        let schedule = GymSchedule::default();
        // 30 every day; on the last day 40 from 12:00.
        let history = minutely(3, |t| if t >= local(2, 12, 0) { 40.0 } else { 30.0 });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let origin = local(2, 13, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let deviations = deviation_series(&data, &profile);
        // Days 0 and 1 open 06:00–23:00 (68 slots each), day 2 until 13:00.
        assert_eq!(deviations.len(), 68 * 2 + 28);
        let last = deviations.last().copied().unwrap_or_default();
        assert!((last - (40.0 - profile.at(origin))).abs() < 1e-9);
        let open = open_slots(&data);
        assert_eq!(open.len(), deviations.len());
        assert_eq!(open.last(), Some(&(origin, Some(40.0))));
    }
}
