//! The slot series as the statistical models walk it.

use chrono::{DateTime, Utc};

use crate::{
    grid::{SLOT, is_open_slot},
    suite::{Observed, profiles::WeeklyProfile},
};

/// What a model sees of one slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Slot {
    /// The gym is closed: nothing to learn, nothing to forecast.
    Closed,
    /// Open, but without a reading.
    Missing,
    /// Open with this value.
    Value(f64),
}

impl Slot {
    /// The same slot with `f` applied to its value.
    pub(super) fn map(self, f: impl FnOnce(f64) -> f64) -> Self {
        match self {
            Slot::Value(v) => Slot::Value(f(v)),
            other => other,
        }
    }
}

/// The slots of `data` that end after `after` (up to its origin), oldest
/// first, with their end.
pub(super) fn slots_after<'a>(
    data: &'a Observed<'_>,
    after: DateTime<Utc>,
) -> impl Iterator<Item = (DateTime<Utc>, Slot)> + 'a {
    let grid = data.grid();
    let first = grid.first_end();
    // Index of the first slot ending after `after`.
    let skip = if after < first {
        0
    } else {
        usize::try_from((after - first).num_minutes() / SLOT.num_minutes() + 1).unwrap_or(0)
    };
    grid.values()
        .iter()
        .enumerate()
        .skip(skip)
        .map(move |(i, &value)| {
            let end = first + SLOT * i32::try_from(i).unwrap_or(i32::MAX);
            let slot = if !is_open_slot(data.schedule(), end) {
                Slot::Closed
            } else if value.is_nan() {
                Slot::Missing
            } else {
                Slot::Value(value)
            };
            (end, slot)
        })
}

/// The same slots as deviations from `profile`.
pub(super) fn deviations_after<'a>(
    data: &'a Observed<'_>,
    profile: &'a WeeklyProfile,
    after: DateTime<Utc>,
) -> impl Iterator<Item = (DateTime<Utc>, Slot)> + 'a {
    slots_after(data, after).map(|(end, slot)| (end, slot.map(|v| v - profile.at(end))))
}

/// The open slots of the training data in `data` as deviations from
/// `profile`, oldest first (`None` where a slot has no reading).
pub(super) fn open_deviations(data: &Observed<'_>, profile: &WeeklyProfile) -> Vec<Option<f64>> {
    let from = data.history().first_time().unwrap_or(data.origin()) - SLOT;
    deviations_after(data, profile, from)
        .filter_map(|(_, slot)| match slot {
            Slot::Closed => None,
            Slot::Missing => Some(None),
            Slot::Value(v) => Some(Some(v)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::Grid,
        history::History,
        test_support::{local, minutely},
    };

    #[test]
    fn test_slots_after_mark_closed_missing_and_values() {
        let schedule = GymSchedule::default();
        // Readings on day 0 until 10:00, then none until 12:00.
        let history = minutely(1, |_| 30.0);
        let history = History::new(
            history
                .view()
                .iter()
                .filter(|(t, _)| *t < local(0, 10, 0) || *t >= local(0, 12, 0))
                .collect(),
        );
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(0, 13, 0));

        let slots: Vec<_> = slots_after(&data, local(0, 5, 30)).collect();
        // 05:45 … 13:00: 30 slots.
        assert_eq!(slots.len(), 30);
        assert_eq!(slots[0], (local(0, 5, 45), Slot::Closed));
        assert_eq!(
            slots[1],
            (local(0, 6, 0), Slot::Closed),
            "06:00 ends a closed slot"
        );
        assert_eq!(slots[2], (local(0, 6, 15), Slot::Value(30.0)));
        let at = |h, m| {
            slots
                .iter()
                .find(|(end, _)| *end == local(0, h, m))
                .map(|(_, s)| *s)
        };
        // Gaps over an hour stay missing.
        assert_eq!(at(11, 0), Some(Slot::Missing));
        assert_eq!(slots.last(), Some(&(local(0, 13, 0), Slot::Value(30.0))));
    }

    #[test]
    fn test_slots_after_the_origin_are_none() {
        let schedule = GymSchedule::default();
        let history = minutely(1, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(0, 13, 0));
        assert_eq!(slots_after(&data, local(0, 13, 0)).count(), 0);
        assert_eq!(slots_after(&data, local(0, 14, 0)).count(), 0);
        // Before the grid starts: from its first slot.
        let all = slots_after(&data, local(-3, 0, 0)).count();
        assert_eq!(all, 13 * 4);
    }

    #[test]
    fn test_slot_map_changes_values_only() {
        assert_eq!(Slot::Value(3.0).map(|v| v * 2.0), Slot::Value(6.0));
        assert_eq!(Slot::Missing.map(|v| v * 2.0), Slot::Missing);
        assert_eq!(Slot::Closed.map(|v| v * 2.0), Slot::Closed);
    }
}
