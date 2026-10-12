//! Profile-based models: forecasts built on the typical day or week, each
//! correcting it with what today looks like in a different way.

mod offset;
mod pca;
mod residual_ar;
mod similar_days;

use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Timelike, Utc};
use hardy_core::{Tz, schedule::is_bavarian_holiday};
pub use offset::ProfileCurrentOffset;
pub use pca::DailyPca;
pub use residual_ar::ProfileResidualAr;
use serde::{Deserialize, Serialize};
pub use similar_days::SimilarDays;

use super::Observed;
use crate::grid::{GridView, SLOT, SLOTS_PER_DAY, SLOTS_PER_WEEK, is_open_slot};

/// The gym-local moment a slot covers, for calendar lookups: one minute
/// before its end, so the slot ending at midnight belongs to the day before.
fn slot_time(end: DateTime<Utc>, tz: Tz) -> chrono::DateTime<Tz> {
    (end - TimeDelta::minutes(1)).with_timezone(&tz)
}

/// Index of the slot ending at `end` within its gym-local day (0–95).
pub(in crate::suite) fn slot_of_day(end: DateTime<Utc>, tz: Tz) -> usize {
    let local = slot_time(end, tz);
    local.hour() as usize * 4 + local.minute() as usize / 15
}

/// Index of the slot ending at `end` within its gym-local week (0–671,
/// Monday first).
pub(in crate::suite) fn slot_of_week(end: DateTime<Utc>, tz: Tz) -> usize {
    let day = slot_time(end, tz).weekday().num_days_from_monday() as usize;
    day * SLOTS_PER_DAY + slot_of_day(end, tz)
}

/// The gym-local date the slot ending at `end` belongs to.
pub(in crate::suite) fn slot_date(end: DateTime<Utc>, tz: Tz) -> NaiveDate {
    slot_time(end, tz).date_naive()
}

/// Weekend and public holidays follow the weekend opening hours.
pub(in crate::suite) fn is_weekend_like(date: NaiveDate) -> bool {
    date.weekday().number_from_monday() > 5 || is_bavarian_holiday(date)
}

/// `(end, value)` of every open slot with a reading in `grid` ending at or
/// after `from`.
pub(in crate::suite) fn open_values<'a>(
    data: &'a Observed<'_>,
    grid: GridView<'a>,
    from: DateTime<Utc>,
) -> impl Iterator<Item = (DateTime<Utc>, f64)> + 'a {
    let first = grid.first_end();
    grid.values()
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(move |(i, v)| (first + SLOT * i32::try_from(i).unwrap_or(i32::MAX), *v))
        .filter(move |(end, _)| *end >= from && is_open_slot(data.schedule(), *end))
}

/// The typical value of each slot of the week: the mean of the open slots
/// observed there. Missing slots fall back to the same time on other days
/// of the same kind (weekday or weekend), then to the overall mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::suite) struct WeeklyProfile {
    tz: Tz,
    week: Vec<f64>,
}

impl WeeklyProfile {
    /// The profile of the open slots in `data` from its first reading on.
    pub(in crate::suite) fn fit(data: &Observed<'_>) -> Self {
        let tz = data.schedule().timezone();
        let from = data.history().first_time().unwrap_or(data.origin());
        let mut sums = vec![(0.0_f64, 0_u32); SLOTS_PER_WEEK];
        for (end, value) in open_values(data, data.grid(), from) {
            let slot = &mut sums[slot_of_week(end, tz)];
            slot.0 += value;
            slot.1 += 1;
        }
        let mean = |(sum, n): (f64, u32)| (n > 0).then(|| sum / f64::from(n));
        let overall = mean(
            sums.iter()
                .fold((0.0, 0), |(s, n), (sum, count)| (s + sum, n + count)),
        )
        .unwrap_or(0.0);
        // Same slot of the day on weekdays (Mon–Fri) and on weekends.
        let kind_mean = |weekend: bool, day_slot: usize| {
            let days = if weekend { 5..7 } else { 0..5 };
            mean(days.fold((0.0, 0), |(s, n), d| {
                let (sum, count) = sums[d * SLOTS_PER_DAY + day_slot];
                (s + sum, n + count)
            }))
        };
        let week = (0..SLOTS_PER_WEEK)
            .map(|i| {
                mean(sums[i])
                    .or_else(|| kind_mean(i >= 5 * SLOTS_PER_DAY, i % SLOTS_PER_DAY))
                    .unwrap_or(overall)
            })
            .collect();
        Self { tz, week }
    }

    /// Typical value of the slot ending at `end`.
    pub(in crate::suite) fn at(&self, end: DateTime<Utc>) -> f64 {
        self.week[slot_of_week(end, self.tz)]
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::Grid,
        test_support::{local, minutely},
    };

    #[test]
    fn test_slot_indices_follow_the_local_calendar() {
        let tz = GymSchedule::default().timezone();
        // Monday 00:15 is the first slot of the week; Monday 00:00 ends the
        // last slot of Sunday.
        assert_eq!(slot_of_week(local(0, 0, 15), tz), 0);
        assert_eq!(slot_of_week(local(0, 0, 0), tz), SLOTS_PER_WEEK - 1);
        assert_eq!(slot_of_day(local(0, 10, 15), tz), 40);
        assert_eq!(slot_of_week(local(1, 10, 15), tz), SLOTS_PER_DAY + 40);
        assert_eq!(
            slot_date(local(1, 0, 0), tz),
            slot_date(local(0, 23, 45), tz)
        );
    }

    #[test]
    fn test_weekend_like_days_include_holidays() {
        let date = |m, d| NaiveDate::from_ymd_opt(2024, m, d).unwrap_or_default();
        assert!(is_weekend_like(date(6, 8)), "Saturday");
        assert!(!is_weekend_like(date(6, 7)), "Friday");
        assert!(is_weekend_like(date(10, 3)), "German Unity Day");
    }

    #[test]
    fn test_weekly_profile_means_and_falls_back() {
        let schedule = GymSchedule::default();
        // Two Mondays at 20 and 40; the rest of the week at 10.
        let history = minutely(9, |t| {
            let day = (t - local(0, 0, 0)).num_days();
            match day {
                0 => 20.0,
                7 => 40.0,
                _ => 10.0,
            }
        });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(9, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(9, 0, 0));
        let profile = WeeklyProfile::fit(&data);

        assert_relative_eq!(profile.at(local(14, 10, 15)), 30.0);
        assert_relative_eq!(profile.at(local(15, 10, 15)), 10.0);
    }

    #[test]
    fn test_weekly_profile_fills_unseen_weekdays_from_their_kind() {
        let schedule = GymSchedule::default();
        // Only Monday and Tuesday seen: Wednesday borrows their mean.
        let history = minutely(2, |t| if t < local(1, 0, 0) { 20.0 } else { 30.0 });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(2, 0, 0));
        let profile = WeeklyProfile::fit(&data);

        assert_relative_eq!(profile.at(local(2, 10, 15)), 25.0);
        // Saturday has no weekend data: the overall mean.
        assert_relative_eq!(profile.at(local(5, 10, 15)), 25.0, epsilon = 1e-9);
    }
}
