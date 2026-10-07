//! Averages per gym-local weekday and hour, with each weekday's opening hours.

use std::ops::Range;

use chrono::NaiveDate;
use hardy_core::{db::HourlyAverage, schedule::GymSchedule};

/// One grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    Closed,
    NoData,
    /// Average occupancy and the number of readings behind it.
    Value(f64, i64),
}

/// Averages per gym-local weekday (0 = Monday) and hour, with the regular
/// opening hours of each weekday.
#[derive(Debug, Clone, PartialEq)]
pub struct WeekGrid {
    values: [[Option<(f64, i64)>; 24]; 7],
    open: [Range<u32>; 7],
}

impl Default for WeekGrid {
    fn default() -> Self {
        Self::new(&[], &GymSchedule::default())
    }
}

impl WeekGrid {
    pub fn new(data: &[HourlyAverage], schedule: &GymSchedule) -> Self {
        let mut values = [[None; 24]; 7];
        for avg in data {
            if let (Ok(day), Ok(hour)) = (usize::try_from(avg.weekday), usize::try_from(avg.hour))
                && day < 7
                && hour < 24
                && avg.sample_count > 0
            {
                values[day][hour] = Some((avg.avg_percentage, avg.sample_count));
            }
        }
        // SAFETY: 2024-07-01 is a Monday and that week has no Bavarian
        // holiday, so each date shows the regular hours of its weekday.
        let open = std::array::from_fn(|day| {
            NaiveDate::from_ymd_opt(2024, 7, 1)
                .and_then(|monday| monday.checked_add_days(chrono::Days::new(day as u64)))
                .map_or(0..0, |date| {
                    schedule.get_open_hour(date)..schedule.get_close_hour(date)
                })
        });
        Self { values, open }
    }

    /// Hours shown: from the earliest opening to the latest closing.
    pub fn hour_span(&self) -> Range<u32> {
        let start = self.open.iter().map(|r| r.start).min().unwrap_or(0);
        let end = self.open.iter().map(|r| r.end).max().unwrap_or(24).min(24);
        if start < end { start..end } else { 0..24 }
    }

    /// Quietest and busiest open hour of `day` with data, skipping the first
    /// `skip_hours` after opening (always empty, so never informative).
    pub fn extremes(&self, day: usize, skip_hours: u32) -> Option<((u32, f64), (u32, f64))> {
        let open = self.open.get(day)?;
        let hours = (open.start + skip_hours)..open.end;
        let values = hours.filter_map(|h| match self.cell(day, h) {
            Cell::Value(v, _) => Some((h, v)),
            Cell::Closed | Cell::NoData => None,
        });
        let mut extremes: Option<((u32, f64), (u32, f64))> = None;
        for (h, v) in values {
            extremes = Some(match extremes {
                None => ((h, v), (h, v)),
                Some((low, high)) => (
                    if v < low.1 { (h, v) } else { low },
                    if v > high.1 { (h, v) } else { high },
                ),
            });
        }
        extremes
    }

    pub fn cell(&self, day: usize, hour: u32) -> Cell {
        let Some(open) = self.open.get(day) else {
            return Cell::Closed;
        };
        if !open.contains(&hour) {
            return Cell::Closed;
        }
        match self.values[day].get(hour as usize).copied().flatten() {
            Some((value, samples)) => Cell::Value(value, samples),
            None => Cell::NoData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avg(weekday: i32, hour: i32, value: f64) -> HourlyAverage {
        HourlyAverage {
            weekday,
            hour,
            avg_percentage: value,
            sample_count: 60,
        }
    }

    #[test]
    fn test_week_grid_follows_schedule() {
        // Default schedule: weekdays 06–23, weekends 09–21.
        let schedule = GymSchedule::default();
        let grid = WeekGrid::new(&[avg(0, 10, 25.0), avg(5, 7, 40.0)], &schedule);
        assert_eq!(grid.cell(0, 10), Cell::Value(25.0, 60));
        assert_eq!(grid.cell(0, 11), Cell::NoData);
        // Saturday 07:00 is before weekend opening: closed despite a value.
        assert_eq!(grid.cell(5, 7), Cell::Closed);
        assert_eq!(grid.cell(0, 3), Cell::Closed);
    }

    #[test]
    fn test_week_grid_extremes_skip_opening_hour() {
        let grid = WeekGrid::new(
            &[
                avg(0, 6, 2.0),
                avg(0, 10, 30.0),
                avg(0, 14, 12.0),
                avg(0, 18, 55.0),
            ],
            &GymSchedule::default(),
        );
        assert_eq!(grid.extremes(0, 1), Some(((14, 12.0), (18, 55.0))));
        assert_eq!(grid.extremes(0, 0).map(|(low, _)| low), Some((6, 2.0)));
        assert_eq!(grid.extremes(1, 1), None);
        assert_eq!(grid.extremes(7, 1), None);
    }

    #[test]
    fn test_week_grid_hour_span_covers_all_opening_hours() {
        let grid = WeekGrid::new(&[], &GymSchedule::default());
        let span = grid.hour_span();
        for day in 0..7 {
            for hour in 0..24 {
                if grid.cell(day, hour) != Cell::Closed {
                    assert!(span.contains(&hour), "day {day} hour {hour}");
                }
            }
        }
        assert!(span.len() < 24);
    }

    #[test]
    fn test_week_grid_ignores_out_of_range_and_empty_rows() {
        let mut empty = avg(1, 12, 30.0);
        empty.sample_count = 0;
        let grid = WeekGrid::new(
            &[avg(7, 12, 1.0), avg(-1, 12, 1.0), avg(1, 24, 1.0), empty],
            &GymSchedule::default(),
        );
        assert_eq!(grid.cell(1, 12), Cell::NoData);
        assert_eq!(grid.cell(9, 12), Cell::Closed);
    }
}
