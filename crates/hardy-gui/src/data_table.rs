//! The Data tab's logic: which period is shown, which rows of the table are
//! in view, and what the readings are made of.

use std::ops::Range;

use chrono::{DateTime, Datelike, TimeDelta, Utc};
use hardy_core::{
    Tz,
    analytics::midnight_local_as_utc,
    db::{DataSource, OccupancyLog},
};

use crate::time_range::gym_week_start;

/// Height of one table row, in logical pixels.
pub const ROW_HEIGHT: f32 = 28.0;
/// Rows built beyond the visible ones on each side, so fast scrolling does
/// not show gaps before the next redraw.
const MARGIN_ROWS: usize = 20;
/// Rows built before the viewport size is known.
const FIRST_ROWS: usize = 60;

/// How much the Data tab shows at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataPeriod {
    #[default]
    Day,
    /// Monday to Sunday, gym-local.
    Week,
}

impl DataPeriod {
    /// `[start, end)` of the period `offset` periods before the current one.
    pub fn window(self, now: DateTime<Utc>, tz: Tz, offset: u32) -> (DateTime<Utc>, DateTime<Utc>) {
        let (first, days) = match self {
            DataPeriod::Day => (now.with_timezone(&tz).date_naive(), 1),
            DataPeriod::Week => (gym_week_start(now, tz).with_timezone(&tz).date_naive(), 7),
        };
        // Whole local days, so a day with a clock change is 23 or 25 hours.
        let first = first - TimeDelta::days(days * i64::from(offset));
        (
            midnight_local_as_utc(first, tz),
            midnight_local_as_utc(first + TimeDelta::days(days), tz),
        )
    }

    /// The period in words: "Fri, 16 Oct" or "12 – 18 Oct".
    pub fn label(self, window: (DateTime<Utc>, DateTime<Utc>), tz: Tz) -> String {
        let first = window.0.with_timezone(&tz).date_naive();
        let last = (window.1 - TimeDelta::seconds(1))
            .with_timezone(&tz)
            .date_naive();
        match self {
            DataPeriod::Day => first.format("%a, %-d %b").to_string(),
            DataPeriod::Week if first.month() == last.month() => {
                format!("{} – {}", first.format("%-d"), last.format("%-d %b"))
            }
            DataPeriod::Week => format!("{} – {}", first.format("%-d %b"), last.format("%-d %b")),
        }
    }
}

/// The rows of `total` to build for a viewport `height` high scrolled
/// `offset` down; before the viewport is known (`height` 0), the first few.
pub fn visible_rows(total: usize, offset: f32, height: f32) -> Range<usize> {
    if height <= 0.0 {
        return 0..total.min(FIRST_ROWS);
    }
    // Row indices from pixel offsets; negative overscroll counts as 0.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let row_at = |y: f32| (y.max(0.0) / ROW_HEIGHT) as usize;
    let first = row_at(offset).saturating_sub(MARGIN_ROWS);
    let last = row_at(offset + height) + MARGIN_ROWS + 1;
    first.min(total)..last.min(total)
}

/// What the readings of a period are made of.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceCounts {
    pub measured: usize,
    /// Filled into gaps or added at opening and closing by Data Repair.
    pub filled: usize,
    /// Measured values Data Repair changed (outliers, smoothing).
    pub changed: usize,
}

impl SourceCounts {
    /// Counts of `rows` by source.
    pub fn of(rows: &[OccupancyLog]) -> Self {
        rows.iter().fold(Self::default(), |mut counts, row| {
            match row.source {
                DataSource::Measured => counts.measured += 1,
                DataSource::Interpolated | DataSource::Boundary => counts.filled += 1,
                DataSource::Smoothed => counts.changed += 1,
            }
            counts
        })
    }
}

/// Display name of a source.
pub fn source_label(source: DataSource) -> &'static str {
    match source {
        DataSource::Measured => "Measured",
        DataSource::Interpolated => "Filled gap",
        DataSource::Boundary => "Opening/closing",
        DataSource::Smoothed => "Smoothed",
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    /// The default gym's zone: Europe/Berlin.
    fn tz() -> Tz {
        hardy_core::GymSchedule::default().timezone()
    }

    /// 2026-10-`day` at `h:mi` in Berlin.
    fn berlin(day: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Ok(tz()
            .with_ymd_and_hms(2026, 10, day, h, mi, 0)
            .single()
            .context("unambiguous local time")?
            .with_timezone(&Utc))
    }

    fn log(source: DataSource) -> OccupancyLog {
        OccupancyLog {
            id: 0,
            timestamp: DateTime::default(),
            percentage: 20.0,
            source,
        }
    }

    #[test]
    fn test_day_window_is_the_local_day_and_steps_back_a_day() -> Result<()> {
        // Friday 16 October.
        let now = berlin(16, 9, 30)?;
        assert_eq!(
            DataPeriod::Day.window(now, tz(), 0),
            (berlin(16, 0, 0)?, berlin(17, 0, 0)?)
        );
        assert_eq!(
            DataPeriod::Day.window(now, tz(), 2),
            (berlin(14, 0, 0)?, berlin(15, 0, 0)?)
        );
        Ok(())
    }

    #[test]
    fn test_day_window_across_the_clock_change_is_25_hours() -> Result<()> {
        // Clocks go back on Sunday 25 October 2026.
        let window = DataPeriod::Day.window(berlin(26, 12, 0)?, tz(), 1);
        assert_eq!(window, (berlin(25, 0, 0)?, berlin(26, 0, 0)?));
        assert_eq!(window.1 - window.0, TimeDelta::hours(25));
        Ok(())
    }

    #[test]
    fn test_week_window_runs_monday_to_monday() -> Result<()> {
        let now = berlin(16, 9, 30)?;
        assert_eq!(
            DataPeriod::Week.window(now, tz(), 0),
            (berlin(12, 0, 0)?, berlin(19, 0, 0)?)
        );
        assert_eq!(
            DataPeriod::Week.window(now, tz(), 1),
            (berlin(5, 0, 0)?, berlin(12, 0, 0)?)
        );
        Ok(())
    }

    #[test]
    fn test_labels_name_the_day_or_the_week() -> Result<()> {
        let now = berlin(16, 9, 30)?;
        let day = DataPeriod::Day.window(now, tz(), 0);
        assert_eq!(DataPeriod::Day.label(day, tz()), "Fri, 16 Oct");
        let week = DataPeriod::Week.window(now, tz(), 0);
        assert_eq!(DataPeriod::Week.label(week, tz()), "12 – 18 Oct");
        // A week across two months names both.
        let earlier = DataPeriod::Week.window(berlin(1, 9, 0)?, tz(), 0);
        assert_eq!(DataPeriod::Week.label(earlier, tz()), "28 Sep – 4 Oct");
        Ok(())
    }

    #[test]
    fn test_visible_rows_before_the_viewport_is_known() {
        assert_eq!(visible_rows(1000, 0.0, 0.0), 0..FIRST_ROWS);
        assert_eq!(visible_rows(10, 0.0, 0.0), 0..10);
        assert_eq!(visible_rows(0, 0.0, 0.0), 0..0);
    }

    #[test]
    fn test_visible_rows_follow_the_scroll_offset() {
        // 10 rows fit; scrolled down by 100 rows.
        let rows = visible_rows(1000, 100.0 * ROW_HEIGHT, 10.0 * ROW_HEIGHT);
        assert_eq!(rows, 100 - MARGIN_ROWS..110 + MARGIN_ROWS + 1);
        // At the end the range stops at the last row.
        let end = visible_rows(1000, 995.0 * ROW_HEIGHT, 10.0 * ROW_HEIGHT);
        assert_eq!(end.end, 1000);
    }

    #[test]
    fn test_source_counts_and_labels() {
        let rows = [
            log(DataSource::Measured),
            log(DataSource::Measured),
            log(DataSource::Interpolated),
            log(DataSource::Boundary),
            log(DataSource::Smoothed),
        ];
        assert_eq!(
            SourceCounts::of(&rows),
            SourceCounts {
                measured: 2,
                filled: 2,
                changed: 1
            }
        );
        assert_eq!(source_label(DataSource::Measured), "Measured");
        assert_eq!(source_label(DataSource::Interpolated), "Filled gap");
        assert_eq!(source_label(DataSource::Boundary), "Opening/closing");
        assert_eq!(source_label(DataSource::Smoothed), "Smoothed");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Every row inside the viewport is built, the range never leaves
        /// the table, and it stays small however long the table is.
        #[test]
        fn test_visible_rows_cover_the_viewport(
            total in 0_usize..20_000,
            first in 0_usize..20_000,
            shown in 1_usize..80,
        ) {
            #[allow(clippy::cast_precision_loss)]
            let (offset, height) = (first as f32 * ROW_HEIGHT, shown as f32 * ROW_HEIGHT);
            let rows = visible_rows(total, offset, height);
            prop_assert!(rows.start <= rows.end && rows.end <= total);
            for row in first.min(total)..(first + shown).min(total) {
                prop_assert!(rows.contains(&row), "row {} not in {:?}", row, rows);
            }
            prop_assert!(rows.len() <= shown + 2 * MARGIN_ROWS + 2);
        }
    }
}
