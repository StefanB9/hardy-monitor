//! Axis helpers and a line-chart element shared by the Forecasts pages.

use chrono::{NaiveDate, TimeDelta};
use iced::{
    Element, Length,
    widget::canvas::{self, Cache},
};

use crate::{
    app::Message,
    widgets::plot::{LineChart, Series},
};

/// Most labels under a day axis.
const MAX_DAY_TICKS: i64 = 6;
/// Height of a chart.
pub(super) const HEIGHT: f32 = 200.0;

/// Where `day` sits on an axis starting at `first`.
#[allow(clippy::cast_precision_loss)]
pub(super) fn day_x(first: NaiveDate, day: NaiveDate) -> f64 {
    (day - first).num_days() as f64
}

/// Where the gym-local time `at` sits on a day axis starting at `first`,
/// with the time of day as a fraction (two runs on one date stay apart).
#[allow(clippy::cast_precision_loss)]
pub(super) fn time_x(first: NaiveDate, at: chrono::NaiveDateTime) -> f64 {
    (at - first.and_time(chrono::NaiveTime::MIN)).num_minutes() as f64 / 1440.0
}

/// The time at position `x` on a day axis starting at `first`.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn x_time(first: NaiveDate, x: f64) -> chrono::NaiveDateTime {
    first.and_time(chrono::NaiveTime::MIN) + TimeDelta::minutes((x * 1440.0).round() as i64)
}

/// The day at position `x` on an axis starting at `first`.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn x_day(first: NaiveDate, x: f64) -> NaiveDate {
    first + TimeDelta::days(x.round() as i64)
}

/// Labels for the days `first..=last`: at most six, evenly spaced.
pub(super) fn day_ticks(first: NaiveDate, last: NaiveDate) -> Vec<(f64, String)> {
    let span = (last - first).num_days().max(0);
    let step = (span / (MAX_DAY_TICKS - 1)).max(1);
    let mut ticks = Vec::with_capacity(usize::try_from(MAX_DAY_TICKS).unwrap_or(6));
    let mut offset = 0;
    while offset <= span {
        let day = first + TimeDelta::days(offset);
        ticks.push((day_x(first, day), day.format("%-d %b").to_string()));
        offset += step;
    }
    ticks
}

/// Labels for lead times in quarter-hour steps: every full hour up to 6 h.
pub(super) fn lead_ticks() -> Vec<(f64, String)> {
    (1..=6)
        .map(|h| (f64::from(h * 4), format!("{h} h")))
        .collect()
}

/// A lead time of `step` quarter hours, e.g. "1 h 15 min ahead".
#[allow(clippy::cast_possible_truncation)]
pub(super) fn lead_label(step: f64) -> String {
    let minutes = (step.round() as i64) * 15;
    match (minutes / 60, minutes % 60) {
        (0, m) => format!("{m} min ahead"),
        (h, 0) => format!("{h} h ahead"),
        (h, m) => format!("{h} h {m} min ahead"),
    }
}

/// A line chart filling the width.
pub(super) fn line_chart<'a>(
    series: Vec<Series>,
    x_ticks: Vec<(f64, String)>,
    x_label: impl Fn(f64) -> String + 'a,
    suffix: &'static str,
    target: Option<f64>,
    cache: &'a Cache,
) -> Element<'a, Message> {
    canvas::Canvas::new(LineChart {
        series,
        x_ticks,
        x_label,
        suffix,
        target,
        cache,
    })
    .width(Length::Fill)
    .height(HEIGHT)
    .into()
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};

    use super::*;

    fn day(d: u32) -> Result<NaiveDate> {
        NaiveDate::from_ymd_opt(2024, 6, d).context("date")
    }

    #[test]
    fn test_day_ticks_are_few_and_evenly_spaced() -> Result<()> {
        let ticks = day_ticks(day(1)?, day(28)?);
        assert_eq!(ticks.len(), 6);
        assert_eq!(ticks[0], (0.0, "1 Jun".to_string()));
        assert_eq!(ticks[1].0, 5.0);
        let short = day_ticks(day(3)?, day(5)?);
        assert_eq!(short.len(), 3);
        assert_eq!(day_ticks(day(3)?, day(3)?).len(), 1);
        Ok(())
    }

    #[test]
    fn test_day_positions_round_trip() -> Result<()> {
        assert_eq!(x_day(day(1)?, day_x(day(1)?, day(17)?)), day(17)?);
        let evening = day(3)?.and_hms_opt(18, 0, 0).context("time")?;
        assert!((time_x(day(1)?, evening) - 2.75).abs() < 1e-9);
        assert_eq!(x_time(day(1)?, 2.75), evening);
        Ok(())
    }

    #[test]
    fn test_lead_labels() {
        assert_eq!(lead_label(1.0), "15 min ahead");
        assert_eq!(lead_label(4.0), "1 h ahead");
        assert_eq!(lead_label(5.0), "1 h 15 min ahead");
        assert_eq!(lead_ticks().last(), Some(&(24.0, "6 h".to_string())));
    }
}
