//! The past forecast layer: what was forecast for each slot an hour before,
//! as a thin line with its likely range, drawn over the readings.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::db::ForecastHistoryEntry;
use iced::widget::canvas::{self, Frame, Path, Stroke};

use super::{HistoryChart, plot::Plot};
use crate::style;

/// Slots further apart than this (a closed night, a gap) are drawn as
/// separate runs.
const MAX_SLOT_GAP: TimeDelta = TimeDelta::minutes(15);

impl HistoryChart<'_> {
    /// The past forecasts inside the range.
    pub(super) fn past_in_range(&self) -> &[ForecastHistoryEntry] {
        in_range(self.past_forecasts, self.range_start, self.range_end)
    }

    pub(super) fn draw_past(&self, frame: &mut Frame, plot: &Plot) {
        for run in runs(self.past_in_range()) {
            for banded in run.chunk_by(|a, b| a.low.is_some() == b.low.is_some()) {
                let outline: Vec<(DateTime<Utc>, f64)> = banded
                    .iter()
                    .filter_map(|e| Some((e.target, e.high?)))
                    .chain(banded.iter().rev().filter_map(|e| Some((e.target, e.low?))))
                    .collect();
                if outline.len() < 4 {
                    continue;
                }
                let band = Path::new(|b| {
                    let mut vertices = outline.iter().map(|&(t, v)| plot.point(t, v));
                    if let Some(start) = vertices.next() {
                        b.move_to(start);
                    }
                    for vertex in vertices {
                        b.line_to(vertex);
                    }
                    b.close();
                });
                frame.fill(&band, style::tint(style::FORECAST, 0.08));
            }
            if run.len() < 2 {
                continue;
            }
            let line = Path::new(|b| {
                for (i, e) in run.iter().enumerate() {
                    let p = plot.point(e.target, e.value);
                    if i == 0 { b.move_to(p) } else { b.line_to(p) }
                }
            });
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(style::tint(style::FORECAST, 0.7))
                    .with_width(1.25)
                    .with_line_join(canvas::LineJoin::Round),
            );
        }
    }
}

/// The entries (sorted by target) with targets in `[start, end]`.
fn in_range(
    entries: &[ForecastHistoryEntry],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> &[ForecastHistoryEntry] {
    let from = entries.partition_point(|e| e.target < start);
    let to = entries.partition_point(|e| e.target <= end);
    &entries[from..to.max(from)]
}

/// `entries` split into runs of consecutive slots.
fn runs(entries: &[ForecastHistoryEntry]) -> impl Iterator<Item = &[ForecastHistoryEntry]> {
    entries.chunk_by(|a, b| b.target - a.target <= MAX_SLOT_GAP)
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    fn at(hour: u32, minute: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2026, 10, 9, hour, minute, 0)
            .single()
            .context("valid time")
    }

    fn entry(target: DateTime<Utc>) -> ForecastHistoryEntry {
        ForecastHistoryEntry {
            target,
            made_at: target - TimeDelta::hours(1),
            model: "naive".to_string(),
            value: 30.0,
            low: Some(25.0),
            high: Some(35.0),
        }
    }

    #[test]
    fn test_runs_split_where_slots_are_missing() -> Result<()> {
        let entries = [
            entry(at(10, 0)?),
            entry(at(10, 15)?),
            entry(at(10, 30)?),
            entry(at(11, 0)?),
            entry(at(11, 15)?),
        ];
        let lengths: Vec<usize> = runs(&entries).map(<[_]>::len).collect();
        assert_eq!(lengths, vec![3, 2]);
        assert_eq!(runs(&[]).count(), 0);
        Ok(())
    }

    #[test]
    fn test_in_range_is_inclusive_of_both_ends() -> Result<()> {
        let entries = [
            entry(at(9, 45)?),
            entry(at(10, 0)?),
            entry(at(10, 15)?),
            entry(at(10, 30)?),
        ];
        let inside = in_range(&entries, at(10, 0)?, at(10, 15)?);
        let targets: Vec<DateTime<Utc>> = inside.iter().map(|e| e.target).collect();
        assert_eq!(targets, vec![at(10, 0)?, at(10, 15)?]);
        assert_eq!(in_range(&entries, at(12, 0)?, at(13, 0)?).len(), 0);
        Ok(())
    }
}
