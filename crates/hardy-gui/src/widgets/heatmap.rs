//! Weekday × hour grid of average occupancy, limited to opening hours.

use std::ops::Range;

use chrono::NaiveDate;
use hardy_core::{analytics::weekday_short, db::HourlyAverage, schedule::GymSchedule};
use iced::{
    Point, Rectangle, Renderer, Size, Theme,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, Action, Path, Stroke, Text},
};

use crate::style;

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

/// Heatmap of a [`WeekGrid`], coloured on the occupancy scale.
pub struct HeatmapWidget<'a> {
    pub grid: &'a WeekGrid,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub cache: &'a canvas::Cache,
    pub tooltip_cache: &'a canvas::Cache,
}

const PAD_LEFT: f32 = 44.0;
const PAD_BOTTOM: f32 = 24.0;
const GAP: f32 = 3.0;

struct Layout {
    hours: Range<u32>,
    cell_w: f32,
    cell_h: f32,
    grid_h: f32,
}

impl HeatmapWidget<'_> {
    fn layout(&self, bounds: Rectangle) -> Layout {
        let hours = self.grid.hour_span();
        #[allow(clippy::cast_precision_loss)]
        let columns = hours.len().max(1) as f32;
        let grid_h = bounds.height - PAD_BOTTOM;
        Layout {
            cell_w: (bounds.width - PAD_LEFT) / columns,
            cell_h: grid_h / 7.0,
            grid_h,
            hours,
        }
    }

    /// `(day, hour)` under the cursor.
    fn cell_at(&self, bounds: Rectangle, cursor: mouse::Cursor) -> Option<(usize, u32)> {
        let layout = self.layout(bounds);
        let pos = cursor.position_in(bounds)?;
        if pos.x < PAD_LEFT || pos.y >= layout.grid_h {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (col, day) = (
            ((pos.x - PAD_LEFT) / layout.cell_w) as u32,
            (pos.y / layout.cell_h) as usize,
        );
        let hour = layout.hours.start + col;
        (day < 7 && layout.hours.contains(&hour)).then_some((day, hour))
    }

    fn draw_tooltip(
        &self,
        frame: &mut canvas::Frame,
        layout: &Layout,
        bounds: Rectangle,
        day: usize,
        hour: u32,
    ) {
        let detail = match self.grid.cell(day, hour) {
            Cell::Closed => "Closed".to_string(),
            Cell::NoData => "No data yet".to_string(),
            Cell::Value(value, samples) => format!("{value:.0}% · {samples} readings"),
        };
        let label = format!(
            "{} {hour:02}:00–{:02}:00\n{detail}",
            day_label(day),
            hour + 1
        );

        #[allow(clippy::cast_precision_loss)]
        let (cx, cy) = (
            PAD_LEFT + (hour - layout.hours.start) as f32 * layout.cell_w,
            day as f32 * layout.cell_h,
        );
        frame.stroke(
            &Path::rounded_rectangle(
                Point::new(cx + 1.0, cy + 1.0),
                Size::new(layout.cell_w - 2.0, layout.cell_h - 2.0),
                4.0.into(),
            ),
            Stroke::default()
                .with_color(style::TEXT_PRIMARY)
                .with_width(2.0),
        );

        let size = Size::new(170.0, 44.0);
        let x = (cx + layout.cell_w + 8.0).min(bounds.width - size.width);
        let x = if x < cx + layout.cell_w {
            cx - size.width - 8.0
        } else {
            x
        };
        let y =
            (cy + layout.cell_h / 2.0 - size.height / 2.0).clamp(0.0, bounds.height - size.height);
        let bg = Path::rounded_rectangle(Point::new(x, y), size, 8.0.into());
        frame.fill(&bg, style::BG_ELEVATED);
        frame.stroke(
            &bg,
            Stroke::default().with_color(style::BORDER).with_width(1.0),
        );
        frame.fill_text(Text {
            content: label,
            position: Point::new(x + 10.0, y + size.height / 2.0),
            color: style::TEXT_PRIMARY,
            size: style::TEXT_CAPTION.into(),
            align_y: Vertical::Center,
            ..Default::default()
        });
    }
}

fn day_label(day: usize) -> &'static str {
    weekday_short(i32::try_from(day).unwrap_or(0))
}

impl<Message> canvas::Program<Message> for HeatmapWidget<'_> {
    type State = Option<(usize, u32)>;

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        if let iced::Event::Mouse(mouse::Event::CursorMoved { .. } | mouse::Event::CursorLeft) =
            event
        {
            let hovered = self.cell_at(bounds, cursor);
            if *state != hovered {
                *state = hovered;
                self.tooltip_cache.clear();
                return Some(Action::request_redraw());
            }
        }
        None
    }

    fn draw(
        &self,
        state: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let layout = self.layout(bounds);
        let show_values = layout.cell_w >= 26.0 && layout.cell_h >= 20.0;

        let grid = self.cache.draw(renderer, bounds.size(), |frame| {
            for day in 0..7 {
                #[allow(clippy::cast_precision_loss)]
                let y = day as f32 * layout.cell_h;
                frame.fill_text(Text {
                    content: day_label(day).to_string(),
                    position: Point::new(0.0, y + layout.cell_h / 2.0),
                    color: style::TEXT_SECONDARY,
                    size: style::TEXT_CAPTION.into(),
                    align_y: Vertical::Center,
                    ..Default::default()
                });

                for (col, hour) in layout.hours.clone().enumerate() {
                    #[allow(clippy::cast_precision_loss)]
                    let x = PAD_LEFT + col as f32 * layout.cell_w;
                    let rect = Path::rounded_rectangle(
                        Point::new(x + GAP / 2.0, y + GAP / 2.0),
                        Size::new(layout.cell_w - GAP, layout.cell_h - GAP),
                        4.0.into(),
                    );
                    let center = Point::new(x + layout.cell_w / 2.0, y + layout.cell_h / 2.0);
                    match self.grid.cell(day, hour) {
                        Cell::Closed => {
                            frame.fill(&rect, style::tint(style::BG_ELEVATED, 0.25));
                        }
                        Cell::NoData => {
                            frame.fill(&rect, style::NO_DATA);
                            frame.stroke(
                                &rect,
                                Stroke::default().with_color(style::BORDER).with_width(1.0),
                            );
                        }
                        Cell::Value(value, _) => {
                            frame.fill(
                                &rect,
                                style::occupancy_color(
                                    value,
                                    self.low_threshold,
                                    self.high_threshold,
                                ),
                            );
                            if show_values {
                                frame.fill_text(Text {
                                    content: format!("{value:.0}"),
                                    position: center,
                                    color: style::tint(style::BG_APP, 0.85),
                                    size: style::TEXT_CAPTION.into(),
                                    align_x: Horizontal::Center.into(),
                                    align_y: Vertical::Center,
                                    ..Default::default()
                                });
                            }
                        }
                    }

                    if day == 6 && hour % 2 == 0 {
                        frame.fill_text(Text {
                            content: format!("{hour:02}"),
                            position: Point::new(x + layout.cell_w / 2.0, layout.grid_h + 8.0),
                            color: style::TEXT_TERTIARY,
                            size: style::TEXT_CAPTION.into(),
                            align_x: Horizontal::Center.into(),
                            align_y: Vertical::Top,
                            ..Default::default()
                        });
                    }
                }
            }
        });

        let tooltip = self.tooltip_cache.draw(renderer, bounds.size(), |frame| {
            if let Some((day, hour)) = *state {
                self.draw_tooltip(frame, &layout, bounds, day, hour);
            }
        });

        vec![grid, tooltip]
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
