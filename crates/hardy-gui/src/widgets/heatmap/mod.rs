//! Weekday × hour grid of average occupancy, limited to opening hours.

mod grid;

use std::ops::Range;

pub use grid::{Cell, WeekGrid};
use hardy_core::analytics::weekday_short;
use iced::{
    Point, Rectangle, Renderer, Size, Theme,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, Action, Path, Stroke, Text},
};

use crate::style;

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
