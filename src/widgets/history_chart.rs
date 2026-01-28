use chrono::{DateTime, Duration as ChronoDuration, Local, Timelike, Utc};
use iced::{
    Color, Point, Rectangle, Renderer, Size, Theme, mouse,
    widget::canvas::{self, Action, Frame, LineDash, Path, Stroke, Text},
};

use crate::{analytics::midnight_utc, db::OccupancyLog, style};

// Interaction event to avoid circular dependency on Message
#[derive(Debug, Clone, Copy)]
pub enum Interaction {
    Hovered,
}

pub struct HistoryChart<'a> {
    pub history: &'a [OccupancyLog],
    pub predictions: &'a [(DateTime<Utc>, f64)],
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub cache: &'a canvas::Cache,
}

impl<'a> canvas::Program<Interaction> for HistoryChart<'a> {
    type State = ();

    fn update(
        &self,
        _state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Interaction>> {
        if let iced::Event::Mouse(mouse::Event::CursorMoved { .. }) = event {
            if cursor.position_in(bounds).is_some() {
                return Some(Action::publish(Interaction::Hovered));
            }
        }
        None
    }

    fn draw(
        &self,
        _: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let pad_left = 35.0;
            let pad_bottom = 25.0;
            let pad_top = 10.0;
            let pad_right = 10.0;
            let w = bounds.width - pad_left - pad_right;
            let h = bounds.height - pad_bottom - pad_top;

            // Y-Axis
            for i in 0..=4 {
                let pct = i as f32 * 25.0;
                let y = pad_top + h - (pct / 100.0 * h);
                let line = Path::line(Point::new(pad_left, y), Point::new(pad_left + w, y));
                frame.stroke(
                    &line,
                    Stroke::default()
                        .with_color(style::STROKE_DIM)
                        .with_width(1.0),
                );
                frame.fill_text(Text {
                    content: format!("{:.0}", pct),
                    position: Point::new(pad_left - 5.0, y),
                    color: style::TEXT_MUTED,
                    size: 10.0.into(),
                    align_x: iced::alignment::Horizontal::Right.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });
            }

            // X-Axis logic
            let dur = (self.range_end - self.range_start).num_seconds();
            if dur > 0 {
                let tick_interval = if dur <= 86400 * 2 {
                    3600 * 4
                } else if dur <= 86400 * 8 {
                    86400
                } else {
                    86400 * 7
                };
                let mut current = self.range_start;

                // Align first tick
                if tick_interval == 3600 * 4 {
                    let rem = current.hour() % 4;
                    if rem != 0 {
                        current += ChronoDuration::hours((4 - rem) as i64);
                    }
                    current = current.with_minute(0).unwrap().with_second(0).unwrap();
                } else if tick_interval >= 86400 {
                    current = midnight_utc(current.date_naive() + ChronoDuration::days(1));
                }

                while current < self.range_end {
                    let offset = current
                        .signed_duration_since(self.range_start)
                        .num_seconds() as f32;
                    let x = pad_left + (offset / dur as f32) * w;
                    frame.stroke(
                        &Path::line(Point::new(x, pad_top), Point::new(x, pad_top + h)),
                        Stroke::default()
                            .with_color(style::STROKE_DIM)
                            .with_width(0.5),
                    );
                    let label = if tick_interval < 86400 {
                        current.with_timezone(&Local).format("%H:%M").to_string()
                    } else {
                        current.with_timezone(&Local).format("%b %d").to_string()
                    };
                    frame.fill_text(Text {
                        content: label,
                        position: Point::new(x, bounds.height - 10.0),
                        color: style::TEXT_MUTED,
                        size: 10.0.into(),
                        align_x: iced::alignment::Horizontal::Center.into(),
                        align_y: iced::alignment::Vertical::Bottom,
                        ..Default::default()
                    });
                    current += ChronoDuration::seconds(tick_interval);
                }
            }

            let dur_f = dur as f32;
            let to_pt = |dt: DateTime<Utc>, val: f64| {
                let offset = dt.signed_duration_since(self.range_start).num_seconds() as f32;
                let x = pad_left + (offset / dur_f) * w;
                let y = pad_top + h - (val as f32 / 100.0 * h);
                Point::new(x, y)
            };

            // Draw History
            let mut last_history_point: Option<(Point, DateTime<Utc>)> = None;
            let points: Vec<_> = self
                .history
                .iter()
                .filter_map(|l| l.datetime().map(|d| (d, l.percentage)))
                .filter(|(d, _)| *d >= self.range_start && *d <= self.range_end)
                .collect();

            if !points.is_empty() {
                let mut builder = canvas::path::Builder::new();
                let first = points[0];
                builder.move_to(to_pt(first.0, first.1));
                for (d, v) in &points {
                    builder.line_to(to_pt(*d, *v));
                }
                let line_path = builder.build();

                let mut fill = canvas::path::Builder::new();
                fill.move_to(Point::new(to_pt(first.0, 0.0).x, pad_top + h));
                for (d, v) in &points {
                    fill.line_to(to_pt(*d, *v));
                }
                if let Some(last) = points.last() {
                    fill.line_to(Point::new(to_pt(last.0, 0.0).x, pad_top + h));
                    last_history_point = Some((to_pt(last.0, last.1), last.0));
                }
                fill.close();

                frame.fill(&fill.build(), Color::from_rgba(0.35, 0.65, 0.95, 0.1));
                frame.stroke(
                    &line_path,
                    Stroke::default()
                        .with_color(style::ACCENT_BLUE)
                        .with_width(2.0),
                );
            }

            // Draw Predictions
            if !self.predictions.is_empty() {
                let mut builder = canvas::path::Builder::new();
                let mut started = false;

                // Connect history to prediction
                if let Some((pt, dt)) = last_history_point {
                    if let Some(first_pred) = self.predictions.first() {
                        if (first_pred.0 - dt).num_hours() < 4 {
                            builder.move_to(pt);
                            started = true;
                        }
                    }
                }

                for (d, v) in self.predictions {
                    if *d >= self.range_start && *d <= self.range_end + ChronoDuration::hours(2) {
                        let pt = to_pt(*d, *v);
                        // Clip width
                        if pt.x <= w + pad_left + 20.0 {
                            if !started {
                                builder.move_to(pt);
                                started = true;
                            } else {
                                builder.line_to(pt);
                            }
                            frame.fill(&Path::circle(pt, 3.0), style::ACCENT_CYAN);
                        }
                    }
                }

                if started {
                    frame.stroke(
                        &builder.build(),
                        Stroke {
                            style: style::ACCENT_CYAN.into(),
                            width: 2.0,
                            line_dash: LineDash {
                                segments: &[4.0, 6.0],
                                offset: 0,
                            },
                            ..Stroke::default()
                        },
                    );
                }
            }
        });

        let mut geometries = vec![geo];

        // Hover Overlay
        if let Some(cursor_pos) = cursor.position_in(bounds) {
            if !self.history.is_empty() {
                let pad_left = 35.0;
                let w = bounds.width - pad_left - 10.0;
                let dur = (self.range_end - self.range_start).num_seconds() as f32;
                let ratio = (cursor_pos.x - pad_left) / w;

                if (0.0..=1.0).contains(&ratio) {
                    let time_offset = ratio * dur;
                    let hover_time = self.range_start + ChronoDuration::seconds(time_offset as i64);
                    let closest = self
                        .history
                        .iter()
                        .filter_map(|l| l.datetime().map(|d| (d, l.percentage)))
                        .min_by_key(|(d, _)| (*d - hover_time).num_seconds().abs());

                    if let Some((d, val)) = closest {
                        if d >= self.range_start && d <= self.range_end {
                            let pad_top = 10.0;
                            let h = bounds.height - 25.0 - pad_top;
                            let x = pad_left
                                + (d.signed_duration_since(self.range_start).num_seconds() as f32
                                    / dur)
                                    * w;
                            let y = pad_top + h - (val as f32 / 100.0 * h);

                            let mut frame = Frame::new(renderer, bounds.size());
                            frame.stroke(
                                &Path::line(Point::new(x, pad_top), Point::new(x, pad_top + h)),
                                Stroke {
                                    style: style::TEXT_BRIGHT.into(),
                                    width: 1.0,
                                    line_dash: LineDash {
                                        segments: &[4.0, 4.0],
                                        offset: 0,
                                    },
                                    ..Stroke::default()
                                },
                            );
                            frame.fill(&Path::circle(Point::new(x, y), 4.0), style::ACCENT_CYAN);

                            // Tooltip
                            let text_str =
                                format!("{}\n{:.1}%", d.with_timezone(&Local).format("%H:%M"), val);
                            let (box_w, box_h) = (60.0, 35.0);
                            let box_x = if x + 10.0 + box_w > bounds.width {
                                x - 10.0 - box_w
                            } else {
                                x + 10.0
                            };
                            let box_y = if y - 20.0 < 0.0 { y + 10.0 } else { y - 20.0 };

                            frame.fill(
                                &Path::rounded_rectangle(
                                    Point::new(box_x, box_y),
                                    Size::new(box_w, box_h),
                                    4.0.into(),
                                ),
                                style::TOOLTIP_BG,
                            );
                            frame.fill_text(Text {
                                content: text_str,
                                position: Point::new(box_x + box_w / 2.0, box_y + box_h / 2.0),
                                color: style::TEXT_BRIGHT,
                                size: 12.0.into(),
                                align_x: iced::alignment::Horizontal::Center.into(),
                                align_y: iced::alignment::Vertical::Center,
                                ..Default::default()
                            });
                            geometries.push(frame.into_geometry());
                        }
                    }
                }
            }
        }
        geometries
    }
}
