//! Occupancy over time: measured readings, the forecast with its 80% range,
//! a "now" marker and the alert threshold.

mod axis;

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{Tz, db::OccupancyLog};
use hardy_ml::PredictionWithConfidence;
use iced::{
    Point, Rectangle, Renderer, Size, Theme,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, Action, Frame, LineDash, Path, Stroke, Text, gradient},
};

use crate::style;

/// Readings further apart than this are drawn as separate segments.
const MAX_GAP: TimeDelta = TimeDelta::minutes(20);
/// The forecast line joins the last reading if it starts within this.
const FORECAST_JOIN: TimeDelta = TimeDelta::hours(2);

const PAD_LEFT: f32 = 44.0;
const PAD_RIGHT: f32 = 12.0;
const PAD_TOP: f32 = 12.0;
const PAD_BOTTOM: f32 = 26.0;

const DASH: &[f32] = &[5.0, 5.0];

/// Chart of one time range.
pub struct HistoryChart<'a> {
    /// Readings sorted by time; only those inside the range are drawn.
    pub history: &'a [OccupancyLog],
    pub forecast: &'a [PredictionWithConfidence],
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub now: DateTime<Utc>,
    /// Alert threshold, drawn while alerts are on.
    pub threshold: Option<f64>,
    /// Gym timezone; axis ticks and labels follow its wall clock.
    pub timezone: Tz,
    pub cache: &'a canvas::Cache,
}

/// Maps times and values to canvas coordinates.
#[derive(Clone, Copy)]
struct Plot {
    start: DateTime<Utc>,
    span_secs: f32,
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    max_value: f64,
}

impl Plot {
    fn new(chart: &HistoryChart<'_>, size: Size, max_value: f64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let span_secs = (chart.range_end - chart.range_start).num_seconds().max(1) as f32;
        Self {
            start: chart.range_start,
            span_secs,
            left: PAD_LEFT,
            top: PAD_TOP,
            width: (size.width - PAD_LEFT - PAD_RIGHT).max(1.0),
            height: (size.height - PAD_TOP - PAD_BOTTOM).max(1.0),
            max_value,
        }
    }

    fn x(&self, t: DateTime<Utc>) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let offset = (t - self.start).num_seconds() as f32;
        self.left + offset / self.span_secs * self.width
    }

    fn y(&self, value: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let ratio = (value.clamp(0.0, self.max_value) / self.max_value) as f32;
        self.top + self.height - ratio * self.height
    }

    fn point(&self, t: DateTime<Utc>, value: f64) -> Point {
        Point::new(self.x(t), self.y(value))
    }

    fn bottom(&self) -> f32 {
        self.top + self.height
    }

    fn time_at(&self, x: f32) -> Option<DateTime<Utc>> {
        let ratio = (x - self.left) / self.width;
        #[allow(clippy::cast_possible_truncation)]
        (0.0..=1.0)
            .contains(&ratio)
            .then(|| self.start + TimeDelta::seconds((ratio * self.span_secs) as i64))
    }
}

impl HistoryChart<'_> {
    fn readings(&self) -> &[OccupancyLog] {
        let from = self
            .history
            .partition_point(|l| l.timestamp < self.range_start);
        let to = self
            .history
            .partition_point(|l| l.timestamp <= self.range_end);
        &self.history[from..to.max(from)]
    }

    fn forecast_in_range(&self) -> impl Iterator<Item = &PredictionWithConfidence> {
        self.forecast
            .iter()
            .filter(|p| p.timestamp >= self.range_start && p.timestamp <= self.range_end)
    }

    fn plot(&self, size: Size) -> Plot {
        let max = axis::value_axis_max(
            self.readings()
                .iter()
                .map(|l| l.percentage)
                .chain(self.forecast_in_range().map(|p| p.confidence_high))
                .chain(self.threshold),
        );
        Plot::new(self, size, max)
    }

    fn draw_grid(&self, frame: &mut Frame, plot: &Plot) {
        let step = if plot.max_value <= 60.0 { 10.0 } else { 20.0 };
        let mut value = 0.0;
        while value <= plot.max_value {
            let y = plot.y(value);
            frame.stroke(
                &Path::line(
                    Point::new(plot.left, y),
                    Point::new(plot.left + plot.width, y),
                ),
                Stroke::default().with_color(style::GRID).with_width(1.0),
            );
            frame.fill_text(Text {
                content: format!("{value:.0}%"),
                position: Point::new(plot.left - 8.0, y),
                color: style::TEXT_TERTIARY,
                size: style::TEXT_CAPTION.into(),
                align_x: Horizontal::Right.into(),
                align_y: Vertical::Center,
                ..Default::default()
            });
            value += step;
        }

        for (t, label) in axis::time_ticks(self.range_start, self.range_end, self.timezone) {
            let x = plot.x(t);
            frame.stroke(
                &Path::line(Point::new(x, plot.top), Point::new(x, plot.bottom())),
                Stroke::default()
                    .with_color(style::tint(style::GRID, 0.6))
                    .with_width(1.0),
            );
            frame.fill_text(Text {
                content: label,
                position: Point::new(x, plot.bottom() + 8.0),
                color: style::TEXT_TERTIARY,
                size: style::TEXT_CAPTION.into(),
                align_x: Horizontal::Center.into(),
                align_y: Vertical::Top,
                ..Default::default()
            });
        }
    }

    /// Draws the measured line; returns the last drawn reading.
    fn draw_history(&self, frame: &mut Frame, plot: &Plot) -> Option<(DateTime<Utc>, f64)> {
        let readings = self.readings();
        let mut start = 0;
        while start < readings.len() {
            let mut end = start + 1;
            while end < readings.len()
                && readings[end].timestamp - readings[end - 1].timestamp <= MAX_GAP
            {
                end += 1;
            }
            let segment = &readings[start..end];
            let line = Path::new(|b| {
                for (i, l) in segment.iter().enumerate() {
                    let p = plot.point(l.timestamp, l.percentage);
                    if i == 0 { b.move_to(p) } else { b.line_to(p) }
                }
            });
            let area = Path::new(|b| {
                b.move_to(Point::new(plot.x(segment[0].timestamp), plot.bottom()));
                for l in segment {
                    b.line_to(plot.point(l.timestamp, l.percentage));
                }
                b.line_to(Point::new(
                    plot.x(segment[segment.len() - 1].timestamp),
                    plot.bottom(),
                ));
                b.close();
            });
            frame.fill(
                &area,
                gradient::Linear::new(Point::new(0.0, plot.top), Point::new(0.0, plot.bottom()))
                    .add_stop(0.0, style::tint(style::ACCENT, 0.28))
                    .add_stop(1.0, style::tint(style::ACCENT, 0.02)),
            );
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(style::ACCENT)
                    .with_width(2.0)
                    .with_line_join(canvas::LineJoin::Round),
            );
            start = end;
        }
        readings.last().map(|l| (l.timestamp, l.percentage))
    }

    fn draw_forecast(
        &self,
        frame: &mut Frame,
        plot: &Plot,
        last_reading: Option<(DateTime<Utc>, f64)>,
    ) {
        let points: Vec<_> = self.forecast_in_range().collect();
        let Some(first) = points.first() else {
            return;
        };

        let band = Path::new(|b| {
            b.move_to(plot.point(first.timestamp, first.confidence_high));
            for p in &points {
                b.line_to(plot.point(p.timestamp, p.confidence_high));
            }
            for p in points.iter().rev() {
                b.line_to(plot.point(p.timestamp, p.confidence_low));
            }
            b.close();
        });
        frame.fill(&band, style::tint(style::FORECAST, 0.14));

        let joined = last_reading.filter(|(t, _)| first.timestamp - *t <= FORECAST_JOIN);
        let line = Path::new(|b| {
            match joined {
                Some((t, v)) => b.move_to(plot.point(t, v)),
                None => b.move_to(plot.point(first.timestamp, first.predicted_value)),
            }
            for p in &points {
                b.line_to(plot.point(p.timestamp, p.predicted_value));
            }
        });
        frame.stroke(
            &line,
            Stroke {
                line_dash: LineDash {
                    segments: DASH,
                    offset: 0,
                },
                ..Stroke::default()
                    .with_color(style::FORECAST)
                    .with_width(2.0)
            },
        );
        for p in &points {
            frame.fill(
                &Path::circle(plot.point(p.timestamp, p.predicted_value), 3.0),
                style::FORECAST,
            );
        }
    }

    fn draw_markers(&self, frame: &mut Frame, plot: &Plot) {
        if let Some(threshold) = self.threshold {
            let y = plot.y(threshold);
            frame.stroke(
                &Path::line(
                    Point::new(plot.left, y),
                    Point::new(plot.left + plot.width, y),
                ),
                Stroke {
                    line_dash: LineDash {
                        segments: DASH,
                        offset: 0,
                    },
                    ..Stroke::default()
                        .with_color(style::tint(style::SUCCESS, 0.7))
                        .with_width(1.0)
                },
            );
            frame.fill_text(Text {
                content: format!("Alert below {threshold:.0}%"),
                position: Point::new(plot.left + plot.width - 4.0, y - 4.0),
                color: style::SUCCESS,
                size: style::TEXT_CAPTION.into(),
                align_x: Horizontal::Right.into(),
                align_y: Vertical::Bottom,
                ..Default::default()
            });
        }

        if self.now > self.range_start && self.now < self.range_end {
            let x = plot.x(self.now);
            frame.stroke(
                &Path::line(Point::new(x, plot.top), Point::new(x, plot.bottom())),
                Stroke::default()
                    .with_color(style::TEXT_TERTIARY)
                    .with_width(1.0),
            );
            frame.fill_text(Text {
                content: "Now".to_string(),
                position: Point::new(x + 4.0, plot.top),
                color: style::TEXT_SECONDARY,
                size: style::TEXT_CAPTION.into(),
                align_y: Vertical::Top,
                ..Default::default()
            });
        }
    }

    /// The reading or forecast point nearest the cursor, with its label.
    fn hovered(&self, plot: &Plot, x: f32) -> Option<(Point, String, iced::Color)> {
        let t = plot.time_at(x)?;
        let readings = self.readings();
        let idx = readings.partition_point(|l| l.timestamp < t);
        let reading = [idx.checked_sub(1), Some(idx)]
            .into_iter()
            .flatten()
            .filter_map(|i| readings.get(i))
            .min_by_key(|l| (l.timestamp - t).abs())
            .map(|l| ((l.timestamp - t).abs(), l));
        let forecast = self
            .forecast_in_range()
            .min_by_key(|p| (p.timestamp - t).abs())
            .map(|p| ((p.timestamp - t).abs(), p));

        let long_range = self.range_end - self.range_start > TimeDelta::hours(26);
        let fmt = if long_range {
            "%a %d · %H:%M"
        } else {
            "%H:%M"
        };
        let when = |time: DateTime<Utc>| time.with_timezone(&self.timezone).format(fmt);

        match (reading, forecast) {
            (Some((dr, l)), forecast) if forecast.is_none_or(|(df, _)| dr <= df) => Some((
                plot.point(l.timestamp, l.percentage),
                format!("{}\n{:.0}%", when(l.timestamp), l.percentage),
                style::ACCENT,
            )),
            (_, Some((_, p))) => Some((
                plot.point(p.timestamp, p.predicted_value),
                format!(
                    "{} forecast\n{:.0}% ({:.0}–{:.0})",
                    when(p.timestamp),
                    p.predicted_value,
                    p.confidence_low,
                    p.confidence_high
                ),
                style::FORECAST,
            )),
            _ => None,
        }
    }
}

impl<Message> canvas::Program<Message> for HistoryChart<'_> {
    type State = ();

    fn update(
        &self,
        (): &mut Self::State,
        event: &iced::Event,
        _: Rectangle,
        _: mouse::Cursor,
    ) -> Option<Action<Message>> {
        // The tooltip follows the cursor; the base layer stays cached.
        matches!(
            event,
            iced::Event::Mouse(mouse::Event::CursorMoved { .. } | mouse::Event::CursorLeft)
        )
        .then(Action::request_redraw)
    }

    fn draw(
        &self,
        (): &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let plot = self.plot(bounds.size());
        let base = self.cache.draw(renderer, bounds.size(), |frame| {
            self.draw_grid(frame, &plot);
            let last = self.draw_history(frame, &plot);
            self.draw_forecast(frame, &plot, last);
            self.draw_markers(frame, &plot);
        });

        let mut layers = vec![base];
        if let Some(pos) = cursor.position_in(bounds)
            && let Some((point, label, color)) = self.hovered(&plot, pos.x)
        {
            let mut frame = Frame::new(renderer, bounds.size());
            frame.stroke(
                &Path::line(
                    Point::new(point.x, plot.top),
                    Point::new(point.x, plot.bottom()),
                ),
                Stroke::default()
                    .with_color(style::TEXT_TERTIARY)
                    .with_width(1.0),
            );
            frame.fill(&Path::circle(point, 5.0), style::BG_CARD);
            frame.stroke(
                &Path::circle(point, 5.0),
                Stroke::default().with_color(color).with_width(2.0),
            );

            let size = Size::new(150.0, 44.0);
            let x = if point.x + 12.0 + size.width > bounds.width {
                point.x - 12.0 - size.width
            } else {
                point.x + 12.0
            };
            let y = (point.y - size.height / 2.0).clamp(0.0, bounds.height - size.height);
            let tooltip = Path::rounded_rectangle(Point::new(x, y), size, 8.0.into());
            frame.fill(&tooltip, style::BG_ELEVATED);
            frame.stroke(
                &tooltip,
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
            layers.push(frame.into_geometry());
        }
        layers
    }
}
