//! Occupancy over time: measured readings, the forecast with its 80% range,
//! a "now" marker and the alert threshold.

mod axis;
mod layers;
mod plot;

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{Tz, db::OccupancyLog};
use hardy_ml::PredictionWithConfidence;
use iced::{
    Point, Rectangle, Renderer, Size, Theme,
    alignment::Vertical,
    mouse,
    widget::canvas::{self, Action, Frame, Path, Stroke, Text},
};
use plot::Plot;

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
