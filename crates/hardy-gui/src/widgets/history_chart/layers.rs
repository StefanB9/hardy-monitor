//! The chart's drawing layers: grid, readings, forecast, markers, hover.

use chrono::{DateTime, TimeDelta, Utc};
use iced::{
    Point,
    alignment::{Horizontal, Vertical},
    widget::canvas::{self, Frame, LineDash, Path, Stroke, Text, gradient},
};

use super::{DASH, FORECAST_JOIN, HistoryChart, MAX_GAP, axis, plot::Plot};
use crate::style;

impl HistoryChart<'_> {
    pub(super) fn draw_grid(&self, frame: &mut Frame, plot: &Plot) {
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
    pub(super) fn draw_history(
        &self,
        frame: &mut Frame,
        plot: &Plot,
    ) -> Option<(DateTime<Utc>, f64)> {
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

    pub(super) fn draw_forecast(
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

    pub(super) fn draw_markers(&self, frame: &mut Frame, plot: &Plot) {
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
    pub(super) fn hovered(&self, plot: &Plot, x: f32) -> Option<(Point, String, iced::Color)> {
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
