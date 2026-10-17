//! A day's expected occupancy per quarter hour: the range as a band, the
//! mean as a line (forecast in its own colour), the suggested window
//! highlighted, gaps where there is not enough data, and "now" on today.

use chrono::{DateTime, Utc};
use hardy_core::{
    Tz,
    analytics::visit::{EstimateSource, QUARTER, SlotEstimate, VisitWindow},
};
use iced::{
    Point, Rectangle, Renderer, Size, Theme,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, Frame, Path, Stroke, Text},
};

use crate::{
    style,
    widgets::history_chart::{time_ticks, value_axis_max},
};

const PAD_LEFT: f32 = 44.0;
const PAD_RIGHT: f32 = 12.0;
const PAD_TOP: f32 = 12.0;
const PAD_BOTTOM: f32 = 26.0;

/// Chart of one planned day.
pub struct DayCurve<'a> {
    /// One per open quarter hour, in order.
    pub estimates: &'a [SlotEstimate],
    pub window: Option<VisitWindow>,
    /// Drawn while it falls inside the day.
    pub now: DateTime<Utc>,
    pub timezone: Tz,
    pub cache: &'a canvas::Cache,
}

/// Runs of consecutive quarter hours with an estimate; unknown quarter
/// hours split them.
pub(crate) fn known_runs(estimates: &[SlotEstimate]) -> Vec<&[SlotEstimate]> {
    estimates
        .split(|e| e.expected.is_none())
        .filter(|run| !run.is_empty())
        .collect()
}

/// Where the forecast gives way to recent weeks, if both are on the day.
pub(crate) fn forecast_end(estimates: &[SlotEstimate]) -> Option<DateTime<Utc>> {
    estimates
        .windows(2)
        .find(|pair| {
            pair[0].source == EstimateSource::Forecast && pair[1].source != EstimateSource::Forecast
        })
        .map(|pair| pair[1].start)
}

/// Maps times and values to canvas coordinates.
#[derive(Clone, Copy)]
struct Plot {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    span_secs: f32,
    width: f32,
    height: f32,
    max: f64,
}

impl Plot {
    fn x(&self, t: DateTime<Utc>) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let offset = (t - self.start).num_seconds() as f32;
        PAD_LEFT + offset / self.span_secs * self.width
    }

    fn y(&self, value: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let ratio = (value.clamp(0.0, self.max) / self.max) as f32;
        PAD_TOP + self.height - ratio * self.height
    }

    fn bottom(&self) -> f32 {
        PAD_TOP + self.height
    }
}

impl DayCurve<'_> {
    fn plot(&self, size: Size) -> Option<Plot> {
        let (first, last) = (self.estimates.first()?, self.estimates.last()?);
        let end = last.start + QUARTER;
        #[allow(clippy::cast_precision_loss)]
        let span_secs = (end - first.start).num_seconds().max(1) as f32;
        Some(Plot {
            start: first.start,
            end,
            span_secs,
            width: (size.width - PAD_LEFT - PAD_RIGHT).max(1.0),
            height: (size.height - PAD_TOP - PAD_BOTTOM).max(1.0),
            max: value_axis_max(
                self.estimates
                    .iter()
                    .filter_map(|e| e.expected.map(|x| x.high)),
            ),
        })
    }

    fn draw_grid(&self, frame: &mut Frame, plot: &Plot) {
        let step = if plot.max <= 60.0 { 10.0 } else { 20.0 };
        let mut value = 0.0;
        while value <= plot.max {
            let y = plot.y(value);
            frame.stroke(
                &Path::line(
                    Point::new(PAD_LEFT, y),
                    Point::new(PAD_LEFT + plot.width, y),
                ),
                Stroke::default().with_color(style::GRID).with_width(1.0),
            );
            frame.fill_text(Text {
                content: format!("{value:.0}%"),
                position: Point::new(PAD_LEFT - 8.0, y),
                color: style::TEXT_TERTIARY,
                size: style::TEXT_CAPTION.into(),
                align_x: Horizontal::Right.into(),
                align_y: Vertical::Center,
                ..Default::default()
            });
            value += step;
        }
        for (t, label) in time_ticks(plot.start, plot.end, self.timezone) {
            let x = plot.x(t);
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

    fn draw_window(&self, frame: &mut Frame, plot: &Plot) {
        let Some(window) = self.window else { return };
        let (x0, x1) = (plot.x(window.start), plot.x(window.end));
        frame.fill_rectangle(
            Point::new(x0, PAD_TOP),
            Size::new((x1 - x0).max(1.0), plot.height),
            style::tint(style::OCC_QUIET, 0.14),
        );
    }

    fn draw_estimates(&self, frame: &mut Frame, plot: &Plot) {
        let half = QUARTER / 2;
        for run in known_runs(self.estimates) {
            let points: Vec<_> = run
                .iter()
                .filter_map(|e| e.expected.map(|x| (e.start + half, x, e.source)))
                .collect();
            let band = Path::new(|b| {
                for (i, (t, x, _)) in points.iter().enumerate() {
                    let p = Point::new(plot.x(*t), plot.y(x.high));
                    if i == 0 { b.move_to(p) } else { b.line_to(p) }
                }
                for (t, x, _) in points.iter().rev() {
                    b.line_to(Point::new(plot.x(*t), plot.y(x.low)));
                }
                b.close();
            });
            frame.fill(&band, style::tint(style::ACCENT, 0.12));
            for pair in points.windows(2) {
                let ((t0, a, s0), (t1, b, s1)) = (pair[0], pair[1]);
                let forecast = s0 == EstimateSource::Forecast && s1 == EstimateSource::Forecast;
                let color = if forecast {
                    style::FORECAST
                } else {
                    style::ACCENT
                };
                frame.stroke(
                    &Path::line(
                        Point::new(plot.x(t0), plot.y(a.mean)),
                        Point::new(plot.x(t1), plot.y(b.mean)),
                    ),
                    Stroke::default().with_color(color).with_width(2.0),
                );
            }
        }
    }

    fn draw_markers(&self, frame: &mut Frame, plot: &Plot) {
        let marker = |frame: &mut Frame, t: DateTime<Utc>, label: &str, color| {
            let x = plot.x(t);
            frame.stroke(
                &Path::line(Point::new(x, PAD_TOP), Point::new(x, plot.bottom())),
                Stroke::default().with_color(color).with_width(1.0),
            );
            frame.fill_text(Text {
                content: label.to_string(),
                position: Point::new(x + 4.0, PAD_TOP),
                color,
                size: style::TEXT_CAPTION.into(),
                align_y: Vertical::Top,
                ..Default::default()
            });
        };
        if let Some(t) = forecast_end(self.estimates) {
            marker(frame, t, "forecast ends", style::FORECAST);
        }
        if self.now > plot.start && self.now < plot.end {
            marker(frame, self.now, "now", style::TEXT_SECONDARY);
        }
    }
}

impl<Message> canvas::Program<Message> for DayCurve<'_> {
    type State = ();

    fn draw(
        &self,
        (): &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let Some(plot) = self.plot(bounds.size()) else {
            return Vec::new();
        };
        vec![self.cache.draw(renderer, bounds.size(), |frame| {
            self.draw_grid(frame, &plot);
            self.draw_window(frame, &plot);
            self.draw_estimates(frame, &plot);
            self.draw_markers(frame, &plot);
        })]
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use hardy_core::analytics::visit::Expected;

    use super::*;

    fn at(quarter: i32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 17, 4, 0, 0)
            .single()
            .unwrap_or_default()
            + QUARTER * quarter
    }

    fn estimate(quarter: i32, known: bool, source: EstimateSource) -> SlotEstimate {
        SlotEstimate {
            start: at(quarter),
            expected: known.then_some(Expected {
                mean: 20.0,
                low: 15.0,
                high: 25.0,
            }),
            source,
        }
    }

    #[test]
    fn test_unknown_quarter_hours_split_the_curve() {
        let weeks = EstimateSource::RecentWeeks;
        let estimates: Vec<_> = (0..8)
            .map(|q| estimate(q, !matches!(q, 2 | 3 | 7), weeks))
            .collect();
        let runs: Vec<Vec<DateTime<Utc>>> = known_runs(&estimates)
            .iter()
            .map(|r| r.iter().map(|e| e.start).collect())
            .collect();
        assert_eq!(runs, vec![vec![at(0), at(1)], vec![at(4), at(5), at(6)]]);
        assert_eq!(known_runs(&[]).len(), 0);
    }

    #[test]
    fn test_forecast_end_is_where_recent_weeks_take_over() {
        let estimates: Vec<_> = (0..6)
            .map(|q| {
                let source = if q < 4 {
                    EstimateSource::Forecast
                } else {
                    EstimateSource::RecentWeeks
                };
                estimate(q, true, source)
            })
            .collect();
        assert_eq!(forecast_end(&estimates), Some(at(4)));
    }

    #[test]
    fn test_no_forecast_end_without_a_switch() {
        let all = |source| -> Vec<_> { (0..4).map(|q| estimate(q, true, source)).collect() };
        assert_eq!(forecast_end(&all(EstimateSource::Forecast)), None);
        assert_eq!(forecast_end(&all(EstimateSource::RecentWeeks)), None);
    }
}
