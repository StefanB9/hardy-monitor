//! Lines over a numeric x axis (days, lead times), with a hover tooltip
//! listing every line's value at the nearest x.

use iced::{
    Color, Point, Rectangle, Renderer, Theme, mouse,
    widget::canvas::{self, Action, Frame, Path, Stroke, path::Builder},
};

use super::{PAD_LEFT, PAD_TOP, Scale, draw_tooltip, draw_value_grid, draw_x_ticks, nice_scale};
use crate::{app::Message, style};

/// Points are marked with dots while a line has at most this many.
const DOTS_UP_TO: usize = 40;
const DASH: &[f32] = &[5.0, 4.0];

/// One line.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// Shown in the tooltip.
    pub name: String,
    pub color: Color,
    pub width: f32,
    pub dashed: bool,
    /// Sorted by x.
    pub points: Vec<(f64, f64)>,
}

/// A line chart; values start at zero.
pub struct LineChart<'a, F> {
    pub series: Vec<Series>,
    pub x_ticks: Vec<(f64, String)>,
    /// Names an x value in the tooltip.
    pub x_label: F,
    /// After each value label, e.g. " pts" or "%".
    pub suffix: &'static str,
    /// A dashed reference value, e.g. the 80 % a band should cover.
    pub target: Option<f64>,
    pub cache: &'a canvas::Cache,
}

/// The x value nearest to `cursor_x` on screen, among `xs`.
pub(super) fn nearest_x(
    xs: impl Iterator<Item = f64>,
    scale: &Scale,
    cursor_x: f32,
) -> Option<f64> {
    xs.min_by(|a, b| {
        (scale.x(*a) - cursor_x)
            .abs()
            .total_cmp(&(scale.x(*b) - cursor_x).abs())
    })
}

impl<F: Fn(f64) -> String> LineChart<'_, F> {
    fn xs(&self) -> impl Iterator<Item = f64> + '_ {
        self.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.0))
    }

    fn scale(&self, size: iced::Size) -> (Scale, f64) {
        let (x_min, x_max) = self
            .xs()
            .fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
        let y_max = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.1))
            .chain(self.target)
            .fold(0.0, f64::max);
        let (top, step) = nice_scale(y_max);
        let x = if x_min <= x_max {
            (x_min, x_max)
        } else {
            (0.0, 1.0)
        };
        (Scale::new(size, x, top), step)
    }

    fn draw_lines(&self, frame: &mut Frame, scale: &Scale) {
        if let Some(target) = self.target {
            let y = scale.y(target);
            frame.stroke(
                &Path::line(
                    Point::new(PAD_LEFT, y),
                    Point::new(PAD_LEFT + scale.width, y),
                ),
                Stroke {
                    line_dash: canvas::LineDash {
                        segments: DASH,
                        offset: 0,
                    },
                    ..Stroke::default()
                        .with_color(style::TEXT_TERTIARY)
                        .with_width(1.0)
                },
            );
        }
        for series in &self.series {
            let path = Path::new(|b: &mut Builder| {
                for (i, (x, y)) in series.points.iter().enumerate() {
                    let p = scale.point(*x, *y);
                    if i == 0 { b.move_to(p) } else { b.line_to(p) }
                }
            });
            let mut stroke = Stroke::default()
                .with_color(series.color)
                .with_width(series.width);
            if series.dashed {
                stroke.line_dash = canvas::LineDash {
                    segments: DASH,
                    offset: 0,
                };
            }
            frame.stroke(&path, stroke);
            if series.points.len() <= DOTS_UP_TO {
                for (x, y) in &series.points {
                    frame.fill(&Path::circle(scale.point(*x, *y), 2.5), series.color);
                }
            }
        }
    }
}

impl<F: Fn(f64) -> String> canvas::Program<Message> for LineChart<'_, F> {
    type State = ();

    fn update(
        &self,
        (): &mut Self::State,
        event: &iced::Event,
        _: Rectangle,
        _: mouse::Cursor,
    ) -> Option<Action<Message>> {
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
        let (scale, step) = self.scale(bounds.size());
        let base = self.cache.draw(renderer, bounds.size(), |frame| {
            draw_value_grid(frame, &scale, step, self.suffix);
            draw_x_ticks(frame, &scale, &self.x_ticks);
            self.draw_lines(frame, &scale);
        });
        let mut frame = Frame::new(renderer, bounds.size());
        if let Some(pos) = cursor.position_in(bounds)
            && let Some(x) = nearest_x(self.xs(), &scale, pos.x)
        {
            let screen_x = scale.x(x);
            frame.stroke(
                &Path::line(
                    Point::new(screen_x, PAD_TOP),
                    Point::new(screen_x, scale.bottom()),
                ),
                Stroke::default()
                    .with_color(style::TEXT_TERTIARY)
                    .with_width(1.0),
            );
            let mut lines = vec![(self.x_label)(x)];
            for series in &self.series {
                if let Some((_, y)) = series.points.iter().find(|p| p.0.total_cmp(&x).is_eq()) {
                    lines.push(format!("{}: {y:.1}{}", series.name, self.suffix));
                }
            }
            draw_tooltip(
                &mut frame,
                bounds.size(),
                Point::new(screen_x, pos.y),
                &lines,
            );
        }
        vec![base, frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use iced::Size;

    use super::*;

    #[test]
    fn test_nearest_x_picks_the_closest_on_screen() {
        let scale = Scale::new(Size::new(162.0, 100.0), (0.0, 10.0), 1.0);
        // 100 px wide plot from x = 48: 10 px per unit.
        let xs = [0.0, 4.0, 10.0];
        assert_eq!(nearest_x(xs.into_iter(), &scale, 48.0 + 31.0), Some(4.0));
        assert_eq!(nearest_x(xs.into_iter(), &scale, 500.0), Some(10.0));
        assert_eq!(nearest_x(std::iter::empty(), &scale, 50.0), None);
    }
}
