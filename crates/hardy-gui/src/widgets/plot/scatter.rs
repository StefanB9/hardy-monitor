//! Points on equal axes with the diagonal where both values agree, and the
//! nearest point's label on hover.

use iced::{
    Color, Point, Rectangle, Renderer, Size, Theme,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, Action, Frame, Path, Stroke},
};

use super::{
    PAD_LEFT, PAD_TOP, Scale, caption, draw_tooltip, draw_value_grid, draw_x_ticks, nice_scale,
    tick_label, ticks,
};
use crate::{app::Message, style};

/// Hover reach around a point, in pixels.
const REACH: f32 = 10.0;

/// One point.
#[derive(Debug, Clone, PartialEq)]
pub struct Dot {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub color: Color,
    /// Drawn larger (e.g. a model in use).
    pub large: bool,
}

/// A scatter plot with the same scale on both axes.
pub struct ScatterPlot<'a> {
    pub dots: Vec<Dot>,
    pub x_title: &'static str,
    pub y_title: &'static str,
    pub suffix: &'static str,
    pub cache: &'a canvas::Cache,
}

/// The dot within reach of `cursor`, nearest first.
pub(super) fn nearest_dot<'d>(dots: &'d [Dot], scale: &Scale, cursor: Point) -> Option<&'d Dot> {
    dots.iter()
        .map(|d| (d, scale.point(d.x, d.y).distance(cursor)))
        .filter(|(_, distance)| *distance <= REACH)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(d, _)| d)
}

impl ScatterPlot<'_> {
    fn scale(&self, size: Size) -> (Scale, f64) {
        let max = self
            .dots
            .iter()
            .flat_map(|d| [d.x, d.y])
            .fold(0.0, f64::max);
        let (top, step) = nice_scale(max);
        (Scale::new(size, (0.0, top), (0.0, top)), step)
    }
}

impl canvas::Program<Message> for ScatterPlot<'_> {
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
            draw_value_grid(frame, &scale, step, "");
            let x_ticks: Vec<(f64, String)> = ticks(0.0, scale.x_max, step)
                .into_iter()
                .map(|v| (v, tick_label(v, step, "")))
                .collect();
            draw_x_ticks(frame, &scale, &x_ticks);
            frame.stroke(
                &Path::line(scale.point(0.0, 0.0), scale.point(scale.x_max, scale.y_max)),
                Stroke::default().with_color(style::BORDER).with_width(1.0),
            );
            frame.fill_text(caption(
                self.y_title.to_string(),
                Point::new(PAD_LEFT + 6.0, PAD_TOP),
                Horizontal::Left,
                Vertical::Top,
            ));
            frame.fill_text(caption(
                self.x_title.to_string(),
                Point::new(PAD_LEFT + scale.width, scale.bottom() - 6.0),
                Horizontal::Right,
                Vertical::Bottom,
            ));
            for dot in &self.dots {
                let radius = if dot.large { 5.5 } else { 3.5 };
                frame.fill(&Path::circle(scale.point(dot.x, dot.y), radius), dot.color);
            }
        });
        let mut frame = Frame::new(renderer, bounds.size());
        if let Some(pos) = cursor.position_in(bounds)
            && let Some(dot) = nearest_dot(&self.dots, &scale, pos)
        {
            let at = scale.point(dot.x, dot.y);
            frame.stroke(
                &Path::circle(at, 7.0),
                Stroke::default()
                    .with_color(style::TEXT_PRIMARY)
                    .with_width(1.5),
            );
            let lines = [
                dot.label.clone(),
                format!("{}: {:.1}{}", self.x_title, dot.x, self.suffix),
                format!("{}: {:.1}{}", self.y_title, dot.y, self.suffix),
            ];
            draw_tooltip(&mut frame, bounds.size(), at, &lines);
        }
        vec![base, frame.into_geometry()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(x: f64, y: f64, label: &str) -> Dot {
        Dot {
            x,
            y,
            label: label.to_string(),
            color: style::ACCENT,
            large: false,
        }
    }

    #[test]
    fn test_nearest_dot_only_within_reach() {
        let scale = Scale::new(Size::new(162.0, 138.0), (0.0, 10.0), (0.0, 10.0));
        let dots = [dot(2.0, 2.0, "a"), dot(3.0, 3.0, "b")];
        let near_b = scale.point(3.0, 3.0) + iced::Vector::new(2.0, 1.0);
        assert_eq!(
            nearest_dot(&dots, &scale, near_b).map(|d| d.label.as_str()),
            Some("b")
        );
        let far = scale.point(8.0, 8.0);
        assert_eq!(nearest_dot(&dots, &scale, far), None);
    }
}
