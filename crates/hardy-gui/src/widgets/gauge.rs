//! Live occupancy as a 270° arc on the occupancy colour scale.

use iced::{
    Point, Radians, Rectangle, Renderer, Theme, Vector,
    alignment::{Horizontal, Vertical},
    mouse,
    widget::canvas::{self, LineCap, Path, Stroke, Text, path::Arc},
};

use crate::style::{self, OccupancyLevel};

/// Where the arc starts (bottom left), clockwise from +x in degrees.
const START_DEG: f32 = 135.0;
const SWEEP_DEG: f32 = 270.0;
const ARC_WIDTH: f32 = 14.0;

/// Gauge for the current reading; `percentage` is `None` while closed or
/// before the first reading.
pub struct GaugeWidget<'a> {
    pub percentage: Option<f64>,
    pub is_open: bool,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub cache: &'a canvas::Cache,
}

/// Angle on the arc for `percentage`, in radians.
fn angle_for(percentage: f64) -> Radians {
    #[allow(clippy::cast_possible_truncation)]
    let fraction = (percentage.clamp(0.0, 100.0) / 100.0) as f32;
    Radians((START_DEG + SWEEP_DEG * fraction).to_radians())
}

fn arc(center: Point, radius: f32, from: f64, to: f64) -> Path {
    Path::new(|b| {
        b.arc(Arc {
            center,
            radius,
            start_angle: angle_for(from),
            end_angle: angle_for(to),
        });
    })
}

impl<Message> canvas::Program<Message> for GaugeWidget<'_> {
    type State = ();

    fn draw(
        &self,
        (): &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let center = frame.center() + Vector::new(0.0, 8.0);
            let radius = bounds.width.min(bounds.height) / 2.0 - ARC_WIDTH;
            let stroke = |color| {
                Stroke::default()
                    .with_color(color)
                    .with_width(ARC_WIDTH)
                    .with_line_cap(LineCap::Round)
            };
            frame.stroke(&arc(center, radius, 0.0, 100.0), stroke(style::BG_ELEVATED));

            // Category boundaries as small notches outside the track.
            for threshold in [self.low_threshold, self.high_threshold] {
                let Radians(a) = angle_for(threshold);
                let dir = Vector::new(a.cos(), a.sin());
                let inner = center + dir * (radius + ARC_WIDTH / 2.0 + 3.0);
                let outer = center + dir * (radius + ARC_WIDTH / 2.0 + 9.0);
                frame.stroke(
                    &Path::line(inner, outer),
                    Stroke::default()
                        .with_color(style::TEXT_TERTIARY)
                        .with_width(2.0),
                );
            }

            let (value, caption, caption_color) = match (self.is_open, self.percentage) {
                (true, Some(p)) => {
                    let color = style::occupancy_color(p, self.low_threshold, self.high_threshold);
                    frame.stroke(&arc(center, radius, 0.0, p.max(0.5)), stroke(color));
                    let level =
                        OccupancyLevel::from_percentage(p, self.low_threshold, self.high_threshold);
                    (format!("{p:.0}%"), level.label(), level.color())
                }
                (true, None) => ("–".to_string(), "Waiting for data", style::TEXT_TERTIARY),
                (false, _) => ("Closed".to_string(), "", style::TEXT_SECONDARY),
            };

            let value_size = if self.is_open {
                style::TEXT_DISPLAY
            } else {
                style::TEXT_TITLE
            };
            frame.fill_text(Text {
                content: value,
                position: center - Vector::new(0.0, 6.0),
                color: style::TEXT_PRIMARY,
                size: value_size.into(),
                align_x: Horizontal::Center.into(),
                align_y: Vertical::Center,
                ..Default::default()
            });
            frame.fill_text(Text {
                content: caption.to_string(),
                position: center + Vector::new(0.0, value_size / 2.0 + 6.0),
                color: caption_color,
                size: style::TEXT_BODY.into(),
                align_x: Horizontal::Center.into(),
                align_y: Vertical::Center,
                ..Default::default()
            });
        });
        vec![geo]
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_angle_for_spans_the_arc() {
        assert_relative_eq!(angle_for(0.0).0, START_DEG.to_radians());
        assert_relative_eq!(
            angle_for(50.0).0,
            (START_DEG + SWEEP_DEG / 2.0).to_radians()
        );
        assert_relative_eq!(angle_for(100.0).0, (START_DEG + SWEEP_DEG).to_radians());
    }

    #[test]
    fn test_angle_for_clamps_out_of_range_values() {
        assert_relative_eq!(angle_for(-5.0).0, angle_for(0.0).0);
        assert_relative_eq!(angle_for(140.0).0, angle_for(100.0).0);
    }
}
