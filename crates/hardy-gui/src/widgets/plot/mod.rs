//! Small charts for the forecast performance page: lines over a numeric
//! x axis, and a scatter of points against a diagonal. Both share the
//! value scale, grid and hover tooltip drawn here.

mod line;
mod scatter;

use iced::{
    Point, Size,
    alignment::{Horizontal, Vertical},
    widget::canvas::{Frame, Path, Stroke, Text},
};

pub use self::{
    line::{LineChart, Series},
    scatter::{Dot, ScatterPlot},
};
use crate::style;

pub(super) const PAD_LEFT: f32 = 48.0;
pub(super) const PAD_RIGHT: f32 = 14.0;
pub(super) const PAD_TOP: f32 = 12.0;
pub(super) const PAD_BOTTOM: f32 = 26.0;

/// Ticks on a value axis.
const TICKS: f64 = 4.0;

/// A rounded axis maximum at or above `max`, and its tick step: steps of
/// 1, 2 or 5 times a power of ten, at most four of them.
pub fn nice_scale(max: f64) -> (f64, f64) {
    if !max.is_finite() || max <= 0.0 {
        return (1.0, 0.25);
    }
    let rough = max / TICKS;
    let magnitude = 10_f64.powf(rough.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|f| f * magnitude)
        .find(|s| *s >= rough)
        .unwrap_or(10.0 * magnitude);
    ((max / step).ceil() * step, step)
}

/// The value axis for values from `min` to `max`: from zero up, or
/// symmetric around zero when any value is negative; with its tick step.
pub fn value_range(min: f64, max: f64) -> (f64, f64, f64) {
    if min < 0.0 {
        let reach = max.max(-min);
        let (_, step) = nice_scale(2.0 * reach);
        let top = (reach / step).ceil() * step;
        (-top, top, step)
    } else {
        let (top, step) = nice_scale(max);
        (0.0, top, step)
    }
}

/// Tick values from `from` to `to` in `step`s, computed by count so that
/// zero is exactly zero.
pub fn ticks(from: f64, to: f64, step: f64) -> Vec<f64> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let count = ((to - from) / step + 1e-6).floor().max(0.0) as usize;
    (0..=count)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let value = from + i as f64 * step;
            if value.abs() < step * 1e-6 {
                0.0
            } else {
                value
            }
        })
        .collect()
}

/// A tick label: whole numbers without decimals.
pub fn tick_label(value: f64, step: f64, suffix: &str) -> String {
    if step >= 1.0 {
        format!("{value:.0}{suffix}")
    } else {
        format!("{value:.1}{suffix}")
    }
}

/// Maps values to a plot area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Scale {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub width: f32,
    pub height: f32,
}

impl Scale {
    pub fn new(size: Size, x: (f64, f64), y: (f64, f64)) -> Self {
        Self {
            x_min: x.0,
            x_max: if x.1 > x.0 { x.1 } else { x.0 + 1.0 },
            y_min: y.0,
            y_max: if y.1 > y.0 { y.1 } else { y.0 + f64::EPSILON },
            width: (size.width - PAD_LEFT - PAD_RIGHT).max(1.0),
            height: (size.height - PAD_TOP - PAD_BOTTOM).max(1.0),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn x(&self, x: f64) -> f32 {
        PAD_LEFT + ((x - self.x_min) / (self.x_max - self.x_min)) as f32 * self.width
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn y(&self, y: f64) -> f32 {
        let ratio = (y.clamp(self.y_min, self.y_max) - self.y_min) / (self.y_max - self.y_min);
        PAD_TOP + self.height - ratio as f32 * self.height
    }

    pub fn point(&self, x: f64, y: f64) -> Point {
        Point::new(self.x(x), self.y(y))
    }

    pub fn bottom(&self) -> f32 {
        PAD_TOP + self.height
    }
}

fn caption(content: String, position: Point, align_x: Horizontal, align_y: Vertical) -> Text {
    Text {
        content,
        position,
        color: style::TEXT_TERTIARY,
        size: style::TEXT_CAPTION.into(),
        align_x: align_x.into(),
        align_y,
        ..Default::default()
    }
}

/// Horizontal grid lines with value labels on the left.
pub(super) fn draw_value_grid(frame: &mut Frame, scale: &Scale, step: f64, suffix: &str) {
    for value in ticks(scale.y_min, scale.y_max, step) {
        let y = scale.y(value);
        // Zero stands out on an axis with negative values.
        let color = if value == 0.0 && scale.y_min < 0.0 {
            style::BORDER
        } else {
            style::GRID
        };
        frame.stroke(
            &Path::line(
                Point::new(PAD_LEFT, y),
                Point::new(PAD_LEFT + scale.width, y),
            ),
            Stroke::default().with_color(color).with_width(1.0),
        );
        frame.fill_text(caption(
            tick_label(value, step, suffix),
            Point::new(PAD_LEFT - 8.0, y),
            Horizontal::Right,
            Vertical::Center,
        ));
    }
}

/// Labels under the plot at the given x values.
pub(super) fn draw_x_ticks(frame: &mut Frame, scale: &Scale, ticks: &[(f64, String)]) {
    for (x, label) in ticks {
        frame.fill_text(caption(
            label.clone(),
            Point::new(scale.x(*x), scale.bottom() + 8.0),
            Horizontal::Center,
            Vertical::Top,
        ));
    }
}

/// A tooltip box with one line per entry, beside `anchor` and inside
/// `bounds`.
pub(super) fn draw_tooltip(frame: &mut Frame, bounds: Size, anchor: Point, lines: &[String]) {
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let size = Size::new(
        20.0 + longest as f32 * 6.6,
        12.0 + lines.len() as f32 * 16.0,
    );
    let x = if anchor.x + 12.0 + size.width > bounds.width {
        anchor.x - 12.0 - size.width
    } else {
        anchor.x + 12.0
    };
    let y = (anchor.y - size.height / 2.0).clamp(0.0, (bounds.height - size.height).max(0.0));
    let tooltip = Path::rounded_rectangle(Point::new(x, y), size, 8.0.into());
    frame.fill(&tooltip, style::BG_ELEVATED);
    frame.stroke(
        &tooltip,
        Stroke::default().with_color(style::BORDER).with_width(1.0),
    );
    for (i, line) in lines.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let line_y = y + 14.0 + i as f32 * 16.0;
        frame.fill_text(Text {
            content: line.clone(),
            position: Point::new(x + 10.0, line_y),
            color: style::TEXT_PRIMARY,
            size: style::TEXT_CAPTION.into(),
            align_y: Vertical::Center,
            ..Default::default()
        });
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_nice_scale_rounds_up_to_simple_steps() {
        assert_eq!(nice_scale(7.3), (8.0, 2.0));
        assert_eq!(nice_scale(10.0), (10.0, 5.0));
        assert_eq!(nice_scale(0.9), (1.0, 0.5));
        assert_eq!(nice_scale(0.0), (1.0, 0.25));
        assert_eq!(nice_scale(f64::NAN), (1.0, 0.25));
    }

    #[test]
    fn test_tick_labels_drop_decimals_for_whole_steps() {
        assert_eq!(tick_label(4.0, 2.0, " pts"), "4 pts");
        assert_eq!(tick_label(0.5, 0.5, ""), "0.5");
    }

    #[test]
    fn test_value_range_is_symmetric_around_zero_for_negative_values() {
        assert_eq!(value_range(0.0, 7.3), (0.0, 8.0, 2.0));
        // Half as many steps each side of zero.
        assert_eq!(value_range(-3.0, 1.0), (-4.0, 4.0, 2.0));
        assert_eq!(value_range(-1.0, 7.3), (-10.0, 10.0, 5.0));
        let labels: Vec<String> = ticks(-0.8, 0.8, 0.4)
            .into_iter()
            .map(|v| tick_label(v, 0.4, ""))
            .collect();
        assert_eq!(labels, ["-0.8", "-0.4", "0.0", "0.4", "0.8"]);
        let signed = Scale::new(Size::new(200.0, 100.0), (0.0, 10.0), (-4.0, 4.0));
        assert_relative_eq!(signed.y(0.0), PAD_TOP + signed.height / 2.0);
        assert_relative_eq!(signed.y(-4.0), signed.bottom());
    }

    #[test]
    fn test_scale_maps_corners() {
        let scale = Scale::new(Size::new(200.0, 100.0), (0.0, 10.0), (0.0, 20.0));
        assert_relative_eq!(scale.x(0.0), PAD_LEFT);
        assert_relative_eq!(scale.x(10.0), 200.0 - PAD_RIGHT);
        assert_relative_eq!(scale.y(0.0), scale.bottom());
        assert_relative_eq!(scale.y(20.0), PAD_TOP);
        // Out-of-range values stay on the plot.
        assert_relative_eq!(scale.y(50.0), PAD_TOP);
        // A single x value still maps.
        let flat = Scale::new(Size::new(200.0, 100.0), (3.0, 3.0), (0.0, 1.0));
        assert_relative_eq!(flat.x(3.0), PAD_LEFT);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn nice_scale_covers_max_with_few_ticks(max in 0.001f64..10_000.0) {
            let (top, step) = nice_scale(max);
            prop_assert!(top >= max);
            prop_assert!(top / step <= TICKS + 1e-9);
            prop_assert!(top < max * 2.5 + step);
        }
    }
}
