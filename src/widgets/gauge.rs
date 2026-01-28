use iced::{
    Color, Rectangle, Renderer, Theme, Vector, mouse,
    widget::canvas::{self, Path, Stroke, Text},
};

use crate::style;

pub struct GaugeWidget<'a> {
    pub percentage: f64,
    pub is_open: bool,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub cache: &'a canvas::Cache,
}

/// Determine the status text based on percentage and thresholds.
pub fn get_status_text(percentage: f64, low_threshold: f64, high_threshold: f64) -> &'static str {
    if percentage < low_threshold {
        "Not Busy"
    } else if percentage < high_threshold {
        "Moderate"
    } else {
        "Crowded"
    }
}

/// Determine the color based on percentage and thresholds.
pub fn get_status_color(percentage: f64, low_threshold: f64, high_threshold: f64) -> Color {
    if percentage < low_threshold {
        style::ACCENT_GREEN
    } else if percentage < high_threshold {
        style::ACCENT_ORANGE
    } else {
        style::ACCENT_RED
    }
}

impl<'a, Message> canvas::Program<Message> for GaugeWidget<'a> {
    type State = ();

    fn draw(
        &self,
        _: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let center = frame.center();
            let radius = bounds.width.min(bounds.height) / 2.0 - 10.0;
            let width = 15.0;

            // Background Arc
            let bg_arc = Path::new(|b| {
                b.arc(canvas::path::Arc {
                    center,
                    radius,
                    start_angle: 0.0.into(),
                    end_angle: 360.0f32.to_radians().into(),
                })
            });
            frame.stroke(
                &bg_arc,
                Stroke::default()
                    .with_color(style::STROKE_DIM)
                    .with_width(width),
            );

            if !self.is_open {
                frame.fill_text(Text {
                    content: "CLOSED".to_string(),
                    position: center,
                    color: style::TEXT_MUTED,
                    size: 32.0.into(),
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });
            } else {
                let color =
                    get_status_color(self.percentage, self.low_threshold, self.high_threshold);

                // Foreground Arc
                let angle = (self.percentage / 100.0 * 360.0).max(1.0);
                let fg_arc = Path::new(|b| {
                    b.arc(canvas::path::Arc {
                        center,
                        radius,
                        start_angle: (-90.0f32).to_radians().into(),
                        end_angle: (angle as f32 - 90.0).to_radians().into(),
                    })
                });
                frame.stroke(
                    &fg_arc,
                    Stroke::default()
                        .with_color(color)
                        .with_width(width)
                        .with_line_cap(canvas::LineCap::Round),
                );

                // Text
                frame.fill_text(Text {
                    content: format!("{:.0}%", self.percentage),
                    position: center + Vector::new(0.0, -5.0),
                    color: style::TEXT_BRIGHT,
                    size: 48.0.into(),
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });

                let status_text =
                    get_status_text(self.percentage, self.low_threshold, self.high_threshold);

                frame.fill_text(Text {
                    content: status_text.into(),
                    position: center + Vector::new(0.0, 30.0),
                    color,
                    size: 14.0.into(),
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });
            }
        });
        vec![geo]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default thresholds for testing (matching typical config values)
    const LOW: f64 = 40.0;
    const HIGH: f64 = 75.0;

    // ==================== get_status_text Tests ====================

    #[test]
    fn test_status_text_below_low_threshold() {
        assert_eq!(get_status_text(0.0, LOW, HIGH), "Not Busy");
        assert_eq!(get_status_text(20.0, LOW, HIGH), "Not Busy");
        assert_eq!(get_status_text(39.9, LOW, HIGH), "Not Busy");
    }

    #[test]
    fn test_status_text_at_low_threshold() {
        // At exactly the threshold, should be "Moderate" (not below)
        assert_eq!(get_status_text(40.0, LOW, HIGH), "Moderate");
    }

    #[test]
    fn test_status_text_between_thresholds() {
        assert_eq!(get_status_text(40.1, LOW, HIGH), "Moderate");
        assert_eq!(get_status_text(50.0, LOW, HIGH), "Moderate");
        assert_eq!(get_status_text(74.9, LOW, HIGH), "Moderate");
    }

    #[test]
    fn test_status_text_at_high_threshold() {
        // At exactly the high threshold, should be "Crowded" (not below)
        assert_eq!(get_status_text(75.0, LOW, HIGH), "Crowded");
    }

    #[test]
    fn test_status_text_above_high_threshold() {
        assert_eq!(get_status_text(75.1, LOW, HIGH), "Crowded");
        assert_eq!(get_status_text(90.0, LOW, HIGH), "Crowded");
        assert_eq!(get_status_text(100.0, LOW, HIGH), "Crowded");
    }

    #[test]
    fn test_status_text_with_custom_thresholds() {
        // Test with different threshold values
        assert_eq!(get_status_text(25.0, 30.0, 60.0), "Not Busy");
        assert_eq!(get_status_text(45.0, 30.0, 60.0), "Moderate");
        assert_eq!(get_status_text(80.0, 30.0, 60.0), "Crowded");
    }

    // ==================== get_status_color Tests ====================

    #[test]
    fn test_color_below_low_threshold() {
        let color = get_status_color(20.0, LOW, HIGH);
        assert_eq!(color, style::ACCENT_GREEN);
    }

    #[test]
    fn test_color_at_low_threshold() {
        let color = get_status_color(40.0, LOW, HIGH);
        assert_eq!(color, style::ACCENT_ORANGE);
    }

    #[test]
    fn test_color_between_thresholds() {
        let color = get_status_color(50.0, LOW, HIGH);
        assert_eq!(color, style::ACCENT_ORANGE);
    }

    #[test]
    fn test_color_at_high_threshold() {
        let color = get_status_color(75.0, LOW, HIGH);
        assert_eq!(color, style::ACCENT_RED);
    }

    #[test]
    fn test_color_above_high_threshold() {
        let color = get_status_color(100.0, LOW, HIGH);
        assert_eq!(color, style::ACCENT_RED);
    }

    #[test]
    fn test_color_consistency_with_status_text() {
        // Ensure color and text are consistent
        let test_values = [0.0, 20.0, 39.9, 40.0, 50.0, 74.9, 75.0, 100.0];

        for &val in &test_values {
            let text = get_status_text(val, LOW, HIGH);
            let color = get_status_color(val, LOW, HIGH);

            match text {
                "Not Busy" => assert_eq!(color, style::ACCENT_GREEN),
                "Moderate" => assert_eq!(color, style::ACCENT_ORANGE),
                "Crowded" => assert_eq!(color, style::ACCENT_RED),
                _ => panic!("Unexpected status text: {}", text),
            }
        }
    }
}
