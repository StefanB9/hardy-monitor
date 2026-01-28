use chrono::{Local, Offset};
use iced::{
    Color, Point, Rectangle, Renderer, Size, Theme, mouse,
    widget::canvas::{self, Path, Stroke, Text},
};

use crate::{db::HourlyAverage, style};

pub struct HeatmapWidget<'a> {
    pub data: &'a [HourlyAverage],
    pub cache: &'a canvas::Cache,
    pub tooltip_cache: &'a canvas::Cache, // Add this
}

impl<'a, Message> canvas::Program<Message> for HeatmapWidget<'a> {
    type State = ();

    fn draw(
        &self,
        _: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let pad_left = 30.0;
        let pad_bottom = 20.0;
        let w = bounds.width - pad_left;
        let h = bounds.height - pad_bottom;
        let days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        let cell_w = w / 24.0;
        let cell_h = h / 7.0;

        // 1. Draw the Heatmap Grid (Cached)
        let grid_geo = self.cache.draw(renderer, bounds.size(), |frame| {
            // Get current offset to map Local Grid -> UTC Data
            let offset_seconds = Local::now().offset().fix().local_minus_utc();
            let seconds_per_week = 7 * 24 * 3600;

            for (d_idx, day) in days.iter().enumerate() {
                // Day Label
                frame.fill_text(Text {
                    content: day.to_string(),
                    position: Point::new(0.0, d_idx as f32 * cell_h + cell_h / 2.0),
                    color: style::TEXT_MUTED,
                    size: 10.0.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });

                for hour in 0..24 {
                    let x = pad_left + hour as f32 * cell_w;
                    let y = d_idx as f32 * cell_h;

                    let is_open_hour = if d_idx >= 5 {
                        (9..21).contains(&hour)
                    } else {
                        (6..23).contains(&hour)
                    };

                    let bg = Path::rounded_rectangle(
                        Point::new(x + 1.0, y + 1.0),
                        Size::new(cell_w - 2.0, cell_h - 2.0),
                        3.0.into(),
                    );

                    if !is_open_hour {
                        frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.3));
                    } else {
                        // Calculate UTC indices
                        let local_seconds = (d_idx as i64 * 24 + hour as i64) * 3600;
                        let utc_seconds = local_seconds - offset_seconds as i64;
                        let wrapped_utc = ((utc_seconds % seconds_per_week) + seconds_per_week)
                            % seconds_per_week;

                        let target_w = (wrapped_utc / 3600) / 24;
                        let target_h = (wrapped_utc / 3600) % 24;

                        let val = self
                            .data
                            .iter()
                            .find(|x| x.weekday == target_w as i32 && x.hour == target_h as i32)
                            .map(|x| x.avg_percentage)
                            .unwrap_or(0.0);

                        // Gradient Logic
                        let color = if val == 0.0 {
                            style::BG_DARK
                        } else {
                            calculate_gradient_color(val)
                        };

                        frame.fill(&bg, color);
                    }

                    // Hour Labels (Bottom)
                    if d_idx == 6 && hour % 4 == 0 {
                        frame.fill_text(Text {
                            content: format!("{:02}", hour),
                            position: Point::new(x + cell_w / 2.0, h + 10.0),
                            color: style::TEXT_MUTED,
                            size: 10.0.into(),
                            align_x: iced::alignment::Horizontal::Center.into(),
                            align_y: iced::alignment::Vertical::Center,
                            ..Default::default()
                        });
                    }
                }
            }
        });

        // 2. Draw Tooltip (Dynamic - Uses separate cache, cleared every frame)
        // We use a cache here because creating Geometry directly is not exposed easily.
        // Clearing it ensures it updates with the cursor.
        self.tooltip_cache.clear();

        let overlay_geo = self.tooltip_cache.draw(renderer, bounds.size(), |frame| {
            if let Some(cursor_pos) = cursor.position_in(bounds) {
                // Check if inside grid area
                if cursor_pos.x > pad_left && cursor_pos.y < h {
                    let col = ((cursor_pos.x - pad_left) / cell_w).floor() as i64;
                    let row = (cursor_pos.y / cell_h).floor() as i64;

                    if col >= 0 && col < 24 && row >= 0 && row < 7 {
                        // Resolve value again for tooltip
                        let offset_seconds = Local::now().offset().fix().local_minus_utc();
                        let seconds_per_week = 7 * 24 * 3600;

                        let local_seconds = (row * 24 + col) * 3600;
                        let utc_seconds = local_seconds - offset_seconds as i64;
                        let wrapped_utc = ((utc_seconds % seconds_per_week) + seconds_per_week)
                            % seconds_per_week;

                        let target_w = (wrapped_utc / 3600) / 24;
                        let target_h = (wrapped_utc / 3600) % 24;

                        let val = self
                            .data
                            .iter()
                            .find(|x| x.weekday == target_w as i32 && x.hour == target_h as i32)
                            .map(|x| x.avg_percentage);

                        if let Some(v) = val {
                            let text = format!("{:.1}%", v);
                            let pos = Point::new(cursor_pos.x + 10.0, cursor_pos.y - 20.0);

                            // Background for tooltip
                            let tooltip_bg =
                                Path::rounded_rectangle(pos, Size::new(50.0, 24.0), 4.0.into());
                            frame.fill(&tooltip_bg, style::BG_CARD);
                            frame.stroke(
                                &tooltip_bg,
                                Stroke::default()
                                    .with_color(style::STROKE_DIM)
                                    .with_width(1.0),
                            );

                            frame.fill_text(Text {
                                content: text,
                                position: Point::new(pos.x + 25.0, pos.y + 12.0),
                                color: style::TEXT_BRIGHT,
                                size: 12.0.into(),
                                align_x: iced::alignment::Horizontal::Center.into(),
                                align_y: iced::alignment::Vertical::Center,
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        });

        vec![grid_geo, overlay_geo]
    }
}

fn calculate_gradient_color(percentage: f64) -> Color {
    // 0% -> Green, 50% -> Yellow, 100% -> Red
    let p = percentage.clamp(0.0, 100.0) / 100.0;

    let low = Color::from_rgb(0.2, 0.8, 0.2); // Green
    let mid = Color::from_rgb(0.9, 0.9, 0.2); // Yellow
    let high = Color::from_rgb(0.9, 0.2, 0.2); // Red

    if p < 0.5 {
        let factor = p * 2.0;
        interpolate_color(low, mid, factor as f32)
    } else {
        let factor = (p - 0.5) * 2.0;
        interpolate_color(mid, high, factor as f32)
    }
}

fn interpolate_color(c1: Color, c2: Color, factor: f32) -> Color {
    Color::from_rgb(
        c1.r + (c2.r - c1.r) * factor,
        c1.g + (c2.g - c1.g) * factor,
        c1.b + (c2.b - c1.b) * factor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== interpolate_color Tests ====================

    #[test]
    fn test_interpolate_color_at_zero_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0); // Red
        let c2 = Color::from_rgb(0.0, 1.0, 0.0); // Green
        let result = interpolate_color(c1, c2, 0.0);

        assert!((result.r - 1.0).abs() < 0.001);
        assert!((result.g - 0.0).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_at_one_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0); // Red
        let c2 = Color::from_rgb(0.0, 1.0, 0.0); // Green
        let result = interpolate_color(c1, c2, 1.0);

        assert!((result.r - 0.0).abs() < 0.001);
        assert!((result.g - 1.0).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_at_half_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0); // Red
        let c2 = Color::from_rgb(0.0, 1.0, 0.0); // Green
        let result = interpolate_color(c1, c2, 0.5);

        assert!((result.r - 0.5).abs() < 0.001);
        assert!((result.g - 0.5).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_with_blue_channel() {
        let c1 = Color::from_rgb(0.0, 0.0, 0.0); // Black
        let c2 = Color::from_rgb(1.0, 1.0, 1.0); // White
        let result = interpolate_color(c1, c2, 0.25);

        assert!((result.r - 0.25).abs() < 0.001);
        assert!((result.g - 0.25).abs() < 0.001);
        assert!((result.b - 0.25).abs() < 0.001);
    }

    // ==================== calculate_gradient_color Tests ====================

    #[test]
    fn test_gradient_color_at_zero_percent() {
        let result = calculate_gradient_color(0.0);
        // At 0%, should be green (0.2, 0.8, 0.2)
        assert!((result.r - 0.2).abs() < 0.01);
        assert!((result.g - 0.8).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_fifty_percent() {
        let result = calculate_gradient_color(50.0);
        // At 50%, should be yellow (0.9, 0.9, 0.2)
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.9).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_hundred_percent() {
        let result = calculate_gradient_color(100.0);
        // At 100%, should be red (0.9, 0.2, 0.2)
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.2).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_twenty_five_percent() {
        let result = calculate_gradient_color(25.0);
        // At 25%, should be between green and yellow
        // factor = 0.25 * 2 = 0.5
        // Expected: interpolate(green, yellow, 0.5)
        assert!(result.r > 0.2 && result.r < 0.9);
        assert!(result.g > 0.8 && result.g < 0.9);
    }

    #[test]
    fn test_gradient_color_at_seventy_five_percent() {
        let result = calculate_gradient_color(75.0);
        // At 75%, should be between yellow and red
        // factor = (0.75 - 0.5) * 2 = 0.5
        assert!(result.r > 0.85); // Close to red
        assert!(result.g > 0.2 && result.g < 0.9); // Between yellow and red
    }

    #[test]
    fn test_gradient_color_clamps_above_hundred() {
        let result = calculate_gradient_color(150.0);
        // Should clamp to 100% (red)
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_clamps_below_zero() {
        let result = calculate_gradient_color(-50.0);
        // Should clamp to 0% (green)
        assert!((result.r - 0.2).abs() < 0.01);
        assert!((result.g - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_handles_boundary_values() {
        // Test values just above and below 50%
        let below = calculate_gradient_color(49.9);
        let above = calculate_gradient_color(50.1);

        // Both should be very close to yellow
        assert!((below.r - above.r).abs() < 0.05);
        assert!((below.g - above.g).abs() < 0.05);
    }
}
