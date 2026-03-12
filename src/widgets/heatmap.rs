use chrono::{Local, Offset};
use iced::{
    Color, Point, Rectangle, Renderer, Size, Theme, mouse,
    widget::canvas::{self, Path, Stroke, Text},
};

use crate::{db::HourlyAverage, style};

pub struct HeatmapWidget<'a> {
    pub data: &'a [HourlyAverage],
    pub cache: &'a canvas::Cache,
    pub tooltip_cache: &'a canvas::Cache,
}

impl<Message> canvas::Program<Message> for HeatmapWidget<'_> {
    type State = ();

    #[allow(clippy::too_many_lines)]
    fn draw(
        &self,
        (): &Self::State,
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

        let grid_geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let offset_seconds = Local::now().offset().fix().local_minus_utc();
            let seconds_per_week = 7 * 24 * 3600;

            for (d_idx, day) in days.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let label_y = d_idx as f32 * cell_h + cell_h / 2.0;

                frame.fill_text(Text {
                    content: day.to_string(),
                    position: Point::new(0.0, label_y),
                    color: style::TEXT_MUTED,
                    size: 10.0.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });

                for hour in 0..24 {
                    #[allow(clippy::cast_precision_loss)]
                    let x = pad_left + hour as f32 * cell_w;

                    #[allow(clippy::cast_precision_loss)]
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

                    if is_open_hour {
                        let d_idx_i64 = i64::try_from(d_idx).unwrap_or_default();
                        let local_seconds = (d_idx_i64 * 24 + i64::from(hour)) * 3600;
                        let utc_seconds = local_seconds - i64::from(offset_seconds);
                        let wrapped_utc = ((utc_seconds % seconds_per_week) + seconds_per_week)
                            % seconds_per_week;

                        let target_w = i32::try_from((wrapped_utc / 3600) / 24).unwrap_or_default();
                        let target_h = i32::try_from((wrapped_utc / 3600) % 24).unwrap_or_default();

                        let val = self
                            .data
                            .iter()
                            .find(|x| x.weekday == target_w && x.hour == target_h)
                            .map_or(0.0, |x| x.avg_percentage);

                        let color = if val == 0.0 {
                            style::BG_DARK
                        } else {
                            calculate_gradient_color(val)
                        };

                        frame.fill(&bg, color);
                    } else {
                        frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.3));
                    }

                    if d_idx == 6 && hour % 4 == 0 {
                        frame.fill_text(Text {
                            content: format!("{hour:02}"),
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

        self.tooltip_cache.clear();

        let overlay_geo = self.tooltip_cache.draw(renderer, bounds.size(), |frame| {
            if let Some(cursor_pos) = cursor.position_in(bounds)
                && cursor_pos.x > pad_left
                && cursor_pos.y < h
            {
                #[allow(clippy::cast_possible_truncation)]
                let col = ((cursor_pos.x - pad_left) / cell_w).floor() as i64;

                #[allow(clippy::cast_possible_truncation)]
                let row = (cursor_pos.y / cell_h).floor() as i64;

                if (0..24).contains(&col) && (0..7).contains(&row) {
                    let offset_seconds = Local::now().offset().fix().local_minus_utc();
                    let seconds_per_week = 7 * 24 * 3600;

                    let local_seconds = (row * 24 + col) * 3600;
                    let utc_seconds = local_seconds - i64::from(offset_seconds);
                    let wrapped_utc =
                        ((utc_seconds % seconds_per_week) + seconds_per_week) % seconds_per_week;

                    let target_w = i32::try_from((wrapped_utc / 3600) / 24).unwrap_or_default();
                    let target_h = i32::try_from((wrapped_utc / 3600) % 24).unwrap_or_default();

                    let val = self
                        .data
                        .iter()
                        .find(|x| x.weekday == target_w && x.hour == target_h)
                        .map(|x| x.avg_percentage);

                    if let Some(v) = val {
                        let text = format!("{v:.1}%");
                        let pos = Point::new(cursor_pos.x + 10.0, cursor_pos.y - 20.0);

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
        });

        vec![grid_geo, overlay_geo]
    }
}

fn calculate_gradient_color(percentage: f64) -> Color {
    let p = percentage.clamp(0.0, 100.0) / 100.0;

    let low = Color::from_rgb(0.2, 0.8, 0.2);
    let mid = Color::from_rgb(0.9, 0.9, 0.2);
    let high = Color::from_rgb(0.9, 0.2, 0.2);

    if p < 0.5 {
        #[allow(clippy::cast_possible_truncation)]
        let factor = (p * 2.0) as f32;
        interpolate_color(low, mid, factor)
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let factor = ((p - 0.5) * 2.0) as f32;
        interpolate_color(mid, high, factor)
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

    #[test]
    fn test_interpolate_color_at_zero_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0);
        let c2 = Color::from_rgb(0.0, 1.0, 0.0);
        let result = interpolate_color(c1, c2, 0.0);

        assert!((result.r - 1.0).abs() < 0.001);
        assert!((result.g - 0.0).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_at_one_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0);
        let c2 = Color::from_rgb(0.0, 1.0, 0.0);
        let result = interpolate_color(c1, c2, 1.0);

        assert!((result.r - 0.0).abs() < 0.001);
        assert!((result.g - 1.0).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_at_half_factor() {
        let c1 = Color::from_rgb(1.0, 0.0, 0.0);
        let c2 = Color::from_rgb(0.0, 1.0, 0.0);
        let result = interpolate_color(c1, c2, 0.5);

        assert!((result.r - 0.5).abs() < 0.001);
        assert!((result.g - 0.5).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_color_with_blue_channel() {
        let c1 = Color::from_rgb(0.0, 0.0, 0.0);
        let c2 = Color::from_rgb(1.0, 1.0, 1.0);
        let result = interpolate_color(c1, c2, 0.25);

        assert!((result.r - 0.25).abs() < 0.001);
        assert!((result.g - 0.25).abs() < 0.001);
        assert!((result.b - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_gradient_color_at_zero_percent() {
        let result = calculate_gradient_color(0.0);
        assert!((result.r - 0.2).abs() < 0.01);
        assert!((result.g - 0.8).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_fifty_percent() {
        let result = calculate_gradient_color(50.0);
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.9).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_hundred_percent() {
        let result = calculate_gradient_color(100.0);
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.2).abs() < 0.01);
        assert!((result.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_at_twenty_five_percent() {
        let result = calculate_gradient_color(25.0);
        assert!(result.r > 0.2 && result.r < 0.9);
        assert!(result.g > 0.8 && result.g < 0.9);
    }

    #[test]
    fn test_gradient_color_at_seventy_five_percent() {
        let result = calculate_gradient_color(75.0);
        assert!(result.r > 0.85);
        assert!(result.g > 0.2 && result.g < 0.9);
    }

    #[test]
    fn test_gradient_color_clamps_above_hundred() {
        let result = calculate_gradient_color(150.0);
        assert!((result.r - 0.9).abs() < 0.01);
        assert!((result.g - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_clamps_below_zero() {
        let result = calculate_gradient_color(-50.0);
        assert!((result.r - 0.2).abs() < 0.01);
        assert!((result.g - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_handles_boundary_values() {
        let below = calculate_gradient_color(49.9);
        let above = calculate_gradient_color(50.1);

        assert!((below.r - above.r).abs() < 0.05);
        assert!((below.g - above.g).abs() < 0.05);
    }
}
