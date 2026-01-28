use iced::{
    Color, Point, Rectangle, Renderer, Size, Theme, mouse,
    widget::canvas::{self, Path, Stroke, Text},
};

use crate::{analytics::HourlyComparison, style};

pub struct ComparisonChartWidget<'a> {
    pub data: &'a [HourlyComparison],
    pub cache: &'a canvas::Cache,
    pub tooltip_cache: &'a canvas::Cache,
}

impl<'a, Message> canvas::Program<Message> for ComparisonChartWidget<'a> {
    type State = ();

    fn draw(
        &self,
        _: &Self::State,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let pad_left = 50.0;
        let pad_right = 20.0;
        let pad_top = 30.0;
        let pad_bottom = 40.0;

        let chart_width = bounds.width - pad_left - pad_right;
        let chart_height = bounds.height - pad_top - pad_bottom;

        // Draw the main chart (cached)
        let chart_geo = self.cache.draw(renderer, bounds.size(), |frame| {
            if self.data.is_empty() {
                // Draw "Insufficient data" message
                frame.fill_text(Text {
                    content: "Insufficient data for comparison".to_string(),
                    position: Point::new(bounds.width / 2.0, bounds.height / 2.0),
                    color: style::TEXT_MUTED,
                    size: 16.0.into(),
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });
                return;
            }

            // Calculate max value for scaling
            let max_val = self
                .data
                .iter()
                .map(|d| d.current_avg.max(d.previous_avg))
                .fold(0.0f64, |a, b| a.max(b))
                .max(10.0); // Minimum scale of 10%

            let num_bars = self.data.len();
            let group_width = chart_width / num_bars as f32;
            let bar_width = (group_width * 0.35).min(20.0);
            let bar_gap = 2.0;

            // Draw Y-axis labels
            for i in 0..=4 {
                let y_val = (max_val / 4.0) * i as f64;
                let y_pos = pad_top + chart_height - (chart_height * (y_val / max_val) as f32);

                frame.fill_text(Text {
                    content: format!("{:.0}%", y_val),
                    position: Point::new(pad_left - 8.0, y_pos),
                    color: style::TEXT_MUTED,
                    size: 10.0.into(),
                    align_x: iced::alignment::Horizontal::Right.into(),
                    align_y: iced::alignment::Vertical::Center,
                    ..Default::default()
                });

                // Grid line
                let line = Path::line(
                    Point::new(pad_left, y_pos),
                    Point::new(bounds.width - pad_right, y_pos),
                );
                frame.stroke(
                    &line,
                    Stroke::default()
                        .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.1))
                        .with_width(1.0),
                );
            }

            // Draw bars and X-axis labels
            let mut last_drawn_hour: Option<u32> = None;

            for (i, comparison) in self.data.iter().enumerate() {
                let group_x = pad_left + i as f32 * group_width + group_width / 2.0;

                // Previous period bar (muted)
                let prev_height = (comparison.previous_avg / max_val) as f32 * chart_height;
                if prev_height > 0.0 {
                    let prev_bar = Path::rounded_rectangle(
                        Point::new(group_x - bar_width - bar_gap / 2.0, pad_top + chart_height - prev_height),
                        Size::new(bar_width, prev_height),
                        2.0.into(),
                    );
                    frame.fill(&prev_bar, Color::from_rgba(0.5, 0.5, 0.6, 0.5));
                }

                // Current period bar (colored by change)
                let curr_height = (comparison.current_avg / max_val) as f32 * chart_height;
                if curr_height > 0.0 {
                    let color = get_change_color(comparison.percent_change);
                    let curr_bar = Path::rounded_rectangle(
                        Point::new(group_x + bar_gap / 2.0, pad_top + chart_height - curr_height),
                        Size::new(bar_width, curr_height),
                        2.0.into(),
                    );
                    frame.fill(&curr_bar, color);
                }

                // X-axis labels (show hour, avoid clutter)
                let should_draw_label = match last_drawn_hour {
                    None => true,
                    Some(last) => {
                        // Draw if hour changed and enough space
                        comparison.hour != last && (i == 0 || i % 2 == 0 || num_bars < 24)
                    }
                };

                if should_draw_label {
                    frame.fill_text(Text {
                        content: format!("{:02}", comparison.hour),
                        position: Point::new(group_x, bounds.height - pad_bottom + 15.0),
                        color: style::TEXT_MUTED,
                        size: 10.0.into(),
                        align_x: iced::alignment::Horizontal::Center.into(),
                        align_y: iced::alignment::Vertical::Center,
                        ..Default::default()
                    });
                    last_drawn_hour = Some(comparison.hour);
                }
            }

            // Draw legend
            let legend_y = 10.0;
            let legend_x = bounds.width - pad_right - 150.0;

            // Previous period legend
            let prev_box = Path::rounded_rectangle(
                Point::new(legend_x, legend_y),
                Size::new(12.0, 12.0),
                2.0.into(),
            );
            frame.fill(&prev_box, Color::from_rgba(0.5, 0.5, 0.6, 0.5));
            frame.fill_text(Text {
                content: "Previous".to_string(),
                position: Point::new(legend_x + 16.0, legend_y + 6.0),
                color: style::TEXT_MUTED,
                size: 10.0.into(),
                align_y: iced::alignment::Vertical::Center,
                ..Default::default()
            });

            // Current period legend
            let curr_box = Path::rounded_rectangle(
                Point::new(legend_x + 70.0, legend_y),
                Size::new(12.0, 12.0),
                2.0.into(),
            );
            frame.fill(&curr_box, style::ACCENT_BLUE);
            frame.fill_text(Text {
                content: "Current".to_string(),
                position: Point::new(legend_x + 86.0, legend_y + 6.0),
                color: style::TEXT_MUTED,
                size: 10.0.into(),
                align_y: iced::alignment::Vertical::Center,
                ..Default::default()
            });
        });

        // Draw tooltip (dynamic)
        self.tooltip_cache.clear();

        let tooltip_geo = self.tooltip_cache.draw(renderer, bounds.size(), |frame| {
            if self.data.is_empty() {
                return;
            }

            if let Some(cursor_pos) = cursor.position_in(bounds) {
                let num_bars = self.data.len();
                let group_width = chart_width / num_bars as f32;

                // Check if cursor is in chart area
                if cursor_pos.x > pad_left
                    && cursor_pos.x < bounds.width - pad_right
                    && cursor_pos.y > pad_top
                    && cursor_pos.y < bounds.height - pad_bottom
                {
                    let bar_index = ((cursor_pos.x - pad_left) / group_width) as usize;

                    if bar_index < self.data.len() {
                        let comparison = &self.data[bar_index];
                        let days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
                        let day_name = days.get(comparison.weekday as usize).unwrap_or(&"?");

                        let text = format!(
                            "{} {:02}:00\nPrev: {:.1}%\nCurr: {:.1}%\nChange: {:+.1}%",
                            day_name,
                            comparison.hour,
                            comparison.previous_avg,
                            comparison.current_avg,
                            comparison.percent_change
                        );

                        // Calculate tooltip position
                        let tooltip_width = 100.0;
                        let tooltip_height = 70.0;
                        let mut tooltip_x = cursor_pos.x + 15.0;
                        let mut tooltip_y = cursor_pos.y - tooltip_height - 10.0;

                        // Keep tooltip in bounds
                        if tooltip_x + tooltip_width > bounds.width {
                            tooltip_x = cursor_pos.x - tooltip_width - 15.0;
                        }
                        if tooltip_y < 0.0 {
                            tooltip_y = cursor_pos.y + 15.0;
                        }

                        // Background
                        let tooltip_bg = Path::rounded_rectangle(
                            Point::new(tooltip_x, tooltip_y),
                            Size::new(tooltip_width, tooltip_height),
                            6.0.into(),
                        );
                        frame.fill(&tooltip_bg, style::BG_CARD);
                        frame.stroke(
                            &tooltip_bg,
                            Stroke::default()
                                .with_color(style::STROKE_DIM)
                                .with_width(1.0),
                        );

                        // Text lines
                        let lines: Vec<&str> = text.lines().collect();
                        for (i, line) in lines.iter().enumerate() {
                            let color = if i == 0 {
                                style::TEXT_BRIGHT
                            } else if line.contains("Change") {
                                get_change_color(comparison.percent_change)
                            } else {
                                style::TEXT_MUTED
                            };

                            frame.fill_text(Text {
                                content: line.to_string(),
                                position: Point::new(tooltip_x + 8.0, tooltip_y + 12.0 + i as f32 * 14.0),
                                color,
                                size: 11.0.into(),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        });

        vec![chart_geo, tooltip_geo]
    }
}

/// Get color based on percent change
fn get_change_color(percent_change: f64) -> Color {
    if percent_change > 5.0 {
        // Busier - red/orange tones
        Color::from_rgb(0.9, 0.4, 0.3)
    } else if percent_change < -5.0 {
        // Quieter - green tones
        Color::from_rgb(0.3, 0.8, 0.5)
    } else {
        // Stable - blue
        style::ACCENT_BLUE
    }
}
