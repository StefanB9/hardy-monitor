use chrono::{DateTime, Local, Utc};
use hardy_monitor::{
    analytics::{self, DayAnalysis, Insight, OccupancyStats, TrendDirection},
    style,
};
use iced::{
    Alignment, Border, Element, Length,
    widget::{Space, column, container, row, scrollable, text},
};

use crate::{app::Message, views::components::card_container};

pub struct InsightsProps<'a> {
    pub trend: Option<TrendDirection>,
    pub stats: Option<&'a OccupancyStats>,
    pub peak_hours: &'a [(i32, i32, f64)],
    pub quiet_hours: &'a [(i32, i32, f64)],
    pub day_analysis: &'a [DayAnalysis],
    pub insights: &'a [Insight],
    pub ml_has_model: bool,
    pub ml_training_in_progress: bool,
    pub ml_last_trained: Option<DateTime<Utc>>,
}

pub fn view(props: InsightsProps<'_>) -> Element<'_, Message> {
    let trend_card = {
        let (trend_icon, trend_text, trend_color) = match props.trend {
            Some(TrendDirection::Increasing) => ("^", "Getting Busier", style::ACCENT_RED),
            Some(TrendDirection::Decreasing) => ("v", "Getting Quieter", style::ACCENT_GREEN),
            Some(TrendDirection::Stable) => ("->", "Staying Stable", style::ACCENT_CYAN),
            Some(TrendDirection::Insufficient) | None => {
                ("?", "Collecting Data", style::TEXT_MUTED)
            }
        };

        card_container(column![
            text("Overall Trend").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            row![
                text(trend_icon).size(32).color(trend_color),
                Space::new().width(15),
                column![
                    text(trend_text).size(20).color(trend_color),
                    text("vs previous 4 weeks")
                        .size(12)
                        .color(style::TEXT_MUTED),
                ]
            ]
            .align_y(Alignment::Center)
        ])
        .width(Length::FillPortion(1))
    };

    let stats_card = if let Some(stats) = props.stats {
        let consistency = if stats.coefficient_of_variation < 0.3 {
            ("Very Predictable", style::ACCENT_GREEN)
        } else if stats.coefficient_of_variation < 0.5 {
            ("Moderately Predictable", style::ACCENT_ORANGE)
        } else {
            ("Highly Variable", style::ACCENT_RED)
        };

        card_container(column![
            text("Statistics").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            row![
                column![
                    text("Average").size(12).color(style::TEXT_MUTED),
                    text(format!("{:.1}%", stats.mean))
                        .size(24)
                        .color(style::TEXT_BRIGHT),
                ],
                Space::new().width(30),
                column![
                    text("Range").size(12).color(style::TEXT_MUTED),
                    text(format!("{:.0}% - {:.0}%", stats.min, stats.max))
                        .size(18)
                        .color(style::TEXT_BRIGHT),
                ],
            ]
            .align_y(Alignment::End),
            Space::new().height(15),
            row![
                text("Consistency: ").size(12).color(style::TEXT_MUTED),
                text(consistency.0).size(12).color(consistency.1),
            ]
        ])
        .width(Length::FillPortion(1))
    } else {
        card_container(column![
            text("Statistics").size(14).color(style::TEXT_MUTED),
            Space::new().height(20),
            text("Loading...").color(style::TEXT_MUTED),
        ])
        .width(Length::FillPortion(1))
    };

    let peak_card = card_container(column![
        text("Busiest Times").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        {
            let mut peak_col = column![].spacing(8);
            for (weekday, hour, pct) in props.peak_hours.iter().take(5) {
                peak_col = peak_col.push(
                    row![
                        container(text(format!("{pct:.0}%")).size(12).color(style::BG_DARK))
                            .padding([4, 8])
                            .style(|_| container::Style {
                                background: Some(style::ACCENT_RED.into()),
                                border: Border {
                                    radius: 4.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                        Space::new().width(10),
                        text(format!(
                            "{} {:02}:00",
                            analytics::weekday_short(*weekday),
                            hour
                        ))
                        .size(14)
                        .color(style::TEXT_BRIGHT),
                    ]
                    .align_y(Alignment::Center),
                );
            }
            peak_col
        }
    ])
    .width(Length::FillPortion(1));

    let quiet_card = card_container(column![
        text("Quietest Times").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        {
            let mut quiet_col = column![].spacing(8);
            for (weekday, hour, pct) in props.quiet_hours.iter().take(5) {
                quiet_col = quiet_col.push(
                    row![
                        container(text(format!("{pct:.0}%")).size(12).color(style::BG_DARK))
                            .padding([4, 8])
                            .style(|_| container::Style {
                                background: Some(style::ACCENT_GREEN.into()),
                                border: Border {
                                    radius: 4.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                        Space::new().width(10),
                        text(format!(
                            "{} {:02}:00",
                            analytics::weekday_short(*weekday),
                            hour
                        ))
                        .size(14)
                        .color(style::TEXT_BRIGHT),
                    ]
                    .align_y(Alignment::Center),
                );
            }
            quiet_col
        }
    ])
    .width(Length::FillPortion(1));

    let days_card = card_container(column![
        text("Daily Patterns").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        {
            let mut days_row = row![].spacing(30);
            for day in props.day_analysis {
                if day.sample_count > 0 {
                    let bar_height = (day.avg_occupancy * 1.5).max(5.0);
                    let color = if day.avg_occupancy < 40.0 {
                        style::ACCENT_GREEN
                    } else if day.avg_occupancy < 60.0 {
                        style::ACCENT_ORANGE
                    } else {
                        style::ACCENT_RED
                    };

                    days_row = days_row.push(
                        column![
                            container(
                                Space::new()
                                    .width(30)
                                    .height(Length::Fixed(bar_height as f32))
                            )
                            .style(move |_| container::Style {
                                background: Some(color.into()),
                                border: Border {
                                    radius: 4.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                            Space::new().height(8),
                            text(&day.day_name[..3]).size(12).color(style::TEXT_MUTED),
                            text(format!("{:.0}%", day.avg_occupancy))
                                .size(12)
                                .color(style::TEXT_BRIGHT),
                        ]
                        .align_x(Alignment::Center),
                    );
                }
            }
            container(days_row)
                .width(Length::Fill)
                .align_x(Alignment::Center)
        }
    ])
    .width(Length::Fill);

    let insights_card = card_container(column![
        text("Key Insights").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        {
            let mut insights_col = column![].spacing(12);
            for insight in props.insights.iter().take(6) {
                let importance_color = match insight.importance {
                    5 => style::ACCENT_GREEN,
                    4 => style::ACCENT_CYAN,
                    3 => style::ACCENT_ORANGE,
                    _ => style::TEXT_MUTED,
                };

                insights_col = insights_col.push(
                    container(column![
                        row![
                            container(
                                text(format!("{}", insight.importance))
                                    .size(10)
                                    .color(style::BG_DARK)
                            )
                            .padding([2, 6])
                            .style(move |_| container::Style {
                                background: Some(importance_color.into()),
                                border: Border {
                                    radius: 8.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                            Space::new().width(10),
                            text(&insight.title).size(14).color(style::TEXT_BRIGHT),
                        ]
                        .align_y(Alignment::Center),
                        Space::new().height(4),
                        text(&insight.description).size(12).color(style::TEXT_MUTED),
                    ])
                    .padding(12)
                    .style(|_| container::Style {
                        background: Some(style::BG_DARK.into()),
                        border: Border {
                            radius: 8.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                );
            }

            if props.insights.is_empty() {
                insights_col = insights_col.push(
                    text("No insights yet. Keep collecting data!")
                        .size(14)
                        .color(style::TEXT_MUTED),
                );
            }

            insights_col
        }
    ])
    .width(Length::Fill);

    let ml_status_card = {
        let (status_text, status_color) = if props.ml_training_in_progress {
            ("Training...", style::ACCENT_ORANGE)
        } else if props.ml_has_model {
            ("Active", style::ACCENT_GREEN)
        } else {
            ("Collecting data", style::TEXT_MUTED)
        };

        let trained_str = props
            .ml_last_trained
            .map(|t| t.with_timezone(&Local).format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "N/A".to_string());

        card_container(column![
            text("ML Prediction Model")
                .size(14)
                .color(style::TEXT_MUTED),
            Space::new().height(15),
            row![
                text("Status:").size(12).color(style::TEXT_MUTED),
                Space::new().width(8),
                text(status_text).size(12).color(status_color),
                Space::new().width(Length::Fill),
                text("Last trained:").size(12).color(style::TEXT_MUTED),
                Space::new().width(8),
                text(trained_str).size(12).color(style::TEXT_BRIGHT),
            ]
            .align_y(Alignment::Center),
        ])
        .width(Length::Fill)
    };

    let content = column![
        row![trend_card, stats_card]
            .spacing(20)
            .height(Length::Fixed(160.0)),
        Space::new().height(20),
        days_card,
        Space::new().height(20),
        row![peak_card, quiet_card].spacing(20),
        Space::new().height(20),
        insights_card,
        Space::new().height(20),
        ml_status_card,
    ]
    .padding(10);

    scrollable(content)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
}
