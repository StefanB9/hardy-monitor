use hardy_core::db::HourlyAverage;
use iced::{
    Alignment, Border, Color, Element, Length, Theme,
    widget::{Canvas, Space, button, canvas::Cache, column, container, row, text},
};

use crate::{
    app::{AnalyticsRange, Message},
    style,
    views::components::{card_container, primary_btn_style, secondary_btn_style},
    widgets::heatmap::HeatmapWidget,
};

#[derive(Clone, Copy)]
pub struct WeeklyPatternProps<'a> {
    pub analytics_data: &'a [HourlyAverage],
    pub analytics_range: AnalyticsRange,
    pub heatmap_cache: &'a Cache,
    pub heatmap_tooltip_cache: &'a Cache,
}

pub fn view(props: WeeklyPatternProps<'_>) -> Element<'_, Message> {
    let range_btn = |label: &str, range: AnalyticsRange| {
        let active = props.analytics_range == range;
        button(text(label.to_string()).size(14))
            .on_press(Message::SwitchAnalyticsRange(range))
            .padding([8, 16])
            .style(move |_, _| {
                if active {
                    primary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
                } else {
                    secondary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
                }
            })
    };

    let controls = row![
        range_btn("This Week", AnalyticsRange::ThisWeek),
        range_btn("Last 2 Weeks", AnalyticsRange::Last2Weeks),
        range_btn("Last 4 Weeks", AnalyticsRange::Last4Weeks),
        range_btn("Last 8 Weeks", AnalyticsRange::Last8Weeks)
    ]
    .spacing(10);

    let heatmap = Canvas::new(HeatmapWidget {
        data: props.analytics_data,
        cache: props.heatmap_cache,
        tooltip_cache: props.heatmap_tooltip_cache,
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let heatmap_element = Element::from(heatmap).map(|()| Message::ChartInteraction);

    let legend_item = |color: Color, label: &str| {
        row![
            container(Space::new().width(12).height(12)).style(move |_| container::Style {
                background: Some(color.into()),
                border: Border {
                    radius: 2.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
            text(label.to_string()).size(12).color(style::TEXT_MUTED)
        ]
        .spacing(6)
        .align_y(Alignment::Center)
    };

    let legend = row![
        legend_item(style::ACCENT_GREEN, "Low"),
        legend_item(style::ACCENT_ORANGE, "Busy"),
        legend_item(style::ACCENT_RED, "Full")
    ]
    .spacing(15);

    let mut row_content = row![].spacing(15);
    for (idx, day_name) in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        .iter()
        .enumerate()
    {
        if let Some(b) = props
            .analytics_data
            .iter()
            .filter(|d| d.weekday == i32::try_from(idx).unwrap_or(0))
            .min_by(|a, b| {
                a.avg_percentage
                    .partial_cmp(&b.avg_percentage)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            row_content = row_content.push(
                column![
                    text(day_name.to_string()).size(12).color(style::TEXT_MUTED),
                    text(format!("{:02}:00", b.hour))
                        .size(14)
                        .color(style::ACCENT_CYAN)
                ]
                .spacing(2),
            );
        }
    }

    card_container(column![
        row![
            text("Weekly Occupancy Heatmap")
                .size(16)
                .color(style::TEXT_MUTED),
            Space::new().width(Length::Fill),
            controls
        ]
        .align_y(Alignment::Center),
        Space::new().height(20),
        heatmap_element,
        Space::new().height(20),
        row![
            container(row_content),
            Space::new().width(Length::Fill),
            legend
        ]
        .align_y(Alignment::Center)
    ])
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}
