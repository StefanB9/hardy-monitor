//! Occupancy chart card with range selection and legend.

use iced::{
    Alignment, Element, Length,
    widget::{Canvas, Space, column, container, row, text},
};

use super::NowProps;
use crate::{
    app::Message,
    style,
    time_range::ChartRange,
    views::components::{card_with_actions, date_input, legend_item, primary_button, segmented},
    widgets::history_chart::HistoryChart,
};

const RANGES: [(&str, ChartRange); 4] = [
    ("Today", ChartRange::Today),
    ("7 days", ChartRange::Days7),
    ("30 days", ChartRange::Days30),
    ("Custom", ChartRange::Custom),
];

pub(super) fn card<'a>(props: &NowProps<'a>) -> container::Container<'a, Message> {
    let mut legend = row![
        legend_item(style::ACCENT, "Measured"),
        legend_item(style::FORECAST, "Forecast"),
        legend_item(style::tint(style::FORECAST, 0.35), "Likely range (80%)"),
    ]
    .spacing(style::SPACE_L)
    .align_y(Alignment::Center);
    if props.alert_active {
        legend = legend.push(legend_item(style::SUCCESS, "Alert threshold"));
    }

    let mut toolbar = row![legend, Space::new().width(Length::Fill)]
        .spacing(style::SPACE_S)
        .align_y(Alignment::Center);
    if props.chart_range == ChartRange::Custom {
        toolbar = toolbar
            .push(date_input(props.custom_start, Message::CustomStartChanged))
            .push(
                text("to")
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_TERTIARY),
            )
            .push(date_input(props.custom_end, Message::CustomEndChanged))
            .push(primary_button("Show", Message::ApplyCustomRange).padding([6, 12]));
    }

    let window = props.chart_range.window(
        props.now,
        props.schedule,
        (props.custom_start, props.custom_end),
    );
    let chart: Element<'a, Message> = match window {
        Some((range_start, range_end)) => Canvas::new(HistoryChart {
            history: props.history,
            forecast: props.forecast,
            range_start,
            range_end,
            now: props.now,
            threshold: props.alert_active.then_some(props.alert_threshold),
            timezone: props.schedule.timezone(),
            cache: props.chart_cache,
        })
        .width(Length::Fill)
        .height(Length::Fill)
        .into(),
        None => container(
            text("Enter a start and end date (YYYY-MM-DD)")
                .size(style::TEXT_BODY)
                .color(style::TEXT_TERTIARY),
        )
        .center(Length::Fill)
        .into(),
    };

    card_with_actions(
        "Occupancy",
        segmented(&RANGES, props.chart_range, Message::ChartRangeSelected),
        column![toolbar, chart]
            .spacing(style::SPACE_M)
            .height(Length::Fill),
    )
    .height(Length::Fill)
}
