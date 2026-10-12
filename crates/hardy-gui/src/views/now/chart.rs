//! Occupancy chart card with range selection and legend.

use chrono::{DateTime, Utc};
use hardy_ml::suite::ModelId;
use iced::{
    Alignment, Border, Color, Element, Length,
    widget::{Canvas, Space, button, column, container, pick_list, row, text},
};

use super::NowProps;
use crate::{
    app::Message,
    forecasting::{ForecastChoice, MAX_OVERLAYS, ModelOption},
    style,
    time_range::{ChartRange, Step, can_step_back},
    views::components::{
        card_with_actions, date_input, legend_item, primary_button, secondary_button, segmented,
    },
    widgets::history_chart::HistoryChart,
};

const RANGES: [(&str, ChartRange); 4] = [
    ("Today", ChartRange::Today),
    ("7 days", ChartRange::Days7),
    ("30 days", ChartRange::Days30),
    ("Custom", ChartRange::Custom),
];

type Window = Option<(DateTime<Utc>, DateTime<Utc>)>;

pub(super) fn card<'a>(props: &NowProps<'a>) -> container::Container<'a, Message> {
    let window = props.chart_range.window(
        props.now,
        props.schedule,
        (props.custom_start, props.custom_end),
        props.chart_offset,
    );
    let (back, forward) = window.map_or((false, false), |(start, end)| {
        (
            can_step_back(start, props.first_reading),
            props
                .chart_range
                .can_step_forward(props.chart_offset, end, props.now),
        )
    });
    let chart = row![
        step_button("‹", Step::Back, back),
        chart_area(props, window),
        step_button("›", Step::Forward, forward),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .height(Length::Fill);

    card_with_actions(
        "Occupancy",
        segmented(&RANGES, props.chart_range, Message::ChartRangeSelected),
        column![toolbar(props, window), models(props), chart]
            .spacing(style::SPACE_M)
            .height(Length::Fill),
    )
    .height(Length::Fill)
}

/// The shown period, the legend and, for a custom range, its dates.
fn toolbar<'a>(props: &NowProps<'a>, window: Window) -> Element<'a, Message> {
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
    let period = window.map(|w| {
        text(
            props
                .chart_range
                .period_label(props.chart_offset, w, props.schedule.timezone()),
        )
        .size(style::TEXT_BODY)
        .color(style::TEXT_PRIMARY)
    });

    let mut toolbar = row![]
        .push(period)
        .push(legend)
        .push(Space::new().width(Length::Fill))
        .spacing(style::SPACE_L)
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
    toolbar.into()
}

/// Which forecast is drawn, and the models drawn beside it for comparison.
fn models<'a>(props: &NowProps<'a>) -> Element<'a, Message> {
    let mut choices = Vec::with_capacity(props.available_models.len() + 1);
    choices.push(ForecastChoice::Best);
    choices.extend(
        props
            .available_models
            .iter()
            .map(|id| ForecastChoice::Model(*id)),
    );
    let shown: Vec<ModelId> = props.overlays.iter().map(|(id, _)| *id).collect();
    let addable: Vec<ModelOption> = props
        .available_models
        .iter()
        .filter(|id| !shown.contains(id) && props.forecast_choice != ForecastChoice::Model(**id))
        .map(|id| ModelOption(*id))
        .collect();

    let mut line = row![
        caption("Forecast"),
        pick_list(
            choices,
            Some(props.forecast_choice),
            Message::ForecastChoiceSelected
        )
        .text_size(style::TEXT_CAPTION)
        .padding([4, 8]),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center);
    for ((id, _), color) in props.overlays.iter().zip(style::COMPARISON) {
        line = line.push(overlay_chip(*id, color));
    }
    if shown.len() < MAX_OVERLAYS && !addable.is_empty() {
        line = line.push(
            pick_list(addable, None::<ModelOption>, Message::OverlayAdded)
                .placeholder("Compare with…")
                .text_size(style::TEXT_CAPTION)
                .padding([4, 8]),
        );
    }
    line.into()
}

/// A comparison model in its line colour, removable.
fn overlay_chip(id: ModelId, color: Color) -> Element<'static, Message> {
    button(
        row![
            legend_item(color, id.name()),
            text("×")
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        ]
        .spacing(style::SPACE_S)
        .align_y(Alignment::Center),
    )
    .on_press(Message::OverlayRemoved(ModelOption(id)))
    .padding([4, 8])
    .style(|_, status| button::Style {
        background: Some(
            if status == button::Status::Hovered {
                style::BG_ELEVATED
            } else {
                style::BG_CARD
            }
            .into(),
        ),
        border: Border {
            color: style::BORDER,
            width: 1.0,
            radius: 12.0.into(),
        },
        ..button::Style::default()
    })
    .into()
}

fn caption(label: &str) -> text::Text<'_> {
    text(label)
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_TERTIARY)
}

/// The chart, or why there is none.
fn chart_area<'a>(props: &NowProps<'a>, window: Window) -> Element<'a, Message> {
    let notice = |message: &'a str| -> Element<'a, Message> {
        container(
            text(message)
                .size(style::TEXT_BODY)
                .color(style::TEXT_TERTIARY),
        )
        .center(Length::Fill)
        .into()
    };
    let Some((range_start, range_end)) = window else {
        return notice("Enter a start and end date (YYYY-MM-DD)");
    };
    // A past period the daemon did not collect.
    let has_readings = props
        .history
        .iter()
        .any(|l| (range_start..range_end).contains(&l.timestamp));
    if range_end <= props.now && !has_readings {
        return notice("No readings in this period");
    }
    Canvas::new(HistoryChart {
        history: props.history,
        forecast: props.forecast,
        overlays: props.overlays,
        range_start,
        range_end,
        now: props.now,
        threshold: props.alert_active.then_some(props.alert_threshold),
        timezone: props.schedule.timezone(),
        cache: props.chart_cache,
    })
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

/// Previous / next period; disabled at the data's start and at now.
fn step_button(glyph: &str, step: Step, enabled: bool) -> Element<'_, Message> {
    secondary_button(glyph, Message::ChartStep(step))
        .on_press_maybe(enabled.then_some(Message::ChartStep(step)))
        .padding([10, 8])
        .into()
}
