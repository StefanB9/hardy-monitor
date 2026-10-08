//! "Week": the weekday × hour heatmap and per-day summaries.

use hardy_core::analytics::{DayAnalysis, weekday_short};
use iced::{
    Alignment, Border, Element, Length,
    widget::{Canvas, Space, canvas::Cache, column, container, row, text},
};

use crate::{
    app::Message,
    style,
    time_range::AnalyticsRange,
    views::components::{card, card_with_actions, empty_state, legend_item, segmented},
    widgets::heatmap::{HeatmapWidget, WeekGrid},
};

const RANGES: [(&str, AnalyticsRange); 4] = [
    ("This week", AnalyticsRange::ThisWeek),
    ("2 weeks", AnalyticsRange::Last2Weeks),
    ("4 weeks", AnalyticsRange::Last4Weeks),
    ("8 weeks", AnalyticsRange::Last8Weeks),
];

/// The first hour after opening is skipped in per-day extremes.
const SKIP_OPENING_HOURS: u32 = 1;
const BAR_MAX_HEIGHT: f32 = 132.0;

/// Everything the Week view shows.
#[derive(Clone, Copy)]
pub struct WeekProps<'a> {
    pub grid: &'a WeekGrid,
    pub days: &'a [DayAnalysis],
    pub range: AnalyticsRange,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub heatmap_cache: &'a Cache,
    pub heatmap_tooltip_cache: &'a Cache,
}

fn swatch<'a>(color: iced::Color, width: f32) -> Element<'a, Message> {
    container(Space::new().width(width).height(10))
        .style(move |_| container::Style {
            background: Some(color.into()),
            ..Default::default()
        })
        .into()
}

/// Gradient strip matching the cell colours, plus the special cells.
fn legend<'a>(low: f64, high: f64) -> Element<'a, Message> {
    let mut strip = row![].spacing(0);
    for step in 0..=20 {
        let p = f64::from(step) * 5.0;
        strip = strip.push(swatch(style::occupancy_color(p, low, high), 8.0));
    }
    let caption = |s: &'a str| {
        text(s)
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_TERTIARY)
    };
    row![
        caption("0%"),
        container(strip).style(|_| container::Style {
            border: Border {
                radius: 3.0.into(),
                ..Default::default()
            },
            ..Default::default()
        }),
        caption("100%"),
        Space::new().width(style::SPACE_L),
        legend_item(style::NO_DATA, "No data yet"),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .into()
}

fn day_bars<'a>(days: &[DayAnalysis], low: f64, high: f64) -> Element<'a, Message> {
    let days: Vec<_> = days.iter().filter(|d| d.sample_count > 0).collect();
    if days.is_empty() {
        return empty_state("No data in this range");
    }
    let max = days.iter().map(|d| d.avg_occupancy).fold(1.0_f64, f64::max);
    let mut bars = row![].spacing(style::SPACE_M).align_y(Alignment::End);
    for day in days {
        #[allow(clippy::cast_possible_truncation)]
        let height = ((day.avg_occupancy / max) as f32 * BAR_MAX_HEIGHT).max(4.0);
        let color = style::occupancy_color(day.avg_occupancy, low, high);
        bars = bars.push(
            column![
                text(format!("{:.0}%", day.avg_occupancy))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_PRIMARY),
                container(Space::new().width(Length::Fill).height(height)).style(move |_| {
                    container::Style {
                        background: Some(color.into()),
                        border: Border {
                            radius: 4.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                }),
                text(weekday_short(day.weekday))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_SECONDARY),
            ]
            .spacing(style::SPACE_XS)
            .align_x(Alignment::Center)
            .width(Length::Fill),
        );
    }
    bars.height(Length::Fill).into()
}

fn extremes_table<'a>(grid: &WeekGrid, low: f64, high: f64) -> Element<'a, Message> {
    let header = |s: &'a str| {
        text(s)
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_TERTIARY)
    };
    let cell = |(hour, value): (u32, f64)| {
        row![
            text(format!("{hour:02}:00"))
                .size(style::TEXT_BODY)
                .color(style::TEXT_PRIMARY),
            text(format!("{value:.0}%"))
                .size(style::TEXT_CAPTION)
                .color(style::occupancy_color(value, low, high)),
        ]
        .spacing(style::SPACE_S)
        .align_y(Alignment::Center)
        .width(Length::Fill)
    };

    let mut table = column![row![
        header("").width(44),
        header("Quietest").width(Length::Fill),
        header("Busiest").width(Length::Fill),
    ]]
    .spacing(6);
    let mut any = false;
    for day in 0..7 {
        if let Some((quiet, busy)) = grid.extremes(day, SKIP_OPENING_HOURS) {
            any = true;
            let name = weekday_short(i32::try_from(day).unwrap_or(0));
            table = table.push(
                row![
                    text(name)
                        .size(style::TEXT_CAPTION)
                        .color(style::TEXT_SECONDARY)
                        .width(44),
                    cell(quiet),
                    cell(busy),
                ]
                .align_y(Alignment::Center),
            );
        }
    }
    if any {
        table.into()
    } else {
        empty_state("No data in this range")
    }
}

/// The Week page.
pub fn view(props: WeekProps<'_>) -> Element<'_, Message> {
    let (low, high) = (props.low_threshold, props.high_threshold);
    let heatmap = Canvas::new(HeatmapWidget {
        grid: props.grid,
        low_threshold: low,
        high_threshold: high,
        cache: props.heatmap_cache,
        tooltip_cache: props.heatmap_tooltip_cache,
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let heatmap_card = card_with_actions(
        "Average occupancy by hour",
        segmented(&RANGES, props.range, Message::SwitchAnalyticsRange),
        column![heatmap, legend(low, high)]
            .spacing(style::SPACE_M)
            .height(Length::Fill),
    )
    .height(Length::Fill);

    let summaries = row![
        card("Average by day", day_bars(props.days, low, high))
            .width(Length::FillPortion(1))
            .height(Length::Fill),
        card(
            "Best and worst hour per day",
            extremes_table(props.grid, low, high)
        )
        .width(Length::FillPortion(1))
        .height(Length::Fill),
    ]
    .spacing(style::SPACE_L)
    .height(Length::Fixed(272.0));

    column![heatmap_card, summaries]
        .spacing(style::SPACE_L)
        .height(Length::Fill)
        .into()
}
