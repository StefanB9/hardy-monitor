//! "Plan": when to go on one of the next seven days, for a visit length and
//! time range that are remembered.

mod labels;

use chrono::{DateTime, Utc};
use hardy_core::{GymSchedule, analytics::visit::VisitWindow};
use iced::{
    Alignment, Border, Element, Length, Theme,
    widget::{Space, button, canvas, column, container, row, text},
};

use crate::{
    app::Message,
    plan_settings::{HourRange, VisitLength},
    planner::{DayPlan, Planner, RangeEnd, RangeKind, is_weekend_like},
    style::{self, OccupancyLevel},
    views::components::{
        badge, card, card_with_actions, empty_state, scroll, segmented, small_button,
    },
    widgets::day_curve::DayCurve,
};

/// Everything the Plan view shows.
#[derive(Clone, Copy)]
pub struct PlanProps<'a> {
    pub planner: &'a Planner,
    pub now: DateTime<Utc>,
    pub schedule: &'a GymSchedule,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub curve_cache: &'a canvas::Cache,
}

impl PlanProps<'_> {
    fn level_color(&self, value: f64) -> iced::Color {
        OccupancyLevel::from_percentage(value, self.low_threshold, self.high_threshold).color()
    }
}

/// One chip per day: its name and best expected occupancy.
fn day_chips<'a>(props: &PlanProps<'a>) -> Element<'a, Message> {
    let today = props
        .now
        .with_timezone(&props.schedule.timezone())
        .date_naive();
    let selected = props.planner.selected();
    let mut chips = row![].spacing(style::SPACE_S);
    for (index, day) in props.planner.days().iter().enumerate() {
        let best = day.plan.best.map(|w| w.expected.mean);
        let value = best.map_or_else(|| "–".to_string(), |v| format!("~{v:.0}%"));
        let value_color = best.map_or(style::TEXT_TERTIARY, |v| props.level_color(v));
        let mut content = column![
            text(labels::chip_title(day.date, today))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY),
            text(value).size(style::TEXT_HEADING).color(value_color),
        ]
        .spacing(2)
        .width(Length::Fill)
        .align_x(Alignment::Center);
        if day.holiday {
            content = content.push(
                text("Holiday")
                    .size(style::TEXT_CAPTION)
                    .color(style::WARNING),
            );
        }
        let active = index == selected;
        chips = chips.push(
            button(content)
                .on_press(Message::PlanDaySelected(index))
                .width(Length::FillPortion(1))
                .padding([8, 6])
                .style(move |_: &Theme, status| {
                    let hovered =
                        matches!(status, button::Status::Hovered | button::Status::Pressed);
                    let (bg, border) = if active {
                        (style::tint(style::ACCENT, 0.16), style::ACCENT)
                    } else if hovered {
                        (style::BG_ELEVATED, style::TEXT_TERTIARY)
                    } else {
                        (style::BG_ELEVATED, style::BORDER)
                    };
                    button::Style {
                        background: Some(bg.into()),
                        text_color: style::TEXT_PRIMARY,
                        border: Border {
                            color: border,
                            width: 1.0,
                            radius: style::RADIUS_CONTROL.into(),
                        },
                        ..Default::default()
                    }
                }),
        );
    }
    chips.into()
}

/// "Workdays  ‹ 17:00 › to ‹ 24:00 ›"; highlighted when it applies to the
/// chosen day.
fn range_control(
    label: &str,
    kind: RangeKind,
    range: HourRange,
    applies: bool,
) -> Element<'_, Message> {
    let stepper = |end: RangeEnd, hour: u32| {
        row![
            small_button(
                "‹",
                Message::PlanRangeChanged(kind, end, hour.saturating_sub(1))
            ),
            text(labels::hour_label(hour))
                .size(style::TEXT_BODY)
                .color(style::TEXT_PRIMARY)
                .width(Length::Fixed(48.0))
                .align_x(Alignment::Center),
            small_button("›", Message::PlanRangeChanged(kind, end, hour + 1)),
        ]
        .spacing(style::SPACE_XS)
        .align_y(Alignment::Center)
    };
    row![
        text(label)
            .size(style::TEXT_BODY)
            .color(if applies {
                style::TEXT_PRIMARY
            } else {
                style::TEXT_TERTIARY
            })
            .width(Length::Fixed(80.0)),
        stepper(RangeEnd::Earliest, range.earliest),
        text("to")
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_TERTIARY),
        stepper(RangeEnd::Latest, range.latest),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .into()
}

fn controls<'a>(props: &PlanProps<'a>) -> Element<'a, Message> {
    let settings = props.planner.settings();
    let weekend = props
        .planner
        .selected_day()
        .is_some_and(|d| is_weekend_like(d.date));
    let lengths: Vec<(&str, VisitLength)> =
        VisitLength::ALL.iter().map(|l| (l.label(), *l)).collect();
    card_with_actions(
        "Plan your visit",
        segmented(&lengths, settings.length, Message::PlanLengthSelected),
        column![
            day_chips(props),
            row![
                range_control("Workdays", RangeKind::Workday, settings.workday, !weekend),
                Space::new().width(Length::Fixed(style::SPACE_XL)),
                range_control("Weekends", RangeKind::Weekend, settings.weekend, weekend),
            ]
            .align_y(Alignment::Center),
        ]
        .spacing(style::SPACE_L),
    )
    .width(Length::Fill)
    .into()
}

fn best_card<'a>(props: &PlanProps<'a>, day: &DayPlan) -> Element<'a, Message> {
    let tz = props.schedule.timezone();
    let body: Element<'a, Message> = match day.plan.best {
        Some(window) => column![
            text(labels::window_text(&window, tz))
                .size(32)
                .color(style::TEXT_PRIMARY),
            text(labels::expected_text(window.expected))
                .size(style::TEXT_BODY)
                .color(props.level_color(window.expected.mean)),
            text(labels::source_text(window.source, day.date))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        ]
        .spacing(style::SPACE_XS)
        .into(),
        None => empty_state(labels::no_window_reason(
            day,
            props.now,
            props.planner.settings().length,
            props.planner.has_averages(),
        )),
    };
    card("Best time", body)
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .into()
}

fn alternatives_card<'a>(props: &PlanProps<'a>, day: &DayPlan) -> Element<'a, Message> {
    let tz = props.schedule.timezone();
    let row_of = |part, window: &VisitWindow| {
        let is_best = day.plan.best.is_some_and(|b| b.start == window.start);
        row![
            text(labels::part_label(part))
                .size(style::TEXT_BODY)
                .color(style::TEXT_SECONDARY)
                .width(Length::Fixed(90.0)),
            text(labels::window_text(window, tz))
                .size(style::TEXT_BODY)
                .color(if is_best {
                    style::TEXT_PRIMARY
                } else {
                    style::TEXT_SECONDARY
                }),
            Space::new().width(Length::Fill),
            badge(
                format!("~{:.0}%", window.expected.mean),
                props.level_color(window.expected.mean),
            ),
        ]
        .align_y(Alignment::Center)
        .spacing(style::SPACE_S)
    };
    let body: Element<'a, Message> = if day.plan.by_part.is_empty() {
        empty_state("No windows")
    } else {
        day.plan
            .by_part
            .iter()
            .fold(column![].spacing(style::SPACE_M), |col, (part, window)| {
                col.push(row_of(*part, window))
            })
            .into()
    };
    card("By part of the day", body)
        .width(Length::FillPortion(1))
        .height(Length::Fill)
        .into()
}

fn curve_card<'a>(props: &PlanProps<'a>, day: &'a DayPlan) -> Element<'a, Message> {
    let curve = canvas(DayCurve {
        estimates: &day.estimates,
        window: day.plan.best,
        now: props.now,
        timezone: props.schedule.timezone(),
        cache: props.curve_cache,
    })
    .width(Length::Fill)
    .height(Length::Fixed(240.0));
    card("Expected occupancy", curve).width(Length::Fill).into()
}

pub fn view(props: PlanProps<'_>) -> Element<'_, Message> {
    let Some(day) = props.planner.selected_day() else {
        return container(empty_state("Collecting data…"))
            .width(Length::Fill)
            .into();
    };
    let content = column![
        controls(&props),
        row![best_card(&props, day), alternatives_card(&props, day)]
            .spacing(style::SPACE_L)
            .height(Length::Fixed(190.0)),
        curve_card(&props, day),
    ]
    .spacing(style::SPACE_L);
    scroll(content)
}
