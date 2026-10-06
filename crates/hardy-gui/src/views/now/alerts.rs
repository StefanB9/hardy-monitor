//! Low-occupancy alert controls (shared with the daemon and the phone).

use hardy_core::alert::AlertDuration;
use iced::{
    Alignment, Border, Color, Length, Theme,
    widget::{Space, column, container, row, slider, text},
};

use super::NowProps;
use crate::{
    app::Message,
    style,
    views::components::{card_with_actions, segmented, switch},
};

const DURATIONS: [(&str, AlertDuration); 3] = [
    ("Until closing", AlertDuration::UntilClosing),
    ("2 hours", AlertDuration::Hours(2)),
    ("Always", AlertDuration::Always),
];

fn label(content: &str) -> text::Text<'_> {
    text(content)
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_TERTIARY)
}

pub(super) fn card<'a>(props: &NowProps<'a>) -> container::Container<'a, Message> {
    let active = props.alert_active;
    let rail = if active {
        style::ACCENT
    } else {
        style::TEXT_TERTIARY
    };

    let threshold = column![
        row![
            label("Notify when below"),
            Space::new().width(Length::Fill),
            text(format!("{:.0}%", props.alert_threshold))
                .size(style::TEXT_HEADING)
                .color(style::TEXT_PRIMARY),
        ]
        .align_y(Alignment::Center),
        slider(
            5.0..=60.0,
            props.alert_threshold,
            Message::NotificationThresholdChanged
        )
        .on_release(Message::NotificationThresholdReleased)
        .step(5.0)
        .style(move |_: &Theme, _| slider::Style {
            rail: slider::Rail {
                backgrounds: (rail.into(), style::BG_ELEVATED.into()),
                width: 4.0,
                border: Border {
                    radius: 2.0.into(),
                    ..Default::default()
                },
            },
            handle: slider::Handle {
                shape: slider::HandleShape::Circle { radius: 8.0 },
                background: style::TEXT_PRIMARY.into(),
                border_width: 2.0,
                border_color: rail,
            },
        }),
    ]
    .spacing(style::SPACE_S);

    let duration = column![
        label("For"),
        segmented(
            &DURATIONS,
            props.alert_duration,
            Message::NotificationDurationSelected
        ),
    ]
    .spacing(style::SPACE_S);

    let status_color: Color = if active {
        style::SUCCESS
    } else {
        style::TEXT_SECONDARY
    };

    card_with_actions(
        "Quiet alerts",
        switch(active, Message::NotificationToggled),
        column![
            threshold,
            duration,
            text("Sent to this desktop and, via the daemon, to your phone.")
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
            Space::new().height(Length::Fill),
            text(props.alert_status.clone())
                .size(style::TEXT_CAPTION)
                .color(status_color),
        ]
        .spacing(style::SPACE_L)
        .height(Length::Fill),
    )
    .height(Length::Fill)
}
