use hardy_monitor::style;
use iced::{
    Border, Color, Element, Shadow, Theme, Vector,
    widget::{button, container},
};

use crate::app::Message;

pub fn card_container<'a>(
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(content).padding(24).style(|_| container::Style {
        background: Some(style::BG_CARD.into()),
        border: Border {
            color: Color::TRANSPARENT,
            width: 0.0,
            radius: 16.0.into(),
        },
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.3),
            offset: Vector::new(0.0, 4.0),
            blur_radius: 10.0,
        },
        ..Default::default()
    })
}

pub fn preset_btn(label: &str, days: i64, current: Option<i64>) -> Element<'_, Message> {
    use iced::widget::text;

    let active = current == Some(days);
    button(text(label.to_string()).size(12))
        .on_press(Message::HistoryPresetSelected(days))
        .padding([6, 12])
        .style(move |_, _| {
            if active {
                primary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
            } else {
                secondary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
            }
        })
        .into()
}

pub fn primary_btn_style(_: &Theme, _: iced::widget::button::Status) -> button::Style {
    button::Style {
        background: Some(style::ACCENT_BLUE.into()),
        text_color: style::BG_DARK,
        border: Border {
            radius: 6.0.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn secondary_btn_style(_: &Theme, _: iced::widget::button::Status) -> button::Style {
    button::Style {
        background: Some(style::BG_DARK.into()),
        text_color: style::TEXT_BRIGHT,
        border: Border {
            radius: 6.0.into(),
            color: style::STROKE_DIM,
            width: 1.0,
        },
        ..Default::default()
    }
}
