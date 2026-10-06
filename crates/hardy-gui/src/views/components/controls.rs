//! Buttons, segmented controls and inputs.

use iced::{
    Border, Element, Length, Theme,
    widget::{button, row, text, text_input, toggler},
};

use crate::{app::Message, style};

fn button_style(
    background: iced::Color,
    text_color: iced::Color,
    border_color: iced::Color,
) -> button::Style {
    button::Style {
        background: Some(background.into()),
        text_color,
        border: Border {
            color: border_color,
            width: 1.0,
            radius: style::RADIUS_CONTROL.into(),
        },
        ..Default::default()
    }
}

fn hovered(status: button::Status) -> bool {
    matches!(status, button::Status::Hovered | button::Status::Pressed)
}

/// Main call to action.
pub fn primary_button(label: &str, message: Message) -> button::Button<'_, Message> {
    button(text(label).size(style::TEXT_BODY))
        .on_press(message)
        .padding([8, 16])
        .style(|_: &Theme, status| {
            let bg = if hovered(status) {
                style::mix(style::ACCENT, style::TEXT_PRIMARY, 0.15)
            } else {
                style::ACCENT
            };
            button_style(bg, style::BG_APP, bg)
        })
}

/// Secondary action on an elevated surface.
pub fn secondary_button(label: &str, message: Message) -> button::Button<'_, Message> {
    button(text(label).size(style::TEXT_BODY))
        .on_press(message)
        .padding([8, 16])
        .style(|_: &Theme, status| {
            let border = if hovered(status) {
                style::TEXT_TERTIARY
            } else {
                style::BORDER
            };
            button_style(style::BG_ELEVATED, style::TEXT_PRIMARY, border)
        })
}

/// Compact secondary button for card headers.
pub fn small_button(label: &str, message: Message) -> button::Button<'_, Message> {
    secondary_button(label, message).padding([4, 10])
}

/// Text-only button (links, toggles).
pub fn ghost_button(label: &str, message: Message) -> button::Button<'_, Message> {
    button(text(label).size(style::TEXT_CAPTION))
        .on_press(message)
        .padding(0)
        .style(|_: &Theme, status| button::Style {
            text_color: if hovered(status) {
                style::TEXT_PRIMARY
            } else {
                style::ACCENT
            },
            ..Default::default()
        })
}

/// A row of mutually exclusive options; the selected one is highlighted.
pub fn segmented<'a, T: Copy + PartialEq + 'a>(
    options: &[(&'a str, T)],
    selected: T,
    on_select: impl Fn(T) -> Message + Copy + 'a,
) -> Element<'a, Message> {
    let mut segments = row![].spacing(2);
    for &(label, value) in options {
        let active = value == selected;
        segments = segments.push(
            button(text(label).size(style::TEXT_CAPTION))
                .on_press(on_select(value))
                .padding([6, 12])
                .style(move |_: &Theme, status| {
                    let (bg, fg) = if active {
                        (style::ACCENT, style::BG_APP)
                    } else if hovered(status) {
                        (style::BORDER, style::TEXT_PRIMARY)
                    } else {
                        (style::BG_ELEVATED, style::TEXT_SECONDARY)
                    };
                    button_style(bg, fg, bg)
                }),
        );
    }
    iced::widget::container(segments)
        .padding(2)
        .style(|_| iced::widget::container::Style {
            background: Some(style::BG_ELEVATED.into()),
            border: Border {
                radius: (style::RADIUS_CONTROL + 2.0).into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .into()
}

/// `YYYY-MM-DD` input.
pub fn date_input(
    value: &str,
    on_change: impl Fn(String) -> Message + 'static,
) -> Element<'_, Message> {
    text_input("YYYY-MM-DD", value)
        .on_input(on_change)
        .padding([6, 10])
        .width(Length::Fixed(118.0))
        .size(style::TEXT_CAPTION)
        .style(|_, status| {
            let border = if matches!(status, text_input::Status::Focused { .. }) {
                style::ACCENT
            } else {
                style::BORDER
            };
            text_input::Style {
                background: style::BG_ELEVATED.into(),
                border: Border {
                    color: border,
                    width: 1.0,
                    radius: style::RADIUS_CONTROL.into(),
                },
                icon: style::TEXT_TERTIARY,
                placeholder: style::TEXT_TERTIARY,
                value: style::TEXT_PRIMARY,
                selection: style::tint(style::ACCENT, 0.4),
            }
        })
        .into()
}

/// On/off switch.
pub fn switch<'a>(is_on: bool, on_toggle: impl Fn(bool) -> Message + 'a) -> Element<'a, Message> {
    toggler(is_on)
        .on_toggle(on_toggle)
        .size(22)
        .style(move |_: &Theme, _| toggler::Style {
            background: if is_on {
                style::ACCENT
            } else {
                style::BG_ELEVATED
            }
            .into(),
            background_border_width: 1.0,
            background_border_color: if is_on { style::ACCENT } else { style::BORDER },
            foreground: style::TEXT_PRIMARY.into(),
            foreground_border_width: 0.0,
            foreground_border_color: iced::Color::TRANSPARENT,
            text_color: None,
            border_radius: None,
            padding_ratio: 0.1,
        })
        .into()
}
