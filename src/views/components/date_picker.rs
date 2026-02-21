use hardy_monitor::style;
use iced::{Border, Element, Length, widget::text_input};

use crate::app::Message;

pub fn styled_input(
    val: &str,
    on_change: impl Fn(String) -> Message + 'static,
) -> Element<'_, Message> {
    text_input("YYYY-MM-DD", val)
        .on_input(on_change)
        .padding(8)
        .width(Length::Fixed(110.0))
        .size(12)
        .style(|_, status| {
            let border_color = if matches!(status, iced::widget::text_input::Status::Focused { .. })
            {
                style::ACCENT_BLUE
            } else {
                style::STROKE_DIM
            };
            text_input::Style {
                background: style::BG_DARK.into(),
                border: Border {
                    color: border_color,
                    width: 1.0,
                    radius: 6.0.into(),
                },
                icon: style::TEXT_MUTED,
                placeholder: style::TEXT_MUTED,
                value: style::TEXT_BRIGHT,
                selection: style::ACCENT_BLUE,
            }
        })
        .into()
}
