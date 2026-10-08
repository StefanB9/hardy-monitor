//! Cards, stat tiles, badges and other non-interactive surfaces.

use iced::{
    Alignment, Border, Color, Element, Length,
    widget::{Space, column, container, row, scrollable, text},
};

use crate::{app::Message, style};

fn card_style() -> container::Style {
    container::Style {
        background: Some(style::BG_CARD.into()),
        border: Border {
            color: style::BORDER,
            width: 1.0,
            radius: style::RADIUS_CARD.into(),
        },
        ..Default::default()
    }
}

fn card_title(title: &str) -> Element<'_, Message> {
    text(title)
        .size(style::TEXT_HEADING)
        .color(style::TEXT_PRIMARY)
        .into()
}

/// A titled card.
pub fn card<'a>(
    title: &'a str,
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(column![card_title(title), content.into()].spacing(style::SPACE_L))
        .padding(style::CARD_PADDING)
        .style(|_| card_style())
}

/// A titled card with controls aligned to the right of the title.
pub fn card_with_actions<'a>(
    title: &'a str,
    actions: impl Into<Element<'a, Message>>,
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    let header = row![
        card_title(title),
        Space::new().width(Length::Fill),
        actions.into()
    ]
    .align_y(Alignment::Center)
    .spacing(style::SPACE_M);
    container(column![header, content.into()].spacing(style::SPACE_L))
        .padding(style::CARD_PADDING)
        .style(|_| card_style())
}

/// A labelled value with an optional caption below.
pub fn stat(
    label: &str,
    value: String,
    value_color: Color,
    caption: Option<String>,
) -> Element<'_, Message> {
    let mut col = column![
        text(label)
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_SECONDARY),
        text(value).size(style::TEXT_TITLE).color(value_color),
    ]
    .spacing(style::SPACE_XS);
    if let Some(caption) = caption {
        col = col.push(
            text(caption)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        );
    }
    col.into()
}

/// A small pill with tinted background, e.g. "42%" or "Forecast".
pub fn badge<'a>(label: String, color: Color) -> Element<'a, Message> {
    container(text(label).size(style::TEXT_CAPTION).color(color))
        .padding([2, 8])
        .style(move |_| container::Style {
            background: Some(style::tint(color, 0.16).into()),
            border: Border {
                radius: 999.0.into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .into()
}

/// Coloured swatch plus label for chart legends.
pub fn legend_item(color: Color, label: &str) -> Element<'_, Message> {
    row![
        container(Space::new().width(10).height(10)).style(move |_| container::Style {
            background: Some(color.into()),
            border: Border {
                radius: 3.0.into(),
                ..Default::default()
            },
            ..Default::default()
        }),
        text(label)
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_SECONDARY)
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .into()
}

/// Muted centred message for sections without data.
pub fn empty_state(message: &str) -> Element<'_, Message> {
    container(
        text(message)
            .size(style::TEXT_BODY)
            .color(style::TEXT_TERTIARY),
    )
    .padding(style::SPACE_XL)
    .center_x(Length::Fill)
    .into()
}

/// Vertical scroll area with a slim scrollbar beside the content.
pub fn scroll<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    scrollable(content)
        .direction(scrollable::Direction::Vertical(
            scrollable::Scrollbar::new()
                .width(6)
                .scroller_width(6)
                .spacing(style::SPACE_S),
        ))
        .height(Length::Fill)
        .style(|theme, status| {
            let scroller = if matches!(
                status,
                scrollable::Status::Hovered { .. } | scrollable::Status::Dragged { .. }
            ) {
                style::TEXT_TERTIARY
            } else {
                style::BORDER
            };
            let rail = scrollable::Rail {
                background: None,
                border: Border::default(),
                scroller: scrollable::Scroller {
                    background: scroller.into(),
                    border: Border {
                        radius: 3.0.into(),
                        ..Default::default()
                    },
                },
            };
            scrollable::Style {
                vertical_rail: rail,
                horizontal_rail: rail,
                ..scrollable::default(theme, status)
            }
        })
        .into()
}
