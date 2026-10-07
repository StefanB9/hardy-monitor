//! The sidebar: navigation and the live occupancy summary.

use iced::{
    Alignment, Border, Element, Length,
    widget::{Space, button, column, container, row, text},
};

use super::{HardyMonitorApp, Message, ViewMode};
use crate::{
    freshness::Freshness,
    style::{self, OccupancyLevel},
    views::opening::opening_status,
};

impl HardyMonitorApp {
    pub(super) fn view_sidebar(&self) -> Element<'_, Message> {
        let brand = row![
            container(text("H").size(style::TEXT_HEADING).color(style::BG_APP))
                .center_x(28)
                .center_y(28)
                .style(|_| container::Style {
                    background: Some(style::ACCENT.into()),
                    border: Border {
                        radius: style::RADIUS_CONTROL.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            column![
                text("Hardy Monitor")
                    .size(style::TEXT_HEADING)
                    .color(style::TEXT_PRIMARY),
                text("Gym occupancy")
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_TERTIARY),
            ]
        ]
        .spacing(style::SPACE_M)
        .align_y(Alignment::Center);

        let mut nav = column![].spacing(style::SPACE_XS);
        for mode in ViewMode::ALL {
            nav = nav.push(self.nav_button(mode));
        }

        container(
            column![
                brand,
                Space::new().height(style::SPACE_XXL),
                nav,
                Space::new().height(Length::Fill),
                self.live_summary(),
            ]
            .height(Length::Fill),
        )
        .width(Length::Fixed(self.config.window.sidebar_width.min(240.0)))
        .height(Length::Fill)
        .padding(style::SPACE_L)
        .style(|_| container::Style {
            background: Some(style::BG_SIDEBAR.into()),
            border: Border {
                color: style::BORDER,
                width: 1.0,
                ..Default::default()
            },
            ..Default::default()
        })
        .into()
    }

    pub(super) fn nav_button(&self, mode: ViewMode) -> Element<'_, Message> {
        let active = self.ui.current_view == mode;
        let (icon_color, label_color) = if active {
            (style::ACCENT, style::TEXT_PRIMARY)
        } else {
            (style::TEXT_TERTIARY, style::TEXT_SECONDARY)
        };
        button(
            row![
                text(mode.icon())
                    .size(style::TEXT_HEADING)
                    .color(icon_color)
                    .width(20),
                text(mode.title()).size(style::TEXT_BODY).color(label_color),
            ]
            .spacing(style::SPACE_M)
            .align_y(Alignment::Center),
        )
        .on_press(Message::SwitchView(mode))
        .width(Length::Fill)
        .padding([10, 12])
        .style(move |_, status| {
            let background = if active {
                Some(style::BG_ELEVATED.into())
            } else if matches!(status, button::Status::Hovered) {
                Some(style::tint(style::BG_ELEVATED, 0.5).into())
            } else {
                None
            };
            button::Style {
                background,
                border: Border {
                    radius: style::RADIUS_CONTROL.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
    }

    /// Current occupancy, always visible at the bottom of the sidebar.
    pub(super) fn live_summary(&self) -> Element<'_, Message> {
        let now = self.clock.now_utc();
        let low = self.config.thresholds.low_occupancy_percent;
        let high = self.config.thresholds.high_occupancy_percent;
        let headline: Element<'_, Message> = match self.data.occupancy {
            Some(p) => {
                let level = OccupancyLevel::from_percentage(p, low, high);
                row![
                    text(format!("{p:.0}%"))
                        .size(style::TEXT_TITLE)
                        .color(style::TEXT_PRIMARY),
                    text(level.label())
                        .size(style::TEXT_BODY)
                        .color(level.color()),
                ]
                .spacing(style::SPACE_S)
                .align_y(Alignment::Center)
                .into()
            }
            None => text("–")
                .size(style::TEXT_TITLE)
                .color(style::TEXT_TERTIARY)
                .into(),
        };
        container(
            column![
                text("LIVE")
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_TERTIARY),
                headline,
                match self
                    .data
                    .latest_reading_at
                    .filter(|_| self.freshness() != Freshness::Live)
                {
                    Some(t) => text(format!(
                        "Last reading {}",
                        t.with_timezone(&self.schedule.timezone()).format("%H:%M")
                    ))
                    .size(style::TEXT_CAPTION)
                    .color(style::WARNING),
                    None => text(opening_status(now, &self.schedule))
                        .size(style::TEXT_CAPTION)
                        .color(style::TEXT_SECONDARY),
                },
            ]
            .spacing(style::SPACE_XS),
        )
        .width(Length::Fill)
        .padding(style::SPACE_M)
        .style(|_| container::Style {
            background: Some(style::BG_CARD.into()),
            border: Border {
                color: style::BORDER,
                width: 1.0,
                radius: style::RADIUS_CONTROL.into(),
            },
            ..Default::default()
        })
        .into()
    }
}
