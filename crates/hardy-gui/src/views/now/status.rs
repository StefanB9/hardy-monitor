//! Live occupancy card.

use iced::{
    Alignment, Length,
    widget::{Canvas, column, container, row, text},
};

use super::NowProps;
use crate::{
    app::Message,
    style,
    views::{components::card as titled_card, opening::opening_status},
    widgets::gauge::GaugeWidget,
};

/// Change from 15 minutes ago, e.g. ("▲ 4 pts in 15 min", busier colour).
fn trend_line(current: f64, earlier: f64) -> (String, iced::Color) {
    let delta = current - earlier;
    if delta.abs() < 1.0 {
        ("Steady over 15 min".to_string(), style::TEXT_SECONDARY)
    } else if delta > 0.0 {
        (format!("▲ {delta:.0} pts in 15 min"), style::OCC_BUSY)
    } else {
        (format!("▼ {:.0} pts in 15 min", -delta), style::OCC_QUIET)
    }
}

pub(super) fn card<'a>(props: &NowProps<'a>) -> container::Container<'a, Message> {
    let is_open = props.schedule.is_open(&props.now);
    let gauge = Canvas::new(GaugeWidget {
        percentage: props.occupancy,
        is_open,
        low_threshold: props.low_threshold,
        high_threshold: props.high_threshold,
        cache: props.gauge_cache,
    })
    .width(Length::Fixed(176.0))
    .height(Length::Fixed(168.0));

    let mut details = row![
        text(opening_status(props.now, props.schedule))
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_SECONDARY)
    ]
    .spacing(style::SPACE_M);
    if is_open && let (Some(current), Some(earlier)) = (props.occupancy, props.reading_15_min_ago) {
        let (line, color) = trend_line(current, earlier);
        details = details.push(text(line).size(style::TEXT_CAPTION).color(color));
    }

    titled_card(
        "Live occupancy",
        column![
            container(gauge).center_x(Length::Fill),
            container(details.wrap()).center_x(Length::Fill),
        ]
        .spacing(style::SPACE_S)
        .align_x(Alignment::Center),
    )
    .height(Length::Fill)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_line_directions() {
        assert_eq!(trend_line(30.0, 30.4).0, "Steady over 15 min");
        assert_eq!(trend_line(34.0, 30.0).0, "▲ 4 pts in 15 min");
        assert_eq!(trend_line(25.0, 30.0).0, "▼ 5 pts in 15 min");
    }
}
