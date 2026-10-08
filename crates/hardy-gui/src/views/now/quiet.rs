//! "Next quiet hour" card with the upcoming hours.

use chrono::{DateTime, Utc};
use hardy_core::Tz;
use iced::{
    Alignment, Border, Element, Length,
    widget::{Space, column, container, row, text},
};

use super::NowProps;
use crate::{
    app::Message,
    quiet_window::Source,
    style::{self, OccupancyLevel},
    views::components::{badge, card_with_actions, empty_state},
};

/// Forecast hours listed under the window.
const UPCOMING_ROWS: usize = 4;

/// "Today", "Tomorrow" or the weekday of `t`, relative to `now`.
fn day_label(t: DateTime<Utc>, now: DateTime<Utc>, tz: Tz) -> String {
    let (day, today) = (
        t.with_timezone(&tz).date_naive(),
        now.with_timezone(&tz).date_naive(),
    );
    if day == today {
        "Today".to_string()
    } else if today.succ_opt() == Some(day) {
        "Tomorrow".to_string()
    } else {
        day.format("%A").to_string()
    }
}

/// Horizontal bar, `value`% of the occupancy axis.
fn bar<'a>(value: f64, color: iced::Color) -> Element<'a, Message> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let filled = (value.clamp(1.0, 100.0)).round() as u16;
    let rounded = |color: iced::Color| {
        move |_: &iced::Theme| container::Style {
            background: Some(color.into()),
            border: Border {
                radius: 3.0.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    };
    container(
        row![
            container(Space::new().height(6))
                .width(Length::FillPortion(filled))
                .style(rounded(color)),
            Space::new().width(Length::FillPortion(100 - filled.min(99))),
        ]
        .width(Length::Fill),
    )
    .width(Length::Fill)
    .style(rounded(style::BG_ELEVATED))
    .into()
}

pub(super) fn card<'a>(props: &NowProps<'a>) -> container::Container<'a, Message> {
    let tz = props.schedule.timezone();
    let (low, high) = (props.low_threshold, props.high_threshold);

    let Some(window) = props.quiet_window else {
        return card_with_actions(
            "Next quiet hour",
            Space::new(),
            empty_state("Collecting data…"),
        )
        .height(Length::Fill);
    };

    let level = OccupancyLevel::from_percentage(window.expected, low, high);
    let source = match window.source {
        Source::Forecast if props.has_model => "Forecast from the trained model",
        Source::Forecast => "Baseline forecast (no model yet)",
        Source::Averages => "Typical for this weekday",
    };
    let headline = column![
        text(format!(
            "{} – {}",
            window.start.with_timezone(&tz).format("%H:%M"),
            window.end.with_timezone(&tz).format("%H:%M")
        ))
        .size(32)
        .color(style::TEXT_PRIMARY),
        row![
            text(format!("~{:.0}% expected", window.expected))
                .size(style::TEXT_BODY)
                .color(level.color()),
            text(format!("· {source}"))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        ]
        .spacing(style::SPACE_S)
        .align_y(Alignment::Center),
    ]
    .spacing(style::SPACE_XS);

    let mut upcoming = column![
        text("Next hours")
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_TERTIARY)
    ]
    .spacing(6);
    let next: Vec<_> = props
        .forecast
        .iter()
        .filter(|p| p.timestamp > props.now && props.schedule.is_open(&p.timestamp))
        .take(UPCOMING_ROWS)
        .collect();
    if next.is_empty() {
        let reason = if props.schedule.is_open(&props.now) {
            "Closing soon"
        } else {
            "Forecasts start when the gym opens"
        };
        upcoming = upcoming.push(
            text(reason)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY),
        );
    }
    for p in next {
        let color = style::occupancy_color(p.predicted_value, low, high);
        upcoming = upcoming.push(
            row![
                text(p.timestamp.with_timezone(&tz).format("%H:%M").to_string())
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_SECONDARY)
                    .width(44),
                bar(p.predicted_value, color),
                text(format!("{:.0}%", p.predicted_value))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_PRIMARY)
                    .width(36)
                    .align_x(Alignment::End),
            ]
            .spacing(style::SPACE_S)
            .align_y(Alignment::Center),
        );
    }

    card_with_actions(
        "Next quiet hour",
        badge(day_label(window.start, props.now, tz), style::ACCENT),
        column![headline, upcoming].spacing(style::SPACE_L),
    )
    .height(Length::Fill)
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{TimeDelta, TimeZone};

    use super::*;

    #[test]
    fn test_day_label_relative_to_now() -> Result<()> {
        let tz = hardy_core::GymSchedule::default().timezone();
        let now = tz
            .with_ymd_and_hms(2024, 6, 17, 22, 0, 0)
            .single()
            .context("valid time")?
            .with_timezone(&Utc);
        assert_eq!(day_label(now + TimeDelta::minutes(30), now, tz), "Today");
        assert_eq!(day_label(now + TimeDelta::hours(12), now, tz), "Tomorrow");
        assert_eq!(day_label(now + TimeDelta::days(2), now, tz), "Wednesday");
        Ok(())
    }
}
