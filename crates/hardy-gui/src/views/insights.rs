//! "Insights": summary statistics, busiest and quietest slots, and findings
//! over the last four weeks.

use hardy_core::analytics::{
    Insight, InsightCategory, OccupancyStats, TrendDirection, weekday_short,
};
use iced::{
    Alignment, Border, Color, Element, Length,
    widget::{column, container, row, text},
};

use crate::{
    app::Message,
    style::{self, OccupancyLevel},
    views::components::{badge, card, empty_state, scroll, stat},
};

/// Everything the Insights view shows.
#[derive(Clone, Copy)]
pub struct InsightsProps<'a> {
    pub trend: Option<TrendDirection>,
    pub stats: Option<&'a OccupancyStats>,
    /// `(weekday, hour, average)` of the busiest slots.
    pub peak_hours: &'a [(i32, i32, f64)],
    /// `(weekday, hour, average)` of the quietest slots.
    pub quiet_hours: &'a [(i32, i32, f64)],
    pub insights: &'a [Insight],
    pub low_threshold: f64,
    pub high_threshold: f64,
}

fn trend_tile(trend: Option<TrendDirection>) -> (String, Color) {
    let (label, color) = match trend {
        Some(TrendDirection::Increasing) => ("▲ Getting busier", style::OCC_BUSY),
        Some(TrendDirection::Decreasing) => ("▼ Getting quieter", style::OCC_QUIET),
        Some(TrendDirection::Stable) => ("● Stable", style::TEXT_PRIMARY),
        Some(TrendDirection::Insufficient) | None => ("Not enough data", style::TEXT_TERTIARY),
    };
    (label.to_string(), color)
}

/// How much occupancy differs between hours of the week, i.e. how much
/// choosing the right time pays off.
fn timing_impact(stats: &OccupancyStats) -> &'static str {
    if stats.coefficient_of_variation < 0.3 {
        "Small"
    } else if stats.coefficient_of_variation < 0.5 {
        "Noticeable"
    } else {
        "Large"
    }
}

fn category_icon(category: InsightCategory) -> (&'static str, Color) {
    match category {
        InsightCategory::Trend => ("↗", style::ACCENT),
        InsightCategory::Peak => ("▲", style::OCC_BUSY),
        InsightCategory::QuietTime => ("▼", style::OCC_QUIET),
        InsightCategory::Anomaly => ("!", style::WARNING),
        InsightCategory::DayPattern => ("▦", style::FORECAST),
        InsightCategory::Consistency => ("≈", style::TEXT_SECONDARY),
    }
}

fn tiles<'a>(props: &InsightsProps<'a>) -> Element<'a, Message> {
    let (trend, trend_color) = trend_tile(props.trend);
    let tile = |content: Element<'a, Message>| {
        container(content)
            .padding(style::CARD_PADDING)
            .width(Length::FillPortion(1))
            .style(|_| container::Style {
                background: Some(style::BG_CARD.into()),
                border: Border {
                    color: style::BORDER,
                    width: 1.0,
                    radius: style::RADIUS_CARD.into(),
                },
                ..Default::default()
            })
    };
    let mut tiles = row![tile(stat(
        "Trend",
        trend,
        trend_color,
        Some("vs. the 4 weeks before".to_string()),
    ))]
    .spacing(style::SPACE_L);

    if let Some(stats) = props.stats {
        let level_color = |p| {
            OccupancyLevel::from_percentage(p, props.low_threshold, props.high_threshold).color()
        };
        tiles = tiles
            .push(tile(stat(
                "Average",
                format!("{:.0}%", stats.mean),
                level_color(stats.mean),
                Some(format!("median {:.0}%", stats.median)),
            )))
            .push(tile(stat(
                "Range of hourly averages",
                format!("{:.0} – {:.0}%", stats.min, stats.max),
                style::TEXT_PRIMARY,
                Some("quietest to busiest hour".to_string()),
            )))
            .push(tile(stat(
                "Impact of timing",
                timing_impact(stats).to_string(),
                style::TEXT_PRIMARY,
                Some(format!("hours differ by ± {:.0} pts", stats.std_dev)),
            )));
    }
    tiles.into()
}

fn slot_list<'a>(
    title: &'a str,
    slots: &[(i32, i32, f64)],
    low: f64,
    high: f64,
) -> Element<'a, Message> {
    if slots.is_empty() {
        return card(title, empty_state("Not enough data yet"))
            .width(Length::FillPortion(1))
            .into();
    }
    let mut list = column![].spacing(style::SPACE_S);
    for &(weekday, hour, value) in slots.iter().take(5) {
        list = list.push(
            row![
                text(format!("{} {hour:02}:00", weekday_short(weekday)))
                    .size(style::TEXT_BODY)
                    .color(style::TEXT_PRIMARY)
                    .width(Length::Fill),
                badge(
                    format!("{value:.0}%"),
                    OccupancyLevel::from_percentage(value, low, high).color()
                ),
            ]
            .align_y(Alignment::Center),
        );
    }
    card(title, list).width(Length::FillPortion(1)).into()
}

fn insight_row(insight: &Insight) -> Element<'_, Message> {
    let (icon, color) = category_icon(insight.category);
    row![
        container(text(icon).size(style::TEXT_HEADING).color(color))
            .center_x(32)
            .center_y(32)
            .style(move |_| container::Style {
                background: Some(style::tint(color, 0.14).into()),
                border: Border {
                    radius: style::RADIUS_CONTROL.into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
        column![
            text(&insight.title)
                .size(style::TEXT_BODY)
                .color(style::TEXT_PRIMARY),
            text(&insight.description)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY),
        ]
        .spacing(2)
        .width(Length::Fill),
    ]
    .spacing(style::SPACE_M)
    .align_y(Alignment::Center)
    .into()
}

/// The Insights page.
pub fn view(props: InsightsProps<'_>) -> Element<'_, Message> {
    let (low, high) = (props.low_threshold, props.high_threshold);

    let findings: Element<'_, Message> = if props.insights.is_empty() {
        empty_state("No findings yet — they appear after a few weeks of data.")
    } else {
        let mut list = column![].spacing(style::SPACE_L);
        for insight in props.insights.iter().take(8) {
            list = list.push(insight_row(insight));
        }
        list.into()
    };

    let content = column![
        tiles(&props),
        row![
            slot_list("Busiest times", props.peak_hours, low, high),
            slot_list("Quietest times", props.quiet_hours, low, high),
        ]
        .spacing(style::SPACE_L),
        card("Findings", findings).width(Length::Fill),
    ]
    .spacing(style::SPACE_L);

    scroll(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(cv: f64) -> OccupancyStats {
        OccupancyStats {
            mean: 30.0,
            median: 28.0,
            std_dev: cv * 30.0,
            min: 5.0,
            max: 60.0,
            sample_count: 100,
            coefficient_of_variation: cv,
        }
    }

    #[test]
    fn test_timing_impact_bands() {
        assert_eq!(timing_impact(&stats(0.2)), "Small");
        assert_eq!(timing_impact(&stats(0.4)), "Noticeable");
        assert_eq!(timing_impact(&stats(0.7)), "Large");
    }

    #[test]
    fn test_trend_tile_labels() {
        assert_eq!(
            trend_tile(Some(TrendDirection::Increasing)).0,
            "▲ Getting busier"
        );
        assert_eq!(trend_tile(None).0, "Not enough data");
    }

    #[test]
    fn test_every_category_has_an_icon() {
        for category in [
            InsightCategory::Trend,
            InsightCategory::Peak,
            InsightCategory::QuietTime,
            InsightCategory::Anomaly,
            InsightCategory::DayPattern,
            InsightCategory::Consistency,
        ] {
            assert_ne!(category_icon(category).0, "");
        }
    }
}
