//! Misses & days: whether the quiet hours picked turned out quiet, the
//! worst days and biggest misses, and one day's forecast against the
//! readings.

use chrono::NaiveDate;
use hardy_core::accuracy::{DecisionSums, QUIET_TOLERANCE};
use iced::{
    Alignment, Element, Length,
    widget::{button, column, row, text},
};

use super::{
    ForecastsProps,
    charts::{day_ticks, day_x, line_chart, x_day},
    signed_points,
};
use crate::{
    app::Message,
    performance::{DaySlot, DecisionDay, Performance, day_slots},
    style,
    views::components::{
        caption, card, card_with_actions, empty_state, legend_item, small_button, stat,
    },
    widgets::plot::Series,
};

/// A share as a whole percentage, or a dash.
fn percent(share: Option<f64>) -> String {
    share.map_or("–".to_string(), |s| format!("{:.0}%", s * 100.0))
}

/// Green when most picks were close to the quietest hour.
fn hit_color(rate: Option<f64>) -> iced::Color {
    match rate {
        Some(r) if r >= 0.75 => style::SUCCESS,
        Some(r) if r >= 0.5 => style::WARNING,
        Some(_) => style::DANGER,
        None => style::TEXT_PRIMARY,
    }
}

fn guide_stats<'a>(title: &'a str, sums: &DecisionSums, what: &'a str) -> Element<'a, Message> {
    let body: Element<'a, Message> = if sums.count == 0 {
        empty_state("Nothing to judge in this period yet.")
    } else {
        row![
            stat(
                "Turned out quiet",
                percent(sums.hit_rate()),
                hit_color(sums.hit_rate()),
                Some(format!("within {QUIET_TOLERANCE:.0} pts of the quietest")),
            ),
            stat(
                "Exactly right",
                percent(sums.exact_rate()),
                style::TEXT_PRIMARY,
                Some("the quietest hour itself".to_string()),
            ),
            stat(
                "Points lost",
                sums.mean_lost()
                    .map_or("–".to_string(), |l| format!("{l:.1}")),
                style::TEXT_PRIMARY,
                Some("busier than the quietest, on average".to_string()),
            ),
            stat(
                "Picks",
                sums.count.to_string(),
                style::TEXT_PRIMARY,
                Some(what.to_string()),
            ),
        ]
        .spacing(style::SPACE_XXL)
        .into()
    };
    column![
        text(title)
            .size(style::TEXT_HEADING)
            .color(style::TEXT_PRIMARY),
        body
    ]
    .spacing(style::SPACE_M)
    .width(Length::FillPortion(1))
    .into()
}

fn lost_line(name: &str, color: iced::Color, dashed: bool, points: Vec<(f64, f64)>) -> Series {
    Series {
        name: name.to_string(),
        color,
        width: 2.0,
        dashed,
        points,
    }
}

fn decisions<'a>(props: &ForecastsProps<'a>, performance: &Performance) -> Element<'a, Message> {
    let quality = &performance.decisions;
    let mut body = column![
        row![
            guide_stats("Forecast · next quiet hour", &quality.forecast, "hourly",),
            guide_stats(
                "Plain averages · best time of day",
                &quality.averages,
                "daily"
            ),
        ]
        .spacing(style::SPACE_XL),
    ]
    .spacing(style::SPACE_L);
    if let (Some(first), Some(last)) = (quality.daily.first(), quality.daily.last()) {
        let (first, last) = (first.day, last.day);
        let points = |pick: fn(&DecisionDay) -> Option<f64>| {
            quality
                .daily
                .iter()
                .filter_map(|d| Some((day_x(first, d.day), pick(d)?)))
                .collect::<Vec<_>>()
        };
        let series = vec![
            lost_line("Forecast", style::FORECAST, false, points(|d| d.forecast)),
            lost_line(
                "Averages",
                style::TEXT_SECONDARY,
                true,
                points(|d| d.averages),
            ),
        ];
        body = body.push(line_chart(
            series,
            day_ticks(first, last),
            move |v| x_day(first, v).format("%a %-d %b").to_string(),
            " pts",
            None,
            &props.caches.decisions_daily,
        ));
        body = body.push(
            row![
                legend_item(style::FORECAST, "Forecast"),
                legend_item(style::TEXT_SECONDARY, "Plain averages"),
            ]
            .spacing(style::SPACE_M),
        );
    }
    body = body.push(Element::from(caption(
        "Points lost: how much busier the picked hour was than the quietest hour that happened. \
         The forecast picks the quietest hour of its next 6 hours every hour; plain averages (the \
         same slots over the four weeks before) pick the quietest hour of each day, as the Plan \
         tab does for days beyond the forecast.",
    )));
    card("Did the quiet hour turn out quiet?", body)
        .width(Length::Fill)
        .into()
}

/// A row that charts its day when clicked.
fn day_row<'a>(
    day: NaiveDate,
    picked: bool,
    cells: impl IntoIterator<Item = (String, u16)>,
) -> Element<'a, Message> {
    let mut line = row![].spacing(style::SPACE_S).align_y(Alignment::Center);
    for (value, portion) in cells {
        line = line.push(
            text(value)
                .size(style::TEXT_CAPTION)
                .width(Length::FillPortion(portion)),
        );
    }
    button(line)
        .on_press(Message::PerformanceDayPicked(day))
        .padding([4, 6])
        .width(Length::Fill)
        .style(move |_, status| button::Style {
            text_color: if picked {
                style::ACCENT
            } else if status == button::Status::Hovered {
                style::TEXT_PRIMARY
            } else {
                style::TEXT_SECONDARY
            },
            background: (picked || status == button::Status::Hovered)
                .then_some(iced::Background::Color(style::BG_ELEVATED)),
            border: iced::Border::default().rounded(style::RADIUS_CONTROL),
            ..Default::default()
        })
        .into()
}

fn header<'a>(cells: &[(&'a str, u16)]) -> Element<'a, Message> {
    cells
        .iter()
        .fold(
            row![].spacing(style::SPACE_S).padding([0, 6]),
            |r, (label, portion)| {
                r.push(
                    text(*label)
                        .size(style::TEXT_CAPTION)
                        .color(style::TEXT_TERTIARY)
                        .width(Length::FillPortion(*portion)),
                )
            },
        )
        .into()
}

fn worst_days<'a>(props: &ForecastsProps<'a>, performance: &Performance) -> Element<'a, Message> {
    let mut list = column![header(&[
        ("Day", 3),
        ("Error", 2),
        ("Bias", 2),
        ("Averages", 2)
    ])];
    for d in &performance.worst_days {
        list = list.push(day_row(
            d.day,
            props.day == Some(d.day),
            [
                (d.day.format("%a %-d %b").to_string(), 3),
                (format!("{:.1} pts", d.mae), 2),
                (signed_points(d.bias), 2),
                (
                    d.baseline
                        .map_or("–".to_string(), |b| format!("{b:.1} pts")),
                    2,
                ),
            ],
        ));
    }
    card(
        "Worst days",
        column![
            list,
            caption("Average error an hour ahead; click a day to chart it.")
        ]
        .spacing(style::SPACE_M),
    )
    .width(Length::FillPortion(1))
    .into()
}

fn biggest_misses<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let tz = props.timezone;
    let mut list = column![header(&[
        ("When", 3),
        ("Forecast", 2),
        ("Reading", 2),
        ("Off by", 2)
    ])];
    for s in &performance.misses {
        let local = s.target.with_timezone(&tz);
        let day = local.date_naive();
        list = list.push(day_row(
            day,
            props.day == Some(day),
            [
                (local.format("%a %-d %b %H:%M").to_string(), 3),
                (format!("{:.0}%", s.pair.forecast), 2),
                (format!("{:.0}%", s.pair.actual), 2),
                (signed_points(s.pair.forecast - s.pair.actual), 2),
            ],
        ));
    }
    card(
        "Biggest misses",
        column![
            list,
            caption("Single forecasts an hour ahead; click one to chart its day.")
        ]
        .spacing(style::SPACE_M),
    )
    .width(Length::FillPortion(1))
    .into()
}

fn series(name: &str, color: iced::Color, dashed: bool, points: Vec<(f64, f64)>) -> Series {
    Series {
        name: name.to_string(),
        color,
        width: if dashed { 1.5 } else { 2.0 },
        dashed,
        points,
    }
}

fn day_chart<'a>(
    props: &ForecastsProps<'a>,
    day: NaiveDate,
    slots: &[DaySlot],
) -> Element<'a, Message> {
    let lines = vec![
        series(
            "Reading",
            style::TEXT_PRIMARY,
            false,
            slots.iter().map(|s| (s.hour, s.actual)).collect(),
        ),
        series(
            "Forecast",
            style::FORECAST,
            false,
            slots.iter().map(|s| (s.hour, s.forecast)).collect(),
        ),
        series(
            "Averages",
            style::TEXT_SECONDARY,
            true,
            slots
                .iter()
                .filter_map(|s| Some((s.hour, s.baseline?)))
                .collect(),
        ),
    ];
    let ticks = (0..24)
        .step_by(2)
        .map(|h| (f64::from(h), format!("{h:02}")))
        .filter(|(h, _)| {
            slots.first().is_some_and(|s| *h >= s.hour.floor())
                && slots.last().is_some_and(|s| *h <= s.hour.ceil())
        })
        .collect();
    let in_band = slots
        .iter()
        .filter(|s| s.band.is_some_and(|(l, h)| (l..=h).contains(&s.actual)))
        .count();
    let banded = slots.iter().filter(|s| s.band.is_some()).count();
    #[allow(clippy::cast_precision_loss)]
    let mae = slots
        .iter()
        .map(|s| (s.forecast - s.actual).abs())
        .sum::<f64>()
        / slots.len().max(1) as f64;
    let summary = format!(
        "{} · off by {mae:.1} pts on average{}",
        day.format("%A %-d %B %Y"),
        if banded > 0 {
            format!(" · {in_band} of {banded} readings in the band")
        } else {
            String::new()
        }
    );
    card_with_actions(
        "The day",
        small_button("Close", Message::PerformanceDayPicked(day)),
        column![
            text(summary)
                .size(style::TEXT_BODY)
                .color(style::TEXT_PRIMARY),
            line_chart(
                lines,
                ticks,
                |h| {
                    #[allow(clippy::cast_possible_truncation)]
                    let minutes = (h * 60.0).round() as i64;
                    format!("{:02}:{:02}", minutes / 60, minutes % 60)
                },
                "%",
                None,
                &props.caches.day,
            ),
            row![
                legend_item(style::TEXT_PRIMARY, "Reading"),
                legend_item(style::FORECAST, "Forecast an hour ahead"),
                legend_item(style::TEXT_SECONDARY, "Plain averages"),
            ]
            .spacing(style::SPACE_M),
        ]
        .spacing(style::SPACE_S),
    )
    .width(Length::Fill)
    .into()
}

pub(super) fn view<'a>(
    props: &ForecastsProps<'a>,
    performance: &'a Performance,
) -> Element<'a, Message> {
    let mut page = column![decisions(props, performance)].spacing(style::SPACE_L);
    if performance.scored.is_empty() {
        return page
            .push(empty_state("Nothing scored in this period yet."))
            .into();
    }
    page = page.push(
        row![
            worst_days(props, performance),
            biggest_misses(props, performance)
        ]
        .spacing(style::SPACE_L),
    );
    if let Some(day) = props.day {
        let slots = day_slots(&performance.scored, day, props.timezone);
        if !slots.is_empty() {
            page = page.push(day_chart(props, day, &slots));
        }
    }
    page.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percent_and_hit_colors() {
        assert_eq!(percent(Some(0.756)), "76%");
        assert_eq!(percent(None), "–");
        assert_eq!(hit_color(Some(0.8)), style::SUCCESS);
        assert_eq!(hit_color(Some(0.6)), style::WARNING);
        assert_eq!(hit_color(Some(0.2)), style::DANGER);
    }
}
