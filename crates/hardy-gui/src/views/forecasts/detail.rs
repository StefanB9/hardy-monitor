//! A model's detail: its live accuracy, the nights it was selected, its
//! nightly backtest error, and its live error by day and by lead time.

use hardy_ml::benchmark::Bucket;
use iced::{
    Element, Length,
    widget::{column, row},
};

use super::{
    ForecastsProps,
    charts::{day_ticks, day_x, lead_label, lead_ticks, line_chart, time_x, x_day, x_time},
};
use crate::{
    app::Message,
    leaderboard::family_label,
    performance::ModelDetail,
    style,
    views::components::{
        caption, card, card_with_actions, empty_state, legend_item, small_button, stat,
    },
    widgets::plot::Series,
};

/// "3 of 28" nights selected for a bucket.
fn selected_text(detail: &ModelDetail) -> String {
    let parts: Vec<String> = Bucket::HORIZONS
        .iter()
        .zip(detail.selected_nights)
        .filter(|(_, n)| *n > 0)
        .map(|(bucket, n)| format!("{} {n}", bucket.label().to_lowercase()))
        .collect();
    if parts.is_empty() {
        "never".to_string()
    } else {
        parts.join(" · ")
    }
}

fn stats<'a>(detail: &ModelDetail) -> Element<'a, Message> {
    let live = detail.live;
    row![
        stat(
            "Live error",
            live.map_or("–".to_string(), |m| format!("{:.1} pts", m.mae)),
            style::TEXT_PRIMARY,
            Some(live.map_or("none logged yet".to_string(), |m| {
                format!("{} forecasts", m.count)
            })),
        ),
        stat(
            "Bias",
            live.map_or("–".to_string(), |m| super::signed_points(m.bias)),
            style::TEXT_PRIMARY,
            Some("+ means too high".to_string()),
        ),
        stat(
            "In band",
            live.and_then(|m| m.coverage)
                .map_or("–".to_string(), |c| format!("{:.0}%", c * 100.0)),
            style::TEXT_PRIMARY,
            Some("target 80%".to_string()),
        ),
        stat(
            "Selected",
            selected_text(detail),
            style::TEXT_PRIMARY,
            Some(format!("nights of {}", detail.nights)),
        ),
    ]
    .spacing(style::SPACE_XXL)
    .into()
}

fn backtest_chart<'a>(props: &ForecastsProps<'a>, detail: &ModelDetail) -> Element<'a, Message> {
    let (Some(first), Some(last)) = (detail.backtest.first(), detail.backtest.last()) else {
        return empty_state("No nightly backtest in this period.");
    };
    let tz = props.timezone;
    let first = first.0.with_timezone(&tz).date_naive();
    let last = last.0.with_timezone(&tz).date_naive();
    let series = Bucket::HORIZONS
        .iter()
        .enumerate()
        .map(|(i, bucket)| Series {
            name: bucket.label().to_string(),
            color: style::COMPARISON[i],
            width: 2.0,
            dashed: false,
            points: detail
                .backtest
                .iter()
                .filter_map(|(at, mae)| {
                    let x = time_x(first, at.with_timezone(&tz).naive_local());
                    Some((x, mae[i + 1]?))
                })
                .collect(),
        })
        .collect();
    let legend = Bucket::HORIZONS
        .iter()
        .enumerate()
        .fold(row![].spacing(style::SPACE_M), |legend, (i, bucket)| {
            legend.push(legend_item(style::COMPARISON[i], bucket.label()))
        });
    column![
        line_chart(
            series,
            day_ticks(first, last),
            move |v| format!("Run {}", x_time(first, v).format("%a %-d %b %H:%M")),
            " pts",
            None,
            &props.caches.backtest,
        ),
        legend
    ]
    .spacing(style::SPACE_S)
    .into()
}

fn live_daily_chart<'a>(props: &ForecastsProps<'a>, detail: &ModelDetail) -> Element<'a, Message> {
    let (Some(first), Some(last)) = (detail.live_daily.first(), detail.live_daily.last()) else {
        return empty_state("No live forecasts scored in this period yet.");
    };
    let first = first.0;
    let series = vec![Series {
        name: "Live".to_string(),
        color: style::FORECAST,
        width: 2.0,
        dashed: false,
        points: detail
            .live_daily
            .iter()
            .map(|(day, mae)| (day_x(first, *day), *mae))
            .collect(),
    }];
    line_chart(
        series,
        day_ticks(first, last.0),
        move |v| x_day(first, v).format("%a %-d %b").to_string(),
        " pts",
        None,
        &props.caches.live_daily,
    )
}

fn lead_time_chart<'a>(props: &ForecastsProps<'a>, detail: &ModelDetail) -> Element<'a, Message> {
    if detail.lead_time.is_empty() {
        return empty_state("No live forecasts scored in this period yet.");
    }
    let series = vec![Series {
        name: "Error".to_string(),
        color: style::ACCENT,
        width: 2.0,
        dashed: false,
        points: detail
            .lead_time
            .iter()
            .map(|(step, mae)| (f64::from(*step), *mae))
            .collect(),
    }];
    line_chart(
        series,
        lead_ticks(),
        lead_label,
        " pts",
        None,
        &props.caches.lead_time,
    )
}

pub(super) fn view<'a>(
    props: &ForecastsProps<'a>,
    detail: &'a ModelDetail,
) -> Element<'a, Message> {
    let id = detail.id;
    let title = id.name();
    let panel = |name: &'static str, chart: Element<'a, Message>| {
        card(name, chart).width(Length::FillPortion(1))
    };
    card_with_actions(
        title,
        row![
            caption(family_label(id.family())),
            small_button("Close", Message::ModelPicked(id)),
        ]
        .spacing(style::SPACE_M)
        .align_y(iced::Alignment::Center),
        column![
            stats(detail),
            row![
                panel("Backtest error by night", backtest_chart(props, detail)),
                panel("Live error by day", live_daily_chart(props, detail)),
                panel("Live error by lead time", lead_time_chart(props, detail)),
            ]
            .spacing(style::SPACE_L),
            caption(format!(
                "Average error in points: the nightly backtest per horizon, and this model's live \
                 forecasts, {}.",
                props.period_label
            )),
        ]
        .spacing(style::SPACE_L),
    )
    .width(Length::Fill)
    .into()
}

#[cfg(test)]
mod tests {
    use hardy_ml::suite::ModelId;

    use super::*;

    fn detail(selected_nights: [usize; 3]) -> ModelDetail {
        ModelDetail {
            id: ModelId::Naive,
            backtest: Vec::new(),
            selected_nights,
            nights: 28,
            live_daily: Vec::new(),
            lead_time: Vec::new(),
            live: None,
        }
    }

    #[test]
    fn test_selected_text_lists_buckets_with_nights() {
        assert_eq!(selected_text(&detail([0, 0, 0])), "never");
        assert_eq!(selected_text(&detail([3, 0, 5])), "next hour 3 · 3–24 h 5");
    }
}
