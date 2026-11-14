//! The Models page: the leaderboard, the picked model's detail, and every
//! model's backtest error against its live one.

use iced::{
    Element, Length,
    widget::{canvas, column},
};

use super::{ForecastsProps, detail, leaderboard};
use crate::{
    app::Message,
    performance::backtest_vs_live,
    style,
    views::components::{caption, card, empty_state},
    widgets::plot::{Dot, ScatterPlot},
};

fn scatter<'a>(props: &ForecastsProps<'a>) -> Element<'a, Message> {
    let points = props
        .performance
        .map_or_else(Vec::new, |p| backtest_vs_live(props.leaderboard, &p.live));
    let body: Element<'a, Message> = if points.is_empty() {
        empty_state(
            "Shows up once models have both a backtest and live forecasts scored in this period.",
        )
    } else {
        let dots = points
            .iter()
            .map(|p| Dot {
                x: p.backtest,
                y: p.live,
                label: p.id.name().to_string(),
                color: if p.selected {
                    style::FORECAST
                } else {
                    style::ACCENT
                },
                large: p.selected || props.picked == Some(p.id),
            })
            .collect();
        column![
            canvas::Canvas::new(ScatterPlot {
                dots,
                x_title: "Backtest",
                y_title: "Live",
                suffix: " pts",
                cache: &props.caches.scatter,
            })
            .width(Length::Fill)
            .height(320),
            caption(
                "Error an hour ahead, in points. Above the line a model does worse live than in \
                 the nightly backtest; the model in use for the next hour is purple.",
            ),
        ]
        .spacing(style::SPACE_S)
        .into()
    };
    card("Backtest vs live", body).width(Length::Fill).into()
}

pub(super) fn view<'a>(props: &ForecastsProps<'a>) -> Element<'a, Message> {
    let mut page = column![leaderboard::card(props)].spacing(style::SPACE_L);
    if let Some(detail) = props.detail {
        page = page.push(detail::view(props, detail));
    }
    page.push(scatter(props)).into()
}
