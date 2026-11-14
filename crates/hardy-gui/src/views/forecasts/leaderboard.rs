//! The forecast models card: every model's backtest or live error per
//! horizon and where it stands; a model's name opens its detail.

use chrono::{DateTime, Utc};
use hardy_core::Tz;
use hardy_ml::{
    benchmark::Bucket,
    suite::{Family, ModelId},
};
use iced::{
    Alignment, Element, Length,
    widget::{button, column, row, text},
};

use super::ForecastsProps;
use crate::{
    app::Message,
    leaderboard::{Row, Sort, SortKey, Standing, family_label},
    performance::{Source, live_board},
    style,
    views::components::{badge, caption, card_with_actions, empty_state, segmented, small_button},
};

/// Width of a number column.
const NUMBER: f32 = 76.0;
/// Shares of the remaining width: names with their family tag need more
/// room than the status.
const NAME: u16 = 5;
const STATUS: u16 = 3;

/// The family chips: every model, or one family.
fn family_filter(selected: Option<Family>) -> Element<'static, Message> {
    let mut options: Vec<(&'static str, Option<Family>)> =
        Vec::with_capacity(Family::ALL.len() + 1);
    options.push(("All", None));
    options.extend(Family::ALL.iter().map(|&f| (family_label(f), Some(f))));
    segmented(&options, selected, Message::LeaderboardFiltered)
}

/// A column title that sorts by its column; the active one shows the
/// direction.
fn sort_header(
    label: &'static str,
    key: SortKey,
    sort: Sort,
    width: Length,
    align: Alignment,
) -> Element<'static, Message> {
    let active = sort.key == key;
    let title = match (active, sort.descending) {
        (false, _) => label.to_string(),
        (true, false) => format!("{label} ↑"),
        (true, true) => format!("{label} ↓"),
    };
    button(
        text(title)
            .size(style::TEXT_CAPTION)
            .width(Length::Fill)
            .align_x(align),
    )
    .on_press(Message::LeaderboardSorted(key))
    .padding([2, 0])
    .width(width)
    .style(move |_, status| button::Style {
        text_color: if active || status == button::Status::Hovered {
            style::TEXT_PRIMARY
        } else {
            style::TEXT_TERTIARY
        },
        ..Default::default()
    })
    .into()
}

/// A bucket's column title; live forecasts reach only 6 h ahead.
fn bucket_label(bucket: Bucket, source: Source) -> &'static str {
    match (bucket, source) {
        (Bucket::ThreeToDay, Source::Live) => "3–6 h",
        _ => bucket.label(),
    }
}

fn header(sort: Sort, source: Source) -> Element<'static, Message> {
    let mut line = row![sort_header(
        "Model",
        SortKey::Name,
        sort,
        Length::FillPortion(NAME),
        Alignment::Start
    )]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center);
    for bucket in [
        Bucket::NextHour,
        Bucket::OneToThreeHours,
        Bucket::ThreeToDay,
        Bucket::Overall,
    ] {
        line = line.push(sort_header(
            bucket_label(bucket, source),
            SortKey::Error(bucket),
            sort,
            Length::Fixed(NUMBER),
            Alignment::End,
        ));
    }
    line.push(sort_header(
        "In band",
        SortKey::Coverage,
        sort,
        Length::Fixed(NUMBER),
        Alignment::End,
    ))
    .push(caption("Status").width(Length::FillPortion(STATUS)))
    .into()
}

fn number(value: Option<f64>, selected: bool) -> Element<'static, Message> {
    let (label, color) = match value {
        Some(v) if selected => (format!("★ {v:.1}"), style::FORECAST),
        Some(v) => (format!("{v:.1}"), style::TEXT_PRIMARY),
        None => ("–".to_string(), style::TEXT_TERTIARY),
    };
    text(label)
        .size(style::TEXT_CAPTION)
        .color(color)
        .width(NUMBER)
        .align_x(Alignment::End)
        .into()
}

/// The status column: how much data a fitted model saw and when it was
/// last fitted, or why it is not ready.
fn standing(
    standing: &Standing,
    fitted_at: Option<DateTime<Utc>>,
    timezone: Tz,
) -> (String, iced::Color) {
    let fitted = |days: i64| match fitted_at {
        Some(at) => format!(
            "{days} d of data · fit {}",
            at.with_timezone(&timezone).format("%a %H:%M")
        ),
        None => format!("{days} d of data"),
    };
    match standing {
        Standing::Ready { days } => (fitted(*days), style::TEXT_SECONDARY),
        Standing::Learning { days, forecasts } => (
            format!("{} · {forecasts} test forecasts so far", fitted(*days)),
            style::TEXT_SECONDARY,
        ),
        Standing::Waiting(message) => (message.clone(), style::TEXT_TERTIARY),
        Standing::Failed(message) => (format!("failed: {message}"), style::DANGER),
        Standing::Pending => ("not run yet".to_string(), style::TEXT_TERTIARY),
    }
}

/// A model's name: a link that opens (or closes) its detail.
fn name(r: &Row, picked: bool) -> Element<'static, Message> {
    let id: ModelId = r.id;
    button(
        row![
            text(r.name).size(style::TEXT_CAPTION),
            text(family_label(r.family))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        ]
        .spacing(style::SPACE_S),
    )
    .on_press(Message::ModelPicked(id))
    .padding(0)
    .width(Length::FillPortion(NAME))
    .style(move |_, status| button::Style {
        text_color: if picked {
            style::ACCENT
        } else if status == button::Status::Hovered {
            style::TEXT_PRIMARY
        } else {
            style::TEXT_SECONDARY
        },
        ..Default::default()
    })
    .into()
}

fn line(r: &Row, timezone: Tz, picked: bool) -> Element<'static, Message> {
    let is_selected = |bucket: Bucket| r.selected.contains(&bucket);
    let (status, status_color) = standing(&r.standing, r.fitted_at, timezone);
    row![
        name(r, picked),
        number(r.mae[1], is_selected(Bucket::NextHour)),
        number(r.mae[2], is_selected(Bucket::OneToThreeHours)),
        number(r.mae[3], is_selected(Bucket::ThreeToDay)),
        number(r.mae[0], false),
        text(
            r.coverage
                .map_or("–".to_string(), |c| format!("{:.0}%", c * 100.0))
        )
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_SECONDARY)
        .width(NUMBER)
        .align_x(Alignment::End),
        text(status)
            .size(style::TEXT_CAPTION)
            .color(status_color)
            .width(Length::FillPortion(STATUS)),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .into()
}

pub(super) fn card<'a>(props: &ForecastsProps<'a>) -> Element<'a, Message> {
    let run: Element<'a, Message> = if props.run_pending {
        badge("Run requested…".to_string(), style::WARNING)
    } else {
        small_button("Run now", Message::ForecastRunRequested).into()
    };
    let sources: Vec<(&str, Source)> = Source::ALL.iter().map(|s| (s.label(), *s)).collect();
    let actions = row![
        segmented(&sources, props.source, Message::LeaderboardSourceSelected),
        run
    ]
    .spacing(style::SPACE_M)
    .align_y(Alignment::Center);

    let live;
    let board = match (props.source, props.performance) {
        (Source::Live, Some(performance)) => {
            live = live_board(props.leaderboard, &performance.live);
            &live
        }
        _ => props.leaderboard,
    };
    let mut body = column![].spacing(style::SPACE_M);
    match (props.source, &board.period, board.finished_at) {
        (Source::Live, _, _) => {
            body = body.push(caption(format!(
                "Live forecasts logged every full hour, {}",
                props.period_label
            )));
        }
        (Source::Backtest, Some(period), Some(finished)) => {
            body = body.push(caption(format!(
                "Backtest {period} · run {}",
                finished.with_timezone(&props.timezone).format("%a %H:%M")
            )));
        }
        (Source::Backtest, _, _) => {
            body = body.push(empty_state(
                "No benchmark yet. The forecaster fits and tests every model each night, and \
                 right away when a model is new; the table fills in after its first run.",
            ));
        }
    }
    let mut table = column![header(props.leaderboard_sort, props.source)].spacing(style::SPACE_XS);
    for r in board.view(props.leaderboard_sort, props.leaderboard_family) {
        table = table.push(line(r, props.timezone, props.picked == Some(r.id)));
    }
    let note = match props.source {
        Source::Backtest => {
            "Average error in percentage points over the backtest; click a column to sort, a name \
             for the model's detail. ★ marks the model shown for that horizon. In band: share of \
             actual values inside the 80 % band."
        }
        Source::Live => {
            "Average error in percentage points of the forecasts the models made live; – where a \
             model has none in the period yet. Click a name for the model's detail."
        }
    };
    body = body
        .push(family_filter(props.leaderboard_family))
        .push(table)
        .push(caption(note));
    if let Some(error) = props.last_run_error {
        body = body.push(
            text(format!("Last run failed: {error}"))
                .size(style::TEXT_CAPTION)
                .color(style::WARNING),
        );
    }
    card_with_actions("Forecast models", actions, body)
        .width(Length::Fill)
        .into()
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_live_columns_reach_six_hours() {
        assert_eq!(bucket_label(Bucket::ThreeToDay, Source::Live), "3–6 h");
        assert_eq!(bucket_label(Bucket::ThreeToDay, Source::Backtest), "3–24 h");
        assert_eq!(bucket_label(Bucket::NextHour, Source::Live), "Next hour");
    }

    #[test]
    fn test_status_shows_data_and_last_fit_in_gym_time() {
        let tz: Tz = "Europe/Berlin".parse().unwrap_or(Tz::UTC);
        // Wednesday 21:36 UTC is 23:36 in Berlin in summer.
        let at = Utc.with_ymd_and_hms(2024, 6, 5, 21, 36, 0).single();
        let (ready, _) = standing(&Standing::Ready { days: 53 }, at, tz);
        assert_eq!(ready, "53 d of data · fit Wed 23:36");
        let (learning, _) = standing(
            &Standing::Learning {
                days: 9,
                forecasts: 10,
            },
            at,
            tz,
        );
        assert_eq!(
            learning,
            "9 d of data · fit Wed 23:36 · 10 test forecasts so far"
        );
        let (unknown, _) = standing(&Standing::Ready { days: 3 }, None, tz);
        assert_eq!(unknown, "3 d of data");
        let (failed, color) = standing(&Standing::Failed("boom".to_string()), at, tz);
        assert_eq!(failed, "failed: boom");
        assert_eq!(color, style::DANGER);
    }
}
