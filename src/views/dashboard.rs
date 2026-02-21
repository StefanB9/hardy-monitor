use chrono::{DateTime, Duration as ChronoDuration, Local, Utc};
use hardy_monitor::{
    PredictionWithConfidence,
    analytics::midnight_local_as_utc,
    db::OccupancyLog,
    schedule::GymSchedule,
    style,
    widgets::{gauge::GaugeWidget, history_chart::HistoryChart},
};
use iced::{
    Alignment, Border, Color, Element, Length, Theme,
    widget::{
        Canvas, Space, button, canvas::Cache, center, checkbox, column, container, row, slider,
        text,
    },
};

use crate::{
    app::Message,
    views::components::{
        card_container, preset_btn, primary_btn_style, secondary_btn_style, styled_input,
    },
};

pub struct DashboardProps<'a> {
    pub occupancy: Option<f64>,
    pub history: &'a [OccupancyLog],
    pub predictions: &'a [(DateTime<Utc>, f64)],
    pub best_time_today: Option<(i32, f64)>,
    pub chart_cache: &'a Cache,
    pub gauge_cache: &'a Cache,
    pub schedule: &'a GymSchedule,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub notification_enabled: bool,
    pub notification_threshold: f64,
    pub history_start_date: &'a str,
    pub history_end_date: &'a str,
    pub history_days_preset: Option<i64>,
    pub ml_predictions: &'a [PredictionWithConfidence],
    pub ml_predictions_simple: &'a [(DateTime<Utc>, f64)],
    pub show_ml_prediction: bool,
    pub ml_has_model: bool,
}

pub fn view(props: DashboardProps<'_>) -> Element<'_, Message> {
    let gauge = Canvas::new(GaugeWidget {
        percentage: props.occupancy.unwrap_or(0.0),
        is_open: props.schedule.is_open(&Local::now()),
        low_threshold: props.low_threshold,
        high_threshold: props.high_threshold,
        cache: props.gauge_cache,
    })
    .width(Length::Fixed(220.0))
    .height(Length::Fixed(220.0));

    let is_checked = props.notification_enabled;
    let active_rail = if is_checked {
        style::ACCENT_BLUE
    } else {
        style::TEXT_MUTED
    };
    let handle_bg = if is_checked {
        style::ACCENT_BLUE
    } else {
        style::TEXT_MUTED
    };
    let text_color = if is_checked {
        style::TEXT_BRIGHT
    } else {
        style::TEXT_MUTED
    };

    let slider_section: Element<'_, Message> = column![
        row![
            text("Threshold:").size(12).color(style::TEXT_MUTED),
            text(format!("{:.0}%", props.notification_threshold))
                .size(12)
                .color(text_color)
        ]
        .spacing(5),
        slider(
            0.0..=60.0,
            props.notification_threshold,
            Message::NotificationThresholdChanged
        )
        .step(5.0)
        .style(move |_: &Theme, _| slider::Style {
            rail: slider::Rail {
                backgrounds: (active_rail.into(), style::BG_DARK.into()),
                width: 4.0,
                border: Border {
                    radius: 2.0.into(),
                    ..Default::default()
                }
            },
            handle: slider::Handle {
                shape: slider::HandleShape::Circle { radius: 8.0 },
                background: handle_bg.into(),
                border_width: 0.0,
                border_color: Color::TRANSPARENT
            }
        })
    ]
    .spacing(5)
    .into();

    let notify_controls = column![
        row![
            checkbox(is_checked)
                .on_toggle(Message::NotificationToggled)
                .size(14)
                .style(move |_theme, _status| checkbox::Style {
                    icon_color: style::TEXT_BRIGHT,
                    background: if is_checked {
                        style::ACCENT_BLUE.into()
                    } else {
                        style::BG_DARK.into()
                    },
                    border: Border {
                        radius: 4.0.into(),
                        width: 1.0,
                        color: style::STROKE_DIM
                    },
                    text_color: None,
                }),
            text("Notify when empty").size(14).color(if is_checked {
                style::TEXT_BRIGHT
            } else {
                style::TEXT_MUTED
            })
        ]
        .spacing(8)
        .align_y(Alignment::Center),
        slider_section
    ]
    .spacing(10)
    .max_width(220);

    let current_card = card_container(column![
        text("Current Status").size(16).color(style::TEXT_MUTED),
        Space::new().height(10),
        center(gauge),
        Space::new().height(20),
        notify_controls
    ]);

    let rec_content = if let Some((hour, avg)) = props.best_time_today {
        column![
            text(format!("Best time on {}s", Local::now().format("%A")))
                .size(16)
                .color(style::TEXT_MUTED),
            Space::new().height(20),
            text(format!("{hour:02}:00"))
                .size(36)
                .color(style::ACCENT_CYAN),
            Space::new().height(10),
            container(
                text(format!("~{avg:.0}% load"))
                    .size(14)
                    .color(style::BG_DARK)
            )
            .padding([6, 12])
            .style(|_| container::Style {
                background: Some(style::ACCENT_CYAN.into()),
                border: Border {
                    radius: 12.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            })
        ]
        .align_x(Alignment::Center)
    } else {
        column![
            text("Best Time Today").size(16).color(style::TEXT_MUTED),
            Space::new().height(20),
            text("Collecting Data...").color(style::TEXT_MUTED)
        ]
        .align_x(Alignment::Center)
    };

    let top_row = row![current_card, card_container(center(rec_content))]
        .spacing(20)
        .height(Length::Fixed(350.0));

    let controls = row![
        preset_btn("Today", 1, props.history_days_preset),
        preset_btn("7D", 7, props.history_days_preset),
        preset_btn("30D", 30, props.history_days_preset),
        Space::new().width(20),
        styled_input(props.history_start_date, Message::HistoryStartDateChanged),
        text("-").color(style::TEXT_MUTED),
        styled_input(props.history_end_date, Message::HistoryEndDateChanged),
        button(text("Go").size(12))
            .on_press(Message::ApplyDateRange)
            .padding([8, 12])
            .style(primary_btn_style),
        Space::new().width(10),
        button(text("Export CSV").size(12))
            .on_press(Message::ExportCsv)
            .padding([8, 12])
            .style(secondary_btn_style)
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let (chart_start, chart_end) = if let Some(days) = props.history_days_preset {
        let local_today = Local::now().date_naive();
        let end_aligned = midnight_local_as_utc(local_today + ChronoDuration::days(1));
        let start_aligned = midnight_local_as_utc(local_today + ChronoDuration::days(1 - days));
        (start_aligned, end_aligned)
    } else {
        match (
            parse_date(props.history_start_date),
            parse_date(props.history_end_date),
        ) {
            (Some(s), Some(e)) => {
                if s == e {
                    (s, s + ChronoDuration::days(1))
                } else {
                    (s, e)
                }
            }
            _ => (Utc::now() - ChronoDuration::days(1), Utc::now()),
        }
    };

    let (active_predictions, confidence_band): (
        &[(DateTime<Utc>, f64)],
        &[PredictionWithConfidence],
    ) = if props.show_ml_prediction && props.ml_has_model && !props.ml_predictions_simple.is_empty()
    {
        (props.ml_predictions_simple, props.ml_predictions)
    } else {
        (props.predictions, &[])
    };

    let chart = Canvas::new(HistoryChart {
        history: props.history,
        predictions: active_predictions,
        confidence_band,
        range_start: chart_start,
        range_end: chart_end,
        cache: props.chart_cache,
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let chart_element = Element::from(chart).map(|_| Message::ChartInteraction);

    let show_ml = props.show_ml_prediction;
    let ml_has_model = props.ml_has_model;
    let ml_text_color = if ml_has_model {
        style::TEXT_BRIGHT
    } else {
        style::TEXT_MUTED
    };
    let ml_checkbox = row![
        {
            let cb = checkbox(show_ml)
                .size(14)
                .style(move |_theme, _status| checkbox::Style {
                    icon_color: style::TEXT_BRIGHT,
                    background: if show_ml {
                        style::ACCENT_CYAN.into()
                    } else {
                        style::BG_DARK.into()
                    },
                    border: Border {
                        radius: 4.0.into(),
                        width: 1.0,
                        color: style::STROKE_DIM,
                    },
                    text_color: None,
                });
            if ml_has_model {
                Element::from(cb.on_toggle(Message::PredictionModeToggled))
            } else {
                Element::from(cb)
            }
        },
        Space::new().width(6),
        text("ML").size(12).color(ml_text_color),
    ]
    .spacing(0)
    .align_y(Alignment::Center);

    column![
        top_row,
        card_container(column![
            row![
                text("Occupancy Trends").size(16).color(style::TEXT_MUTED),
                Space::new().width(Length::Fill),
                controls,
                Space::new().width(16),
                ml_checkbox,
            ]
            .align_y(Alignment::Center),
            Space::new().height(20),
            chart_element
        ])
        .height(Length::Fill)
    ]
    .spacing(20)
    .into()
}

fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .map(midnight_local_as_utc)
}
