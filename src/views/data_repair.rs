use hardy_monitor::{
    error::AppError,
    repair::{RepairProgress, RepairSummary},
    style,
};
use iced::{
    Alignment, Border, Element, Length,
    widget::{Space, button, column, container, row, text},
};

use crate::{
    app::{Message, RepairPreset},
    views::components::{card_container, primary_btn_style, secondary_btn_style, styled_input},
};

pub struct DataRepairProps<'a> {
    pub start_date: &'a str,
    pub end_date: &'a str,
    pub is_running: bool,
    pub progress: Option<&'a RepairProgress>,
    pub last_result: Option<&'a Result<RepairSummary, AppError>>,
}

pub fn view(props: DataRepairProps<'_>) -> Element<'_, Message> {
    let preset_btn = |label: &str, preset: RepairPreset| {
        button(text(label.to_string()).size(12))
            .on_press(Message::RepairPresetSelected(preset))
            .padding([6, 12])
            .style(secondary_btn_style)
    };

    let date_inputs = row![
        styled_input(props.start_date, Message::RepairStartDateChanged),
        text("to").color(style::TEXT_MUTED).size(14),
        styled_input(props.end_date, Message::RepairEndDateChanged),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let presets = row![
        preset_btn("Last 7 days", RepairPreset::Last7Days),
        preset_btn("Last 30 days", RepairPreset::Last30Days),
        preset_btn("All data", RepairPreset::AllData),
    ]
    .spacing(10);

    let start_button = if props.is_running {
        button(text("Running...").size(14))
            .padding([12, 24])
            .style(|_, _| button::Style {
                background: Some(style::TEXT_MUTED.into()),
                text_color: style::BG_DARK,
                border: Border {
                    radius: 8.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            })
    } else {
        button(text("Start Repair").size(14))
            .on_press(Message::StartRepairJob)
            .padding([12, 24])
            .style(primary_btn_style)
    };

    let progress_section: Element<'_, Message> = if props.is_running {
        if let Some(progress) = props.progress {
            let pct = if progress.total_days > 0 {
                (progress.processed_days as f32 / progress.total_days as f32) * 100.0
            } else {
                0.0
            };
            column![
                text(format!(
                    "Processing: {} (Day {} of {})",
                    progress.current_day,
                    progress.processed_days + 1,
                    progress.total_days
                ))
                .size(14)
                .color(style::TEXT_MUTED),
                Space::new().height(10),
                container(
                    container(
                        Space::new()
                            .width(Length::FillPortion((pct as u16).max(1)))
                            .height(8)
                    )
                    .style(|_| container::Style {
                        background: Some(style::ACCENT_BLUE.into()),
                        border: Border {
                            radius: 4.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                )
                .width(Length::Fill)
                .style(|_| container::Style {
                    background: Some(style::BG_DARK.into()),
                    border: Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
            ]
            .into()
        } else {
            text("Starting repair job...")
                .size(14)
                .color(style::TEXT_MUTED)
                .into()
        }
    } else {
        Space::new().height(0).into()
    };

    let result_section: Element<'_, Message> = if let Some(result) = props.last_result {
        match result {
            Ok(summary) => card_container(column![
                text("Last Repair Results")
                    .size(16)
                    .color(style::ACCENT_GREEN),
                Space::new().height(15),
                row![
                    text("Days processed:").size(14).color(style::TEXT_MUTED),
                    Space::new().width(10),
                    text(summary.days_processed.to_string())
                        .size(14)
                        .color(style::TEXT_BRIGHT),
                ],
                Space::new().height(5),
                row![
                    text("Gaps filled:").size(14).color(style::TEXT_MUTED),
                    Space::new().width(10),
                    text(summary.gaps_filled.to_string())
                        .size(14)
                        .color(style::ACCENT_CYAN),
                ],
                Space::new().height(5),
                row![
                    text("Records deleted:").size(14).color(style::TEXT_MUTED),
                    Space::new().width(10),
                    text(summary.records_deleted.to_string())
                        .size(14)
                        .color(style::ACCENT_ORANGE),
                ],
                Space::new().height(5),
                row![
                    text("Bound entries added:")
                        .size(14)
                        .color(style::TEXT_MUTED),
                    Space::new().width(10),
                    text(summary.boundary_entries_added.to_string())
                        .size(14)
                        .color(style::TEXT_BRIGHT),
                ],
            ])
            .into(),
            Err(e) => card_container(column![
                text("Repair Failed").size(16).color(style::ACCENT_RED),
                Space::new().height(10),
                text(e.to_string()).size(14).color(style::TEXT_MUTED),
            ])
            .into(),
        }
    } else {
        Space::new().height(0).into()
    };

    let description = column![
        text("Repair occupancy data by:")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(8),
        row![
            text("*").color(style::ACCENT_CYAN),
            Space::new().width(8),
            text("Removing data outside opening hours")
                .size(13)
                .color(style::TEXT_MUTED),
        ],
        Space::new().height(4),
        row![
            text("*").color(style::ACCENT_CYAN),
            Space::new().width(8),
            text("Anchoring start and end times to 0%")
                .size(13)
                .color(style::TEXT_MUTED),
        ],
        Space::new().height(4),
        row![
            text("*").color(style::ACCENT_CYAN),
            Space::new().width(8),
            text("Filling gaps up to 5 minutes")
                .size(13)
                .color(style::TEXT_MUTED),
        ],
        Space::new().height(4),
        row![
            text("*").color(style::ACCENT_CYAN),
            Space::new().width(8),
            text("Smoothing outliers and spikes")
                .size(13)
                .color(style::TEXT_MUTED),
        ],
    ];

    card_container(column![
        text("Select Date Range").size(16).color(style::TEXT_BRIGHT),
        Space::new().height(20),
        date_inputs,
        Space::new().height(15),
        presets,
        Space::new().height(25),
        description,
        Space::new().height(25),
        start_button,
        Space::new().height(20),
        progress_section,
        Space::new().height(20),
        result_section,
    ])
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}
