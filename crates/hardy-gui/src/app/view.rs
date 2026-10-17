//! Window layout: sidebar, header and the active view.

use iced::{
    Alignment, Border, Color, Element, Length, Shadow, Vector,
    widget::{Space, column, container, row, stack, text},
};

use super::{HardyMonitorApp, Message, ViewMode};
use crate::{
    style::{self},
    time_range::can_step_back,
    views::{
        self, DataProps, InsightsProps, ModelDataProps, NowProps, PlanProps, WeekProps,
        components::secondary_button, schema_notice::SchemaGate,
    },
};

impl HardyMonitorApp {
    /// The whole window: schema notice, or sidebar, header and active view.
    pub fn view(&self) -> Element<'_, Message> {
        if self.schema != SchemaGate::Ready {
            return views::schema_notice::view(&self.schema);
        }
        let content = match self.ui.current_view {
            ViewMode::Now => views::now::view(&self.now_props()),
            ViewMode::Week => views::week::view(WeekProps {
                grid: &self.data.week_grid,
                days: &self.data.week_days,
                range: self.ui.analytics_range,
                low_threshold: self.config.thresholds.low_occupancy_percent,
                high_threshold: self.config.thresholds.high_occupancy_percent,
                heatmap_cache: &self.ui.heatmap_cache,
                heatmap_tooltip_cache: &self.ui.heatmap_tooltip_cache,
                busy_ranges: &self.data.week_busy,
                quiet_ranges: &self.data.week_quiet,
            }),
            ViewMode::Data => {
                let window = self.data_window();
                views::data::view(DataProps {
                    rows: &self.data.data_rows,
                    period: self.ui.data_period,
                    window,
                    can_step_back: can_step_back(window.0, self.data.first_reading),
                    can_step_forward: self.ui.data_offset > 0,
                    scroll: self.ui.data_scroll,
                    timezone: self.schedule.timezone(),
                    low_threshold: self.config.thresholds.low_occupancy_percent,
                    high_threshold: self.config.thresholds.high_occupancy_percent,
                })
            }
            ViewMode::Insights => views::insights::view(InsightsProps {
                trend: self.data.trend,
                stats: self.data.stats.as_ref(),
                busy_ranges: &self.data.busy_ranges,
                quiet_ranges: &self.data.quiet_ranges,
                insights: &self.data.insights,
                low_threshold: self.config.thresholds.low_occupancy_percent,
                high_threshold: self.config.thresholds.high_occupancy_percent,
            }),
            ViewMode::Plan => views::plan::view(PlanProps {
                planner: &self.data.plan,
                now: self.clock.now_utc(),
                schedule: &self.schedule,
                low_threshold: self.config.thresholds.low_occupancy_percent,
                high_threshold: self.config.thresholds.high_occupancy_percent,
                curve_cache: &self.ui.plan_cache,
            }),
            ViewMode::ModelData => views::model_data::view(ModelDataProps {
                timezone: self.schedule.timezone(),
                leaderboard: self.data.forecasting.leaderboard(),
                leaderboard_sort: self.ui.leaderboard_sort,
                leaderboard_family: self.ui.leaderboard_family,
                accuracy: self.data.accuracy.as_ref(),
                run_pending: self.data.forecasting.run_pending(),
                last_run_error: self.data.forecasting.last_error(),
                repair_start: &self.repair.start_date,
                repair_end: &self.repair.end_date,
                repair_running: self.repair.is_running,
                repair_result: self.repair.last_result.as_ref(),
                export_status: self.export_status.as_deref(),
            }),
        };

        let main = container(
            column![self.view_header(), content]
                .spacing(style::SPACE_XL)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .padding([style::SPACE_XL, style::SPACE_XXL]);

        let layout = container(row![self.view_sidebar(), main])
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_| container::Style {
                background: Some(style::BG_APP.into()),
                ..Default::default()
            });

        match &self.export_status {
            Some(status) => stack![layout, toast(status)].into(),
            None => layout.into(),
        }
    }

    fn now_props(&self) -> NowProps<'_> {
        let now = self.clock.now_utc();
        NowProps {
            now,
            schedule: &self.schedule,
            occupancy: self.data.occupancy,
            reading_15_min_ago: self
                .data
                .forecasting
                .reading_near(now - chrono::TimeDelta::minutes(15)),
            last_update: self.data.last_update,
            stale_warning: self.freshness().warning(self.schedule.timezone()),
            low_threshold: self.config.thresholds.low_occupancy_percent,
            high_threshold: self.config.thresholds.high_occupancy_percent,
            quiet_window: self.data.forecasting.quiet_window(),
            forecast: self.data.forecasting.forecasts(),
            forecast_source: self.data.forecasting.forecast_source(),
            forecast_notice: self
                .data
                .forecasting
                .forecast_notice(now, self.schedule.timezone()),
            forecast_choice: self.data.forecasting.choice(),
            forecast_picker: self.data.forecasting.forecast_picker(),
            compare_picker: self.data.forecasting.compare_picker(),
            overlays: self.data.forecasting.overlay_lines(),
            alert_active: self.alerts.is_active(now),
            alert_threshold: self.alerts.threshold(),
            alert_duration: self.alerts.duration(),
            alert_status: self.alerts.status_line(now, &self.schedule),
            history: &self.data.history,
            past_forecasts: &self.data.past_forecasts,
            chart_range: self.ui.chart_range,
            chart_offset: self.ui.chart_offset,
            first_reading: self.data.first_reading,
            custom_start: &self.ui.custom_start,
            custom_end: &self.ui.custom_end,
            chart_cache: &self.ui.chart_cache,
            gauge_cache: &self.ui.gauge_cache,
        }
    }

    fn view_header(&self) -> Element<'_, Message> {
        let tz = self.schedule.timezone();
        let date = self
            .clock
            .now_utc()
            .with_timezone(&tz)
            .format("%A, %-d %B")
            .to_string();

        let (dot, status): (Color, String) = if self.is_updating() {
            (style::ACCENT, "Updating…".to_string())
        } else if let Some(e) = self.errors.latest() {
            (style::DANGER, e.to_string())
        } else if let Some(warning) = self.freshness().warning(tz) {
            (style::WARNING, warning)
        } else {
            match self.data.last_update {
                Some(t) => (
                    style::SUCCESS,
                    format!("Updated {}", t.with_timezone(&tz).format("%H:%M")),
                ),
                None => (style::TEXT_TERTIARY, "Waiting for data".to_string()),
            }
        };

        let status = row![
            container(Space::new().width(8).height(8)).style(move |_| container::Style {
                background: Some(dot.into()),
                border: Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
            text(status)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY),
        ]
        .spacing(style::SPACE_S)
        .align_y(Alignment::Center);

        row![
            column![
                text(self.ui.current_view.title())
                    .size(style::TEXT_TITLE)
                    .color(style::TEXT_PRIMARY),
                text(date)
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_TERTIARY),
            ]
            .spacing(2),
            Space::new().width(Length::Fill),
            container(status).max_width(420),
            secondary_button("↻  Refresh", Message::RefreshNow),
        ]
        .spacing(style::SPACE_L)
        .align_y(Alignment::Center)
        .into()
    }
}

fn toast(message: &str) -> Element<'_, Message> {
    container(
        container(
            text(message)
                .size(style::TEXT_BODY)
                .color(style::TEXT_PRIMARY),
        )
        .padding([12, 20])
        .style(|_| container::Style {
            background: Some(style::BG_ELEVATED.into()),
            border: Border {
                radius: style::RADIUS_CONTROL.into(),
                width: 1.0,
                color: style::BORDER,
            },
            shadow: Shadow {
                color: Color::from_rgba(0.0, 0.0, 0.0, 0.5),
                offset: Vector::new(0.0, 4.0),
                blur_radius: 16.0,
            },
            ..Default::default()
        }),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .align_x(Alignment::Center)
    .align_y(Alignment::End)
    .padding(style::SPACE_XXL)
    .into()
}
