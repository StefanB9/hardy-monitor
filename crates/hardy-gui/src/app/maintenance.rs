//! Export and data repair messages.

use std::time::Duration;

use hardy_core::{error::AppError, repair::DataRepairer};
use iced::Task;

use super::{HardyMonitorApp, Message, RepairPreset, loads::ErrorSource};
use crate::time_range::parse_date;

impl HardyMonitorApp {
    /// Export and data repair.
    pub(super) fn update_maintenance(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ExportCsv => {
                self.export_status = Some("Exporting…".to_string());
                self.export_csv()
            }
            Message::ExportCompleted(result) => {
                self.export_status = Some(match self.errors.record(ErrorSource::Export, result) {
                    Some(path) => format!("Saved to {path}"),
                    None => "Export failed".to_string(),
                });
                Task::perform(
                    async { tokio::time::sleep(Duration::from_secs(5)).await },
                    |()| Message::ClearExportStatus,
                )
            }
            Message::ClearExportStatus => {
                self.export_status = None;
                Task::none()
            }
            Message::RepairStartDateChanged(d) => {
                self.repair.start_date = d;
                Task::none()
            }
            Message::RepairEndDateChanged(d) => {
                self.repair.end_date = d;
                Task::none()
            }
            Message::RepairPresetSelected(preset) => {
                self.select_repair_preset(preset);
                Task::none()
            }
            Message::StartRepairJob => self.start_repair(),
            Message::RepairCompleted(result) => {
                self.repair.is_running = false;
                self.repair.last_result = Some(result);
                Task::batch([self.load_chart_history(), self.load_analytics()])
            }
            _ => Task::none(),
        }
    }

    fn select_repair_preset(&mut self, preset: RepairPreset) {
        let today = self
            .clock
            .now_utc()
            .with_timezone(&self.schedule.timezone())
            .date_naive();
        let start = match preset {
            RepairPreset::Last7Days => today - chrono::TimeDelta::days(7),
            RepairPreset::Last30Days => today - chrono::TimeDelta::days(30),
            RepairPreset::AllData => chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap_or(today),
        };
        self.repair.start_date = start.format("%Y-%m-%d").to_string();
        self.repair.end_date = today.format("%Y-%m-%d").to_string();
    }

    fn start_repair(&mut self) -> Task<Message> {
        if self.repair.is_running {
            return Task::none();
        }
        let (Some(start), Some(end)) = (
            parse_date(&self.repair.start_date),
            parse_date(&self.repair.end_date),
        ) else {
            self.errors.raise(
                ErrorSource::RepairDatesInput,
                AppError::validation("Enter dates as YYYY-MM-DD"),
            );
            return Task::none();
        };
        if start > end {
            self.errors.raise(
                ErrorSource::RepairDatesInput,
                AppError::validation("Start date must be before end date"),
            );
            return Task::none();
        }

        self.repair.is_running = true;
        self.repair.last_result = None;
        self.errors.clear(ErrorSource::RepairDatesInput);

        let repairer = DataRepairer::new(self.db.clone(), self.schedule.clone());
        // Failures show in the repair card, not the header.
        self.load(
            async move { repairer.repair_date_range(start, end, None).await },
            |r| Message::RepairCompleted(r.map_err(|e| AppError::from_anyhow_db(e, "repair"))),
        )
    }
}
