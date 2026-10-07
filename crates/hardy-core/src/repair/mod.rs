//! Data Repair Module
//!
//! This module provides functionality to repair gaps and outliers in occupancy
//! data:
//! - Deletes data outside working hours
//! - Sets opening (e.g. 06:00) and closing time (e.g. 23:00) to 0%
//! - Fills gaps up to 5 minutes with linear interpolation
//! - Replaces huge outliers and spikes
//! - Smooths data using a moving average

mod nightly;
mod steps;

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate};
use futures::{StreamExt, stream};
pub use nightly::{AFTER_CLOSE, MAX_CATCH_UP_DAYS, RETRY_AFTER, repair_due, run_nightly_repair};
use tokio::sync::mpsc;

use crate::{db::Database, schedule::GymSchedule};

const CONCURRENT_DAYS: usize = 4;

/// Progress report sent while a range is repaired.
#[derive(Debug, Clone)]
pub struct RepairProgress {
    pub current_day: NaiveDate,
    pub total_days: u32,
    pub processed_days: u32,
}

/// What a repair changed.
#[derive(Debug, Clone, Default)]
pub struct RepairSummary {
    pub days_processed: u32,
    pub gaps_filled: u32,
    pub records_deleted: u32,
    pub records_smoothed: u32,
    pub boundary_entries_added: u32,
}

#[derive(Debug, Default)]
struct DayRepairResult {
    gaps_filled: u32,
    records_deleted: u32,
    records_smoothed: u32,
    boundary_entries_added: u32,
}

/// Repairs stored readings day by day (see the module docs for the steps).
pub struct DataRepairer {
    db: Arc<Database>,
    schedule: GymSchedule,
}

impl DataRepairer {
    /// Repairer writing through `db`, using `schedule` for opening hours.
    pub fn new(db: Arc<Database>, schedule: GymSchedule) -> Self {
        Self { db, schedule }
    }

    /// Repairs every gym-local day from `start` to `end` (inclusive), a
    /// few days concurrently; progress goes to `progress_tx` when given.
    pub async fn repair_date_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        progress_tx: Option<mpsc::UnboundedSender<RepairProgress>>,
    ) -> Result<RepairSummary> {
        let dates: Vec<NaiveDate> = {
            let mut dates = Vec::new();
            let mut current = start;
            while current <= end {
                dates.push(current);
                current += Duration::days(1);
            }
            dates
        };

        let total_days = u32::try_from(dates.len()).context("date range too large for repair")?;

        let results: Vec<Result<(NaiveDate, DayRepairResult)>> = stream::iter(dates)
            .map(|date| async move {
                let result = self.repair_day(date).await?;
                Ok((date, result))
            })
            .buffer_unordered(CONCURRENT_DAYS)
            .collect()
            .await;

        let mut summary = RepairSummary {
            days_processed: 0,
            gaps_filled: 0,
            records_deleted: 0,
            records_smoothed: 0,
            boundary_entries_added: 0,
        };

        for result in results {
            let (date, day_result) = result?;

            summary.days_processed += 1;
            summary.gaps_filled += day_result.gaps_filled;
            summary.records_deleted += day_result.records_deleted;
            summary.records_smoothed += day_result.records_smoothed;
            summary.boundary_entries_added += day_result.boundary_entries_added;

            if let Some(ref tx) = progress_tx {
                let _ = tx.send(RepairProgress {
                    current_day: date,
                    total_days,
                    processed_days: summary.days_processed,
                });
            }
        }

        Ok(summary)
    }

    async fn repair_day(&self, date: NaiveDate) -> Result<DayRepairResult> {
        let mut result = DayRepairResult::default();

        let open_hour = self.schedule.get_open_hour(date);
        let close_hour = self.schedule.get_close_hour(date);

        let records = self
            .db
            .get_records_for_date(date, self.schedule.timezone())
            .await?;
        let (deleted, zeroed) = self
            .clean_outside_hours(&records, date, open_hour, close_hour)
            .await?;
        result.records_deleted = deleted;
        if zeroed > 0 {
            result.records_smoothed += zeroed;
        }

        if self
            .ensure_start_of_day_entry(&records, date, open_hour)
            .await?
        {
            result.boundary_entries_added += 1;
        }
        if self
            .ensure_end_of_day_entry(&records, date, close_hour)
            .await?
        {
            result.boundary_entries_added += 1;
        }

        let records = self
            .db
            .get_records_for_date(date, self.schedule.timezone())
            .await?;
        result.gaps_filled = self
            .fill_gaps(&records, date, open_hour, close_hour)
            .await?;

        let records = self
            .db
            .get_records_for_date(date, self.schedule.timezone())
            .await?;
        result.records_smoothed += self.smooth_and_filter(&records).await?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_repair_summary_default() {
        let summary = RepairSummary {
            days_processed: 0,
            gaps_filled: 0,
            records_deleted: 0,
            records_smoothed: 0,
            boundary_entries_added: 0,
        };
        assert_eq!(summary.days_processed, 0);
    }

    #[test]
    fn test_repair_progress_creation() -> Result<()> {
        let progress = RepairProgress {
            current_day: NaiveDate::from_ymd_opt(2024, 1, 15)
                .ok_or_else(|| anyhow::anyhow!("Invalid date"))?,
            total_days: 30,
            processed_days: 5,
        };
        assert_eq!(progress.total_days, 30);
        assert_eq!(progress.processed_days, 5);

        Ok(())
    }
}
