//! Data Repair Module
//!
//! This module provides functionality to repair gaps and outliers in occupancy
//! data:
//! - Deletes data outside working hours
//! - Sets opening (e.g. 06:00) and closing time (e.g. 23:00) to 0%
//! - Fills gaps up to 5 minutes with linear interpolation
//! - Replaces huge outliers and spikes
//! - Smooths data using a moving average

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
use futures::{StreamExt, stream};
use tokio::sync::mpsc;

use crate::{
    db::{Database, OccupancyLog},
    schedule::GymSchedule,
};

const MAX_GAP_MINUTES: i64 = 5;

const CONCURRENT_DAYS: usize = 4;

#[derive(Debug, Clone)]
pub struct RepairProgress {
    pub current_day: NaiveDate,
    pub total_days: u32,
    pub processed_days: u32,
}

#[derive(Debug, Clone)]
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

pub struct DataRepairer {
    db: Arc<Database>,
    schedule: GymSchedule,
}

impl DataRepairer {
    pub fn new(db: Arc<Database>, schedule: GymSchedule) -> Self {
        Self { db, schedule }
    }

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

        let records = self.db.get_records_for_date(date).await?;
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

        let records = self.db.get_records_for_date(date).await?;
        result.gaps_filled = self
            .fill_gaps(&records, date, open_hour, close_hour)
            .await?;

        let records = self.db.get_records_for_date(date).await?;
        result.records_smoothed += self.smooth_and_filter(&records).await?;

        Ok(result)
    }

    async fn clean_outside_hours(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<(u32, u32)> {
        let local_tz = Local;

        let open_time = NaiveTime::from_hms_opt(open_hour, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("invalid open hour: {open_hour}"))?;
        let close_time = NaiveTime::from_hms_opt(close_hour, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("invalid close hour: {close_hour}"))?;

        let mut ids_to_delete = Vec::new();
        let mut ids_to_zero = Vec::new();

        for record in records {
            let local_dt = record.timestamp.with_timezone(&local_tz);
            let local_date = local_dt.date_naive();
            let local_time = local_dt.time();

            if local_date != date {
                continue;
            }

            if local_time < open_time || local_time > close_time {
                ids_to_delete.push(record.id);
            } else if (local_time == open_time || local_time == close_time)
                && record.percentage != 0.0
            {
                ids_to_zero.push(record.id);
            }
        }

        if !ids_to_delete.is_empty() {
            self.db.batch_delete(&ids_to_delete).await?;
        }
        if !ids_to_zero.is_empty() {
            let updates: Vec<(i64, f64)> = ids_to_zero.iter().map(|&id| (id, 0.0)).collect();
            self.db.batch_update_percentage(&updates).await?;
        }

        let deleted_count =
            u32::try_from(ids_to_delete.len()).context("too many records to delete")?;
        let zeroed_count = u32::try_from(ids_to_zero.len()).context("too many records to zero")?;

        Ok((deleted_count, zeroed_count))
    }

    async fn smooth_and_filter(&self, records: &[OccupancyLog]) -> Result<u32> {
        if records.len() < 3 {
            return Ok(0);
        }

        let mut modified_count = 0;

        let mut values: Vec<f64> = records.iter().map(|r| r.percentage).collect();
        let mut changed = vec![false; values.len()];

        for i in 0..values.len() {
            if values[i] > 100.0 {
                values[i] = 100.0;
                changed[i] = true;
            }
        }

        for i in 1..values.len() - 1 {
            let prev = values[i - 1];
            let next = values[i + 1];
            let curr = values[i];

            let avg_neighbors = f64::midpoint(prev, next);
            if (curr - avg_neighbors).abs() > 30.0 && (prev - next).abs() < 20.0 {
                values[i] = avg_neighbors;
                changed[i] = true;
            }
        }

        let mut prev_original = values[0];
        for i in 1..values.len() - 1 {
            let curr_original = values[i];
            let smoothed = (prev_original + curr_original + values[i + 1]) / 3.0;

            if (values[i] - smoothed).abs() > 1.0 {
                values[i] = smoothed;
                changed[i] = true;
            }
            prev_original = curr_original;
        }

        for i in 0..records.len() {
            if changed[i] {
                self.db.update_percentage(records[i].id, values[i]).await?;
                modified_count += 1;
            }
        }

        Ok(modified_count)
    }

    async fn fill_gaps(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<u32> {
        let mut filled_count = 0;
        let local_tz = Local;

        let mut data_points: Vec<(i64, f64)> = Vec::new();

        for record in records {
            let local_dt = record.timestamp.with_timezone(&local_tz);
            let local_date = local_dt.date_naive();

            if local_date == date {
                let minute_of_day = i64::from(local_dt.hour()) * 60 + i64::from(local_dt.minute());
                data_points.push((minute_of_day, record.percentage));
            }
        }

        data_points.sort_by_key(|(m, _)| *m);

        if data_points.len() < 2 {
            return Ok(0);
        }

        let open_minute = i64::from(open_hour) * 60;
        let close_minute = i64::from(close_hour) * 60;

        let mut inserts: Vec<(DateTime<Utc>, f64)> = Vec::new();

        for i in 0..data_points.len() - 1 {
            let (m1, v1) = data_points[i];
            let (m2, v2) = data_points[i + 1];

            let gap_minutes = m2 - m1;

            if gap_minutes > 1
                && gap_minutes <= MAX_GAP_MINUTES
                && m1 >= open_minute
                && m2 <= close_minute
            {
                for m in (m1 + 1)..m2 {
                    #[allow(clippy::cast_precision_loss)]
                    let t = (m - m1) as f64 / gap_minutes as f64;
                    let interpolated = v1 + t * (v2 - v1);

                    let hour = u32::try_from(m / 60).unwrap_or(0);
                    let minute = u32::try_from(m % 60).unwrap_or(0);
                    let local_time = NaiveTime::from_hms_opt(hour, minute, 0)
                        .ok_or_else(|| anyhow::anyhow!("invalid time {hour}:{minute}"))?;
                    let local_dt = local_tz
                        .from_local_datetime(&date.and_time(local_time))
                        .single()
                        .context("Invalid local datetime for interpolation")?;
                    let utc_dt = local_dt.with_timezone(&Utc);

                    inserts.push((utc_dt, interpolated));
                    filled_count += 1;
                }
            }
        }

        if !inserts.is_empty() {
            self.db.batch_insert(inserts).await?;
        }

        Ok(filled_count)
    }

    async fn ensure_start_of_day_entry(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
    ) -> Result<bool> {
        let local_tz = Local;

        let start_time = NaiveTime::from_hms_opt(open_hour, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("invalid open hour: {open_hour}"))?;
        let local_dt = local_tz
            .from_local_datetime(&date.and_time(start_time))
            .single()
            .context("Invalid local datetime for start of day entry")?;
        let utc_dt = local_dt.with_timezone(&Utc);

        let exists = records.iter().any(|r| {
            let local = r.timestamp.with_timezone(&local_tz);
            local.date_naive() == date && local.hour() == open_hour && local.minute() == 0
        });

        if exists {
            Ok(false)
        } else {
            self.db.insert_at_timestamp(utc_dt, 0.0).await?;
            Ok(true)
        }
    }

    async fn ensure_end_of_day_entry(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        close_hour: u32,
    ) -> Result<bool> {
        let local_tz = Local;

        let end_time = NaiveTime::from_hms_opt(close_hour, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("invalid close hour: {close_hour}"))?;
        let local_dt = local_tz
            .from_local_datetime(&date.and_time(end_time))
            .single()
            .context("Invalid local datetime for end of day entry")?;
        let utc_dt = local_dt.with_timezone(&Utc);

        let exists = records.iter().any(|r| {
            let local = r.timestamp.with_timezone(&local_tz);
            local.date_naive() == date && local.hour() == close_hour && local.minute() == 0
        });

        if exists {
            Ok(false)
        } else {
            self.db.insert_at_timestamp(utc_dt, 0.0).await?;
            Ok(true)
        }
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
