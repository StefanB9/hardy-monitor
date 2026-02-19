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

/// Maximum gap in minutes that will be filled with interpolation.
const MAX_GAP_MINUTES: i64 = 5;

/// Number of days to process concurrently during repair.
const CONCURRENT_DAYS: usize = 4;

/// Progress update for a repair job.
#[derive(Debug, Clone)]
pub struct RepairProgress {
    pub current_day: NaiveDate,
    pub total_days: u32,
    pub processed_days: u32,
}

/// Summary of a completed repair job.
#[derive(Debug, Clone)]
pub struct RepairSummary {
    pub days_processed: u32,
    pub gaps_filled: u32,
    pub records_deleted: u32,
    pub records_smoothed: u32,
    pub boundary_entries_added: u32, // Renamed from end_entries_added
}

/// Result of repairing a single day.
#[derive(Debug, Default)]
struct DayRepairResult {
    gaps_filled: u32,
    records_deleted: u32,
    records_smoothed: u32,
    boundary_entries_added: u32,
}

/// Data repairer that handles filling gaps and normalizing occupancy data.
pub struct DataRepairer {
    db: Arc<Database>,
    schedule: GymSchedule,
}

impl DataRepairer {
    /// Create a new DataRepairer.
    pub fn new(db: Arc<Database>, schedule: GymSchedule) -> Self {
        Self { db, schedule }
    }

    /// Repair data for a date range.
    pub async fn repair_date_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        progress_tx: Option<mpsc::UnboundedSender<RepairProgress>>,
    ) -> Result<RepairSummary> {
        // Collect all dates to process
        let dates: Vec<NaiveDate> = {
            let mut dates = Vec::new();
            let mut current = start;
            while current <= end {
                dates.push(current);
                current += Duration::days(1);
            }
            dates
        };

        let total_days = dates.len() as u32;

        // Process days concurrently using buffer_unordered
        let results: Vec<Result<(NaiveDate, DayRepairResult)>> = stream::iter(dates)
            .map(|date| async move {
                let result = self.repair_day(date).await?;
                Ok((date, result))
            })
            .buffer_unordered(CONCURRENT_DAYS)
            .collect()
            .await;

        // Aggregate results
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

            // Send progress update after each completed day
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

    /// Repair data for a single day.
    async fn repair_day(&self, date: NaiveDate) -> Result<DayRepairResult> {
        let mut result = DayRepairResult::default();

        // Get opening hours for this day
        let open_hour = self.schedule.get_open_hour(date);
        let close_hour = self.schedule.get_close_hour(date);

        // Step 1: Clean outside hours
        // - Delete records < open_hour
        // - Delete records > close_hour
        // - Set records == open_hour OR close_hour to 0
        let records = self.db.get_records_for_date(date).await?;
        let (deleted, zeroed) = self
            .clean_outside_hours(&records, date, open_hour, close_hour)
            .await?;
        result.records_deleted = deleted;
        if zeroed > 0 {
            result.records_smoothed += zeroed;
        }

        // Step 2: Ensure boundary entries exist (Start and End of day)
        if self.ensure_start_of_day_entry(date, open_hour).await? {
            result.boundary_entries_added += 1;
        }
        if self.ensure_end_of_day_entry(date, close_hour).await? {
            result.boundary_entries_added += 1;
        }

        // Step 3: Fill gaps with interpolation
        // Reload records after cleanup
        let records = self.db.get_records_for_date(date).await?;
        result.gaps_filled = self
            .fill_gaps(&records, date, open_hour, close_hour)
            .await?;

        // Step 4: Process outliers and smooth data
        // Reload records again to include interpolated values
        let records = self.db.get_records_for_date(date).await?;
        result.records_smoothed += self.smooth_and_filter(&records).await?;

        Ok(result)
    }

    /// Clean records outside valid operating hours.
    /// Returns (count_deleted, count_zeroed).
    async fn clean_outside_hours(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<(u32, u32)> {
        let mut deleted_count = 0;
        let mut zeroed_count = 0;
        let local_tz = Local;

        // Opening time is open_hour:00, closing time is close_hour:00
        let open_time = NaiveTime::from_hms_opt(open_hour, 0, 0).unwrap();
        let close_time = NaiveTime::from_hms_opt(close_hour, 0, 0).unwrap();

        for record in records {
            let local_dt = record.timestamp.with_timezone(&local_tz);
            let local_date = local_dt.date_naive();
            let local_time = local_dt.time();

            if local_date != date {
                continue;
            }

            if local_time < open_time || local_time > close_time {
                // Delete strictly outside
                self.db.delete_record(record.id).await?;
                deleted_count += 1;
            } else if local_time == open_time || local_time == close_time {
                // Exact opening or closing time: ensure 0
                if record.percentage != 0.0 {
                    self.db.update_percentage(record.id, 0.0).await?;
                    zeroed_count += 1;
                }
            }
        }

        Ok((deleted_count, zeroed_count))
    }

    /// Remove outliers and smooth data using a moving average.
    async fn smooth_and_filter(&self, records: &[OccupancyLog]) -> Result<u32> {
        if records.len() < 3 {
            return Ok(0);
        }

        let mut modified_count = 0;

        // 1. Convert to simple vector for processing
        let mut values: Vec<f64> = records.iter().map(|r| r.percentage).collect();
        let mut changed = vec![false; values.len()];

        // 2. Clamp huge outliers (> 100%)
        for i in 0..values.len() {
            if values[i] > 100.0 {
                values[i] = 100.0;
                changed[i] = true;
            }
        }

        // 3. Despike
        for i in 1..values.len() - 1 {
            let prev = values[i - 1];
            let next = values[i + 1];
            let curr = values[i];

            let avg_neighbors = (prev + next) / 2.0;
            // If current deviates from neighbors average by > 30% and neighbors are
            // relatively close
            if (curr - avg_neighbors).abs() > 30.0 && (prev - next).abs() < 20.0 {
                values[i] = avg_neighbors;
                changed[i] = true;
            }
        }

        // 4. Smooth (3-point moving average)
        let source_values = values.clone();
        for i in 1..values.len() - 1 {
            let smoothed = (source_values[i - 1] + source_values[i] + source_values[i + 1]) / 3.0;

            if (values[i] - smoothed).abs() > 1.0 {
                values[i] = smoothed;
                changed[i] = true;
            }
        }

        // 5. Write back changes
        for i in 0..records.len() {
            if changed[i] {
                self.db.update_percentage(records[i].id, values[i]).await?;
                modified_count += 1;
            }
        }

        Ok(modified_count)
    }

    /// Fill gaps in the data with linear interpolation.
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
                let minute_of_day = local_dt.hour() as i64 * 60 + local_dt.minute() as i64;
                data_points.push((minute_of_day, record.percentage));
            }
        }

        data_points.sort_by_key(|(m, _)| *m);

        if data_points.len() < 2 {
            return Ok(0);
        }

        let open_minute = open_hour as i64 * 60;
        let close_minute = close_hour as i64 * 60;

        let mut inserts: Vec<(DateTime<Utc>, f64)> = Vec::new();

        for i in 0..data_points.len() - 1 {
            let (m1, v1) = data_points[i];
            let (m2, v2) = data_points[i + 1];

            let gap_minutes = m2 - m1;

            if gap_minutes > 1 && gap_minutes <= MAX_GAP_MINUTES {
                if m1 >= open_minute && m2 <= close_minute {
                    for m in (m1 + 1)..m2 {
                        let t = (m - m1) as f64 / gap_minutes as f64;
                        let interpolated = v1 + t * (v2 - v1);

                        let hour = (m / 60) as u32;
                        let minute = (m % 60) as u32;
                        let local_time = NaiveTime::from_hms_opt(hour, minute, 0).unwrap();
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
        }

        if !inserts.is_empty() {
            self.db.batch_insert(inserts).await?;
        }

        Ok(filled_count)
    }

    /// Ensure a start-of-day entry exists at open_hour:00.
    async fn ensure_start_of_day_entry(&self, date: NaiveDate, open_hour: u32) -> Result<bool> {
        let local_tz = Local;

        // Start of day time is open_hour:00
        let start_time = NaiveTime::from_hms_opt(open_hour, 0, 0).unwrap();
        let local_dt = local_tz
            .from_local_datetime(&date.and_time(start_time))
            .single()
            .context("Invalid local datetime for start of day entry")?;
        let utc_dt = local_dt.with_timezone(&Utc);

        let records = self.db.get_records_for_date(date).await?;

        let exists = records.iter().any(|r| {
            let local = r.timestamp.with_timezone(&local_tz);
            local.date_naive() == date && local.hour() == open_hour && local.minute() == 0
        });

        if !exists {
            self.db.insert_at_timestamp(utc_dt, 0.0).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Ensure an end-of-day entry exists at close_hour:00.
    async fn ensure_end_of_day_entry(&self, date: NaiveDate, close_hour: u32) -> Result<bool> {
        let local_tz = Local;

        // End of day time is close_hour:00
        let end_time = NaiveTime::from_hms_opt(close_hour, 0, 0).unwrap();
        let local_dt = local_tz
            .from_local_datetime(&date.and_time(end_time))
            .single()
            .context("Invalid local datetime for end of day entry")?;
        let utc_dt = local_dt.with_timezone(&Utc);

        // Check if an entry already exists at this time
        let records = self.db.get_records_for_date(date).await?;

        let exists = records.iter().any(|r| {
            let local = r.timestamp.with_timezone(&local_tz);
            local.date_naive() == date && local.hour() == close_hour && local.minute() == 0
        });

        if !exists {
            self.db.insert_at_timestamp(utc_dt, 0.0).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn test_repair_progress_creation() {
        let progress = RepairProgress {
            current_day: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            total_days: 30,
            processed_days: 5,
        };
        assert_eq!(progress.total_days, 30);
        assert_eq!(progress.processed_days, 5);
    }
}
