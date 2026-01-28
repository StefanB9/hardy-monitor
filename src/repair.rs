//! Data Repair Module
//!
//! This module provides functionality to repair gaps in occupancy data:
//! - Fill missing minute-by-minute data with linear interpolation (gaps up to 5
//!   minutes)
//! - Normalize values outside opening hours to 0
//! - Ensure end-of-day closure entries exist at close_hour:01

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
use tokio::sync::mpsc;

use crate::{
    db::{Database, OccupancyLog},
    schedule::GymSchedule,
};

/// Maximum gap in minutes that will be filled with interpolation.
const MAX_GAP_MINUTES: i64 = 5;

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
    pub records_zeroed: u32,
    pub end_entries_added: u32,
}

/// Result of repairing a single day.
#[derive(Debug, Default)]
struct DayRepairResult {
    gaps_filled: u32,
    records_zeroed: u32,
    end_entry_added: bool,
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
    ///
    /// This will:
    /// 1. Zero out records outside opening hours
    /// 2. Fill gaps up to 5 minutes with linear interpolation
    /// 3. Add end-of-day entries at close_hour:01 if missing
    pub async fn repair_date_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        progress_tx: Option<mpsc::UnboundedSender<RepairProgress>>,
    ) -> Result<RepairSummary> {
        let mut summary = RepairSummary {
            days_processed: 0,
            gaps_filled: 0,
            records_zeroed: 0,
            end_entries_added: 0,
        };

        let total_days = (end - start).num_days() as u32 + 1;
        let mut current = start;

        while current <= end {
            // Send progress update
            if let Some(ref tx) = progress_tx {
                let _ = tx.send(RepairProgress {
                    current_day: current,
                    total_days,
                    processed_days: summary.days_processed,
                });
            }

            let result = self.repair_day(current).await?;

            summary.days_processed += 1;
            summary.gaps_filled += result.gaps_filled;
            summary.records_zeroed += result.records_zeroed;
            if result.end_entry_added {
                summary.end_entries_added += 1;
            }

            current += Duration::days(1);
        }

        Ok(summary)
    }

    /// Repair data for a single day.
    async fn repair_day(&self, date: NaiveDate) -> Result<DayRepairResult> {
        let mut result = DayRepairResult::default();

        // Get opening hours for this day
        let open_hour = self.schedule.get_open_hour(date);
        let close_hour = self.schedule.get_close_hour(date);

        // Load all records for the day
        let records = self.db.get_records_for_date(date).await?;

        // Step A: Zero records outside opening hours
        result.records_zeroed = self
            .zero_outside_hours(&records, date, open_hour, close_hour)
            .await?;

        // Step B: Fill gaps with interpolation
        // Reload records after zeroing (to get updated values)
        let records = self.db.get_records_for_date(date).await?;
        result.gaps_filled = self
            .fill_gaps(&records, date, open_hour, close_hour)
            .await?;

        // Step C: Ensure end-of-day entry exists
        result.end_entry_added = self.ensure_end_of_day_entry(date, close_hour).await?;

        Ok(result)
    }

    /// Zero out records that fall outside the opening hours.
    async fn zero_outside_hours(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<u32> {
        let mut zeroed_count = 0;
        let local_tz = Local;

        // Opening time is open_hour:00, closing time is close_hour:00
        let open_time = NaiveTime::from_hms_opt(open_hour, 0, 0).unwrap();
        let close_time = NaiveTime::from_hms_opt(close_hour, 0, 0).unwrap();

        for record in records {
            if let Some(utc_dt) = record.datetime() {
                let local_dt = utc_dt.with_timezone(&local_tz);
                let local_date = local_dt.date_naive();
                let local_time = local_dt.time();

                // Only process records from the target date
                if local_date != date {
                    continue;
                }

                // Check if outside opening hours and not already zero
                let is_outside = local_time < open_time || local_time > close_time;
                if is_outside && record.percentage != 0.0 {
                    self.db.update_percentage(record.id, 0.0).await?;
                    zeroed_count += 1;
                }
            }
        }

        Ok(zeroed_count)
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

        // Build a list of (minute_of_day, percentage) for records on this date
        let mut data_points: Vec<(i64, f64)> = Vec::new();

        for record in records {
            if let Some(utc_dt) = record.datetime() {
                let local_dt = utc_dt.with_timezone(&local_tz);
                let local_date = local_dt.date_naive();

                if local_date == date {
                    let minute_of_day = local_dt.hour() as i64 * 60 + local_dt.minute() as i64;
                    data_points.push((minute_of_day, record.percentage));
                }
            }
        }

        // Sort by minute of day
        data_points.sort_by_key(|(m, _)| *m);

        if data_points.len() < 2 {
            return Ok(0);
        }

        // Opening and closing in minutes of day
        let open_minute = open_hour as i64 * 60;
        let close_minute = close_hour as i64 * 60;

        // Find gaps and interpolate
        let mut inserts: Vec<(DateTime<Utc>, f64)> = Vec::new();

        for i in 0..data_points.len() - 1 {
            let (m1, v1) = data_points[i];
            let (m2, v2) = data_points[i + 1];

            let gap_minutes = m2 - m1;

            // Only fill gaps that are:
            // 1. Within opening hours
            // 2. Greater than 1 minute (missing data)
            // 3. Less than or equal to MAX_GAP_MINUTES
            if gap_minutes > 1 && gap_minutes <= MAX_GAP_MINUTES {
                // Check if the gap is within opening hours
                if m1 >= open_minute && m2 <= close_minute {
                    // Linear interpolation for each missing minute
                    for m in (m1 + 1)..m2 {
                        let t = (m - m1) as f64 / gap_minutes as f64;
                        let interpolated = v1 + t * (v2 - v1);

                        // Convert minute of day back to timestamp
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

        // Batch insert the interpolated values
        if !inserts.is_empty() {
            self.db.batch_insert(inserts).await?;
        }

        Ok(filled_count)
    }

    /// Ensure an end-of-day entry exists at close_hour:01.
    async fn ensure_end_of_day_entry(&self, date: NaiveDate, close_hour: u32) -> Result<bool> {
        let local_tz = Local;

        // End of day time is close_hour:01
        let end_time = NaiveTime::from_hms_opt(close_hour, 1, 0).unwrap();
        let local_dt = local_tz
            .from_local_datetime(&date.and_time(end_time))
            .single()
            .context("Invalid local datetime for end of day entry")?;
        let utc_dt = local_dt.with_timezone(&Utc);

        // Check if an entry already exists at this time
        let records = self.db.get_records_for_date(date).await?;

        let exists = records.iter().any(|r| {
            if let Some(dt) = r.datetime() {
                let local = dt.with_timezone(&local_tz);
                local.date_naive() == date && local.hour() == close_hour && local.minute() == 1
            } else {
                false
            }
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
            records_zeroed: 0,
            end_entries_added: 0,
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
