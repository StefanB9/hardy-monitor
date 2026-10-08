//! The steps that repair one day, in the order `repair_day` runs them.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};

use super::DataRepairer;
use crate::db::{DataSource, OccupancyLog};

/// Gaps up to this long are filled by interpolation.
const MAX_GAP_MINUTES: i64 = 5;

impl DataRepairer {
    pub(super) async fn clean_outside_hours(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<(u32, u32)> {
        let local_tz = self.schedule.timezone();

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
            self.db
                .batch_update_percentage(&updates, DataSource::Boundary)
                .await?;
        }

        let deleted_count =
            u32::try_from(ids_to_delete.len()).context("too many records to delete")?;
        let zeroed_count = u32::try_from(ids_to_zero.len()).context("too many records to zero")?;

        Ok((deleted_count, zeroed_count))
    }

    pub(super) async fn smooth_and_filter(&self, records: &[OccupancyLog]) -> Result<u32> {
        if records.len() < 3 {
            return Ok(0);
        }

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

        let updates: Vec<(i64, f64)> = records
            .iter()
            .zip(&values)
            .zip(&changed)
            .filter(|(_, is_changed)| **is_changed)
            .map(|((record, &value), _)| (record.id, value))
            .collect();
        if !updates.is_empty() {
            self.db
                .batch_update_percentage(&updates, DataSource::Smoothed)
                .await?;
        }

        u32::try_from(updates.len()).context("too many records smoothed")
    }

    pub(super) async fn fill_gaps(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
        close_hour: u32,
    ) -> Result<u32> {
        let mut filled_count = 0;
        let local_tz = self.schedule.timezone();

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
            self.db
                .batch_insert(&inserts, DataSource::Interpolated)
                .await?;
        }

        Ok(filled_count)
    }

    pub(super) async fn ensure_start_of_day_entry(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        open_hour: u32,
    ) -> Result<bool> {
        let local_tz = self.schedule.timezone();

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
            self.db
                .insert_with_source(utc_dt, 0.0, DataSource::Boundary)
                .await?;
            Ok(true)
        }
    }

    pub(super) async fn ensure_end_of_day_entry(
        &self,
        records: &[OccupancyLog],
        date: NaiveDate,
        close_hour: u32,
    ) -> Result<bool> {
        let local_tz = self.schedule.timezone();

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
            self.db
                .insert_with_source(utc_dt, 0.0, DataSource::Boundary)
                .await?;
            Ok(true)
        }
    }
}
