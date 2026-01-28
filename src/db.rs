use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Serialize;
use sqlx::{FromRow, PgPool};

use crate::traits::Clock;

/// Represents a single occupancy log entry from the database.
#[derive(Debug, Clone, FromRow, Serialize)]
pub struct OccupancyLog {
    /// Populated by SQLx.
    #[allow(dead_code)]
    pub id: i64,
    pub timestamp: String,
    pub percentage: f64,
}

impl OccupancyLog {
    pub fn datetime(&self) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(&self.timestamp)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }
}

#[derive(Debug, Clone)]
pub struct HourlyAverage {
    pub weekday: i32, // 0=Monday, 6=Sunday
    pub hour: i32,    // 0-23
    pub avg_percentage: f64,
    #[allow(dead_code)]
    pub sample_count: i64,
}

#[derive(Clone, Debug)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL database")?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .context("Failed to run database migrations")?;

        Ok(Self { pool })
    }

    pub async fn insert_record(&self, timestamp: DateTime<Utc>, percentage: f64) -> Result<i64> {
        let timestamp_str = timestamp.to_rfc3339();

        // Use RETURNING to get the inserted ID (PostgreSQL)
        let result = sqlx::query_scalar!(
            "INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2) RETURNING id",
            timestamp_str,
            percentage
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert occupancy record")?;

        Ok(result)
    }

    pub async fn get_history(&self, days: i64) -> Result<Vec<OccupancyLog>> {
        let cutoff = Utc::now() - chrono::Duration::days(days);
        self.get_history_from(cutoff).await
    }

    /// Get the most recent occupancy record.
    pub async fn get_latest_record(&self) -> Result<Option<OccupancyLog>> {
        let log = sqlx::query_as!(
            OccupancyLog,
            r#"
            SELECT
                id as "id!",
                timestamp as "timestamp!",
                percentage as "percentage!"
            FROM occupancy_logs
            ORDER BY timestamp DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch latest occupancy record")?;

        Ok(log)
    }

    pub async fn get_history_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OccupancyLog>> {
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        let logs = sqlx::query_as!(
            OccupancyLog,
            r#"
            SELECT
                id as "id!",
                timestamp as "timestamp!",
                percentage as "percentage!"
            FROM occupancy_logs
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp ASC
            "#,
            start_str,
            end_str
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch occupancy history for date range")?;

        Ok(logs)
    }

    async fn get_history_from(&self, cutoff: DateTime<Utc>) -> Result<Vec<OccupancyLog>> {
        let cutoff_str = cutoff.to_rfc3339();

        let logs = sqlx::query_as!(
            OccupancyLog,
            r#"
            SELECT
                id as "id!",
                timestamp as "timestamp!",
                percentage as "percentage!"
            FROM occupancy_logs
            WHERE timestamp >= $1
            ORDER BY timestamp ASC
            "#,
            cutoff_str
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch occupancy history")?;

        Ok(logs)
    }

    pub async fn get_averages_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<HourlyAverage>> {
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        // PostgreSQL version:
        // - ISODOW returns 1=Monday through 7=Sunday, subtract 1 to get 0=Monday
        // - EXTRACT(HOUR ...) returns the hour (0-23)
        // - Cast timestamp TEXT to TIMESTAMPTZ for date functions
        let logs = sqlx::query_as!(
            HourlyAverage,
            r#"
            SELECT
                weekday as "weekday!: i32",
                hour as "hour!: i32",
                AVG(percentage) as "avg_percentage!: f64",
                COUNT(*) as "sample_count!: i64"
            FROM (
                SELECT
                    (EXTRACT(ISODOW FROM timestamp::timestamptz)::INTEGER - 1) as weekday,
                    EXTRACT(HOUR FROM timestamp::timestamptz)::INTEGER as hour,
                    percentage
                FROM occupancy_logs
                WHERE timestamp >= $1 AND timestamp < $2
            ) AS subquery
            GROUP BY weekday, hour
            ORDER BY weekday, hour
            "#,
            start_str,
            end_str
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch aggregated data")?;

        Ok(logs)
    }

    /// Export all occupancy logs to a CSV file.
    ///
    /// This function fetches all records from the database and writes them
    /// to a timestamped CSV file in the specified output directory.
    ///
    /// # Arguments
    /// * `output_dir` - Directory where the CSV file will be created
    /// * `clock` - Clock for generating the timestamp in the filename
    ///
    /// # Returns
    /// The path to the created CSV file on success.
    pub async fn export_to_csv<C: Clock>(&self, output_dir: &Path, clock: &C) -> Result<PathBuf> {
        let logs = self
            .get_history(365 * 10)
            .await
            .context("Failed to fetch history for export")?;

        let export_time = clock.now_utc();
        let filename = format!(
            "hardy_monitor_export_{}.csv",
            export_time.format("%Y%m%d_%H%M%S")
        );

        let output_path = output_dir.join(&filename);

        // Clone path and logs for the blocking task
        let path = output_path.clone();
        let logs_clone = logs;

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut wtr = csv::Writer::from_path(&path).context("Failed to create CSV writer")?;

            for log in logs_clone {
                wtr.serialize(log)
                    .context("Failed to serialize log entry")?;
            }

            wtr.flush().context("Failed to flush CSV writer")?;
            Ok(())
        })
        .await
        .context("CSV export task failed")??;

        Ok(output_path)
    }

    /// Get all records for a specific local date.
    ///
    /// This returns all occupancy logs where the timestamp falls within the
    /// given date when converted to local time.
    pub async fn get_records_for_date(&self, date: NaiveDate) -> Result<Vec<OccupancyLog>> {
        // Convert local date boundaries to UTC
        let local_tz = chrono::Local;
        let start_of_day = local_tz
            .from_local_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
            .single()
            .context("Invalid local datetime for start of day")?
            .with_timezone(&Utc);
        let end_of_day = local_tz
            .from_local_datetime(&date.and_hms_opt(23, 59, 59).unwrap())
            .single()
            .context("Invalid local datetime for end of day")?
            .with_timezone(&Utc);

        self.get_history_range(start_of_day, end_of_day).await
    }

    /// Update a record's percentage by ID.
    pub async fn update_percentage(&self, id: i64, percentage: f64) -> Result<()> {
        sqlx::query!(
            "UPDATE occupancy_logs SET percentage = $1 WHERE id = $2",
            percentage,
            id
        )
        .execute(&self.pool)
        .await
        .context("Failed to update percentage")?;
        Ok(())
    }

    /// Insert a record at a specific timestamp.
    pub async fn insert_at_timestamp(
        &self,
        timestamp: DateTime<Utc>,
        percentage: f64,
    ) -> Result<i64> {
        self.insert_record(timestamp, percentage).await
    }

    /// Batch insert multiple records.
    pub async fn batch_insert(&self, records: Vec<(DateTime<Utc>, f64)>) -> Result<()> {
        for (timestamp, percentage) in records {
            self.insert_record(timestamp, percentage).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Datelike, Timelike};

    use super::*;

    // ==================== OccupancyLog::datetime() Tests ====================

    fn make_log(timestamp: &str) -> OccupancyLog {
        OccupancyLog {
            id: 1,
            timestamp: timestamp.to_string(),
            percentage: 50.0,
        }
    }

    #[test]
    fn test_datetime_valid_rfc3339() {
        let log = make_log("2024-06-15T14:30:00+00:00");
        let result = log.datetime();
        assert!(result.is_some());
        let dt = result.unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 6);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 14);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn test_datetime_utc_timezone() {
        let log = make_log("2024-01-01T00:00:00Z");
        let result = log.datetime();
        assert!(result.is_some());
        let dt = result.unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 1);
    }

    #[test]
    fn test_datetime_with_offset() {
        let log = make_log("2024-06-15T16:30:00+02:00");
        let result = log.datetime();
        assert!(result.is_some());
        // Should be converted to UTC (14:30 UTC)
        let dt = result.unwrap();
        assert_eq!(dt.hour(), 14);
    }

    #[test]
    fn test_datetime_with_negative_offset() {
        let log = make_log("2024-06-15T10:30:00-04:00");
        let result = log.datetime();
        assert!(result.is_some());
        // Should be converted to UTC (14:30 UTC)
        let dt = result.unwrap();
        assert_eq!(dt.hour(), 14);
    }

    #[test]
    fn test_datetime_invalid_format() {
        let log = make_log("not-a-date");
        let result = log.datetime();
        assert!(result.is_none());
    }

    #[test]
    fn test_datetime_empty_string() {
        let log = make_log("");
        let result = log.datetime();
        assert!(result.is_none());
    }

    #[test]
    fn test_datetime_partial_date() {
        let log = make_log("2024-06-15");
        let result = log.datetime();
        assert!(result.is_none()); // RFC3339 requires time component
    }

    #[test]
    fn test_datetime_leap_second() {
        // Some systems handle leap seconds
        let log = make_log("2024-06-30T23:59:60Z");
        // This may or may not parse depending on chrono version
        let _result = log.datetime();
        // Just ensure it doesn't panic
    }

    #[test]
    fn test_datetime_milliseconds() {
        let log = make_log("2024-06-15T14:30:00.123Z");
        let result = log.datetime();
        assert!(result.is_some());
    }

    #[test]
    fn test_datetime_nanoseconds() {
        let log = make_log("2024-06-15T14:30:00.123456789Z");
        let result = log.datetime();
        assert!(result.is_some());
    }

    // ==================== HourlyAverage Struct Tests ====================

    #[test]
    fn test_hourly_average_fields() {
        let avg = HourlyAverage {
            weekday: 0, // Monday
            hour: 10,
            avg_percentage: 45.5,
            sample_count: 100,
        };
        assert_eq!(avg.weekday, 0);
        assert_eq!(avg.hour, 10);
        assert_eq!(avg.avg_percentage, 45.5);
        assert_eq!(avg.sample_count, 100);
    }

    #[test]
    fn test_hourly_average_boundary_values() {
        // Sunday at 23:00
        let avg = HourlyAverage {
            weekday: 6,
            hour: 23,
            avg_percentage: 0.0,
            sample_count: 1,
        };
        assert_eq!(avg.weekday, 6);
        assert_eq!(avg.hour, 23);
    }
}
