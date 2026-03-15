use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use futures::TryStreamExt;
use serde::Serialize;
use sqlx::{FromRow, PgPool};

use crate::traits::Clock;

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct OccupancyLog {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub percentage: f64,
}

const _: () = assert!(
    std::mem::size_of::<OccupancyLog>() <= 32,
    "OccupancyLog size regression — check for unintended field additions or alignment padding"
);

#[derive(Debug, Clone)]
pub struct HourlyAverage {
    pub weekday: i32,
    pub hour: i32,
    pub avg_percentage: f64,
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

        sqlx::migrate!("../../migrations")
            .run(&pool)
            .await
            .context("Failed to run database migrations")?;

        Ok(Self { pool })
    }

    #[tracing::instrument(skip_all, fields(db.operation = "insert", %timestamp))]
    pub async fn insert_record(&self, timestamp: DateTime<Utc>, percentage: f64) -> Result<i64> {
        let result = sqlx::query_scalar!(
            "INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2) RETURNING id",
            timestamp,
            percentage
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert occupancy record")?;

        Ok(result)
    }

    #[tracing::instrument(skip_all, fields(db.operation = "get_history", days))]
    pub async fn get_history(&self, days: i64) -> Result<Vec<OccupancyLog>> {
        let cutoff = Utc::now() - chrono::Duration::days(days);
        self.get_history_from(cutoff).await
    }

    #[tracing::instrument(skip_all, fields(db.operation = "get_latest"))]
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

    #[tracing::instrument(skip_all, fields(db.operation = "get_history_range", %start, %end))]
    pub async fn get_history_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OccupancyLog>> {
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
            start,
            end
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch occupancy history for date range")?;

        Ok(logs)
    }

    async fn get_history_from(&self, cutoff: DateTime<Utc>) -> Result<Vec<OccupancyLog>> {
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
            cutoff
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch occupancy history")?;

        Ok(logs)
    }

    #[tracing::instrument(skip_all, fields(db.operation = "get_averages_range", %start, %end))]
    pub async fn get_averages_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<HourlyAverage>> {
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
                    (EXTRACT(ISODOW FROM timestamp)::INTEGER - 1) as weekday,
                    EXTRACT(HOUR FROM timestamp)::INTEGER as hour,
                    percentage
                FROM occupancy_logs
                WHERE timestamp >= $1 AND timestamp < $2
            ) AS subquery
            GROUP BY weekday, hour
            ORDER BY weekday, hour
            "#,
            start,
            end
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch aggregated data")?;

        Ok(logs)
    }

    #[tracing::instrument(skip_all, fields(db.operation = "export_csv", output_dir = %output_dir.display()))]
    pub async fn export_to_csv(&self, output_dir: &Path, clock: &dyn Clock) -> Result<PathBuf> {
        let export_time = clock.now_utc();
        let filename = format!(
            "hardy_monitor_export_{}.csv",
            export_time.format("%Y%m%d_%H%M%S")
        );
        let output_path = output_dir.join(&filename);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<OccupancyLog>(256);

        let path = output_path.clone();
        let writer_task = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut wtr = csv::Writer::from_path(&path).context("Failed to create CSV writer")?;
            while let Some(log) = rx.blocking_recv() {
                wtr.serialize(log)
                    .context("Failed to serialize log entry")?;
            }
            wtr.flush().context("Failed to flush CSV writer")
        });

        let mut stream = sqlx::query_as!(
            OccupancyLog,
            r#"
            SELECT
                id as "id!",
                timestamp as "timestamp!",
                percentage as "percentage!"
            FROM occupancy_logs
            ORDER BY timestamp ASC
            "#
        )
        .fetch(&self.pool);

        while let Some(log) = stream
            .try_next()
            .await
            .context("Failed to stream record during export")?
        {
            if tx.send(log).await.is_err() {
                break;
            }
        }

        drop(tx);

        writer_task
            .await
            .context("CSV export writer task panicked")??;

        Ok(output_path)
    }

    #[tracing::instrument(skip_all, fields(db.operation = "get_records_for_date", %date))]
    pub async fn get_records_for_date(&self, date: NaiveDate) -> Result<Vec<OccupancyLog>> {
        let local_tz = chrono::Local;
        let start_of_day = local_tz
            .from_local_datetime(
                &date
                    .and_hms_opt(0, 0, 0)
                    .context("failed to construct start-of-day time (possible DST gap)")?,
            )
            .single()
            .context("Invalid local datetime for start of day")?
            .with_timezone(&Utc);
        let end_of_day = local_tz
            .from_local_datetime(
                &date
                    .and_hms_opt(23, 59, 59)
                    .context("failed to construct end-of-day time (possible DST gap)")?,
            )
            .single()
            .context("Invalid local datetime for end of day")?
            .with_timezone(&Utc);

        self.get_history_range(start_of_day, end_of_day).await
    }

    #[tracing::instrument(skip_all, fields(db.operation = "update_percentage", id))]
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

    #[tracing::instrument(skip_all, fields(db.operation = "insert_at_timestamp", %timestamp))]
    pub async fn insert_at_timestamp(
        &self,
        timestamp: DateTime<Utc>,
        percentage: f64,
    ) -> Result<i64> {
        self.insert_record(timestamp, percentage).await
    }

    #[tracing::instrument(skip_all, fields(db.operation = "batch_insert", count = records.len()))]
    pub async fn batch_insert(&self, records: Vec<(DateTime<Utc>, f64)>) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin transaction")?;

        for (timestamp, percentage) in records {
            sqlx::query!(
                "INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2)",
                timestamp,
                percentage
            )
            .execute(&mut *tx)
            .await
            .context("failed to insert record in batch")?;
        }

        tx.commit().await.context("failed to commit batch insert")
    }

    #[tracing::instrument(skip_all, fields(db.operation = "delete_record", id))]
    pub async fn delete_record(&self, id: i64) -> Result<()> {
        sqlx::query!("DELETE FROM occupancy_logs WHERE id = $1", id)
            .execute(&self.pool)
            .await
            .context("Failed to delete record")?;
        Ok(())
    }

    /// Delete multiple records by ID in a single query.
    #[tracing::instrument(skip_all, fields(db.operation = "batch_delete", count = ids.len()))]
    pub async fn batch_delete(&self, ids: &[i64]) -> Result<u64> {
        let result = sqlx::query!("DELETE FROM occupancy_logs WHERE id = ANY($1)", ids)
            .execute(&self.pool)
            .await
            .context("Failed to batch delete records")?;
        Ok(result.rows_affected())
    }

    /// Update percentage for multiple records in a single transaction.
    #[tracing::instrument(skip_all, fields(db.operation = "batch_update_percentage", count = updates.len()))]
    pub async fn batch_update_percentage(&self, updates: &[(i64, f64)]) -> Result<u64> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin batch update transaction")?;

        let mut total = 0u64;
        for &(id, percentage) in updates {
            let result = sqlx::query!(
                "UPDATE occupancy_logs SET percentage = $1 WHERE id = $2",
                percentage,
                id
            )
            .execute(&mut *tx)
            .await
            .context("Failed to update percentage in batch")?;
            total += result.rows_affected();
        }

        tx.commit()
            .await
            .context("failed to commit batch update")?;
        Ok(total)
    }

    pub async fn close(self) {
        self.pool.close().await;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::{Datelike, TimeZone, Timelike};

    use super::*;

    fn make_log(timestamp: DateTime<Utc>) -> OccupancyLog {
        OccupancyLog {
            id: 1,
            timestamp,
            percentage: 50.0,
        }
    }

    #[test]
    fn test_timestamp_utc_fields() -> Result<()> {
        let ts = Utc
            .with_ymd_and_hms(2024, 6, 15, 14, 30, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
        let log = make_log(ts);
        assert_eq!(log.timestamp.year(), 2024);
        assert_eq!(log.timestamp.month(), 6);
        assert_eq!(log.timestamp.day(), 15);
        assert_eq!(log.timestamp.hour(), 14);
        assert_eq!(log.timestamp.minute(), 30);
        Ok(())
    }

    #[test]
    fn test_timestamp_year_boundary() -> Result<()> {
        let ts = Utc
            .with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
        let log = make_log(ts);
        assert_eq!(log.timestamp.year(), 2024);
        assert_eq!(log.timestamp.month(), 1);
        assert_eq!(log.timestamp.day(), 1);
        Ok(())
    }

    #[test]
    fn test_timestamp_roundtrips_via_rfc3339() -> Result<()> {
        let ts = Utc
            .with_ymd_and_hms(2024, 6, 15, 14, 30, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
        let log = make_log(ts);
        let reparsed =
            DateTime::parse_from_rfc3339(&log.timestamp.to_rfc3339())?.with_timezone(&Utc);
        assert_eq!(log.timestamp, reparsed);
        Ok(())
    }

    #[test]
    fn test_timestamp_subsecond_precision() -> Result<()> {
        use chrono::NaiveDateTime;
        let ndt =
            NaiveDateTime::parse_from_str("2024-06-15T14:30:00.123456789", "%Y-%m-%dT%H:%M:%S%.f")?;
        let ts = Utc.from_utc_datetime(&ndt);
        let log = make_log(ts);
        assert_eq!(log.timestamp.nanosecond(), 123_456_789);
        Ok(())
    }

    #[test]
    fn test_hourly_average_fields() {
        let avg = HourlyAverage {
            weekday: 0,
            hour: 10,
            avg_percentage: 45.5,
            sample_count: 100,
        };
        assert_eq!(avg.weekday, 0);
        assert_eq!(avg.hour, 10);
        assert_relative_eq!(avg.avg_percentage, 45.5);
        assert_eq!(avg.sample_count, 100);
    }

    #[test]
    fn test_hourly_average_boundary_values() {
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
