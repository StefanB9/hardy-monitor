use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use chrono::{DateTime, DurationRound, NaiveDate, TimeDelta, Utc};
use chrono_tz::Tz;
use futures::TryStreamExt;
use serde::Serialize;
use sqlx::{FromRow, PgPool, postgres::PgPoolOptions};

use crate::{config::DatabaseConfig, error::AppError, traits::Clock};

mod alert_settings;
mod ml;
mod repair_state;
mod schema;

pub use ml::{MlState, ModelInfo, NewModel};
pub use repair_state::RepairState;
pub use schema::{Migrations, SchemaStatus, app_schema_version};

/// Where a stored value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum DataSource {
    /// Reported by the gym API and stored by the daemon.
    Measured,
    /// Filled into a gap by Data Repair.
    Interpolated,
    /// Zero entry at opening or closing time added or set by Data Repair.
    Boundary,
    /// A measured value that Data Repair changed (outlier or smoothing).
    Smoothed,
}

impl DataSource {
    /// The value stored in the `source` column.
    pub fn as_str(self) -> &'static str {
        match self {
            DataSource::Measured => "measured",
            DataSource::Interpolated => "interpolated",
            DataSource::Boundary => "boundary",
            DataSource::Smoothed => "smoothed",
        }
    }
}

/// One occupancy value per UTC minute.
#[derive(Debug, Clone, FromRow, Serialize)]
pub struct OccupancyLog {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub percentage: f64,
    pub source: DataSource,
}

/// The UTC minute a timestamp belongs to; the database stores exactly one
/// row per minute slot.
pub fn minute_slot(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    timestamp
        .duration_trunc(TimeDelta::minutes(1))
        .unwrap_or(timestamp)
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
    /// Connects with default pool settings and runs pending migrations.
    pub async fn new(database_url: &str) -> Result<Self> {
        Self::connect(&DatabaseConfig::with_url(database_url), Migrations::Apply).await
    }

    /// Connects with the configured pool limits; applies pending migrations
    /// only with [`Migrations::Apply`].
    #[tracing::instrument(skip_all, fields(
        max_connections = config.max_connections,
        acquire_timeout_secs = config.acquire_timeout_secs,
        ?migrations,
    ))]
    pub async fn connect(config: &DatabaseConfig, migrations: Migrations) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .acquire_timeout(Duration::from_secs(config.acquire_timeout_secs))
            .connect(&config.url)
            .await
            .context("Failed to connect to PostgreSQL database")?;

        if migrations == Migrations::Apply {
            // sqlx would also refuse, but with an obscure "missing migration"
            // message that hides what to do.
            let app = schema::app_schema_version();
            if let Some(db) = schema::applied_version(&pool).await?
                && db > app
            {
                pool.close().await;
                return Err(AppError::SchemaTooNew { db, app }.into());
            }
            schema::MIGRATOR
                .run(&pool)
                .await
                .context("Failed to run database migrations")?;
        }

        Ok(Self { pool })
    }

    /// Stores a measured reading in its minute slot. Returns `None` when the
    /// slot is already filled; the existing row is kept.
    pub async fn insert_record(
        &self,
        timestamp: DateTime<Utc>,
        percentage: f64,
    ) -> Result<Option<i64>> {
        self.insert_with_source(timestamp, percentage, DataSource::Measured)
            .await
    }

    /// Stores a value with the given provenance in its minute slot. Returns
    /// `None` when the slot is already filled; the existing row is kept.
    #[tracing::instrument(skip_all, fields(db.operation = "insert", %timestamp, source = source.as_str()))]
    pub async fn insert_with_source(
        &self,
        timestamp: DateTime<Utc>,
        percentage: f64,
        source: DataSource,
    ) -> Result<Option<i64>> {
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO occupancy_logs (timestamp, percentage, source)
            VALUES ($1, $2, $3)
            ON CONFLICT (timestamp) DO NOTHING
            RETURNING id
            "#,
            minute_slot(timestamp),
            percentage,
            source.as_str()
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to insert occupancy record")?;

        Ok(id)
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
                percentage as "percentage!",
                source as "source!: DataSource"
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
                percentage as "percentage!",
                source as "source!: DataSource"
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
                percentage as "percentage!",
                source as "source!: DataSource"
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

    /// Average occupancy per (weekday, hour) slot for records in `[start,
    /// end)`.
    ///
    /// Data Repair's artificial opening/closing (`boundary`) rows are
    /// excluded. Slots are wall-clock time in `tz` (weekday 0 = Monday),
    /// computed by
    /// Postgres with the IANA rules, so a slot means the same local hour on
    /// both sides of a DST change and does not depend on the session or host
    /// timezone.
    #[tracing::instrument(skip_all, fields(db.operation = "get_averages_range", %start, %end, %tz))]
    pub async fn get_averages_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        tz: Tz,
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
                    (EXTRACT(ISODOW FROM timestamp AT TIME ZONE $3)::INTEGER - 1) as weekday,
                    EXTRACT(HOUR FROM timestamp AT TIME ZONE $3)::INTEGER as hour,
                    percentage
                FROM occupancy_logs
                WHERE timestamp >= $1 AND timestamp < $2
                  -- Repair's 0% opening/closing entries are not observations.
                  AND source <> 'boundary'
            ) AS subquery
            GROUP BY weekday, hour
            ORDER BY weekday, hour
            "#,
            start,
            end,
            tz.name()
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
                percentage as "percentage!",
                source as "source!: DataSource"
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

    /// All records whose timestamp falls on `date` as observed in `tz`.
    #[tracing::instrument(skip_all, fields(db.operation = "get_records_for_date", %date, %tz))]
    pub async fn get_records_for_date(&self, date: NaiveDate, tz: Tz) -> Result<Vec<OccupancyLog>> {
        let start_of_day = crate::analytics::midnight_local_as_utc(date, tz);
        let next_day = date
            .succ_opt()
            .with_context(|| format!("no day after {date}"))?;
        // SAFETY: Postgres stores microseconds, so the last representable
        // instant before the next local midnight closes the inclusive range
        // without dropping sub-second records at 23:59:59.
        let end_of_day = crate::analytics::midnight_local_as_utc(next_day, tz)
            - chrono::Duration::microseconds(1);

        self.get_history_range(start_of_day, end_of_day).await
    }

    /// Stores many values with one provenance in a single statement, skipping
    /// minute slots that are already filled. Returns the number of rows
    /// inserted.
    #[tracing::instrument(skip_all, fields(db.operation = "batch_insert", count = records.len(), source = source.as_str()))]
    pub async fn batch_insert(
        &self,
        records: &[(DateTime<Utc>, f64)],
        source: DataSource,
    ) -> Result<u64> {
        let mut timestamps = Vec::with_capacity(records.len());
        let mut percentages = Vec::with_capacity(records.len());
        for &(timestamp, percentage) in records {
            timestamps.push(minute_slot(timestamp));
            percentages.push(percentage);
        }

        let result = sqlx::query!(
            r#"
            INSERT INTO occupancy_logs (timestamp, percentage, source)
            SELECT t, p, $3
            FROM UNNEST($1::timestamptz[], $2::float8[]) AS u(t, p)
            ON CONFLICT (timestamp) DO NOTHING
            "#,
            &timestamps,
            &percentages,
            source.as_str()
        )
        .execute(&self.pool)
        .await
        .context("failed to batch insert records")?;

        Ok(result.rows_affected())
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

    /// Sets new percentages in a single statement. Rows that were
    /// `measured` are re-labelled `source_if_measured`; repaired rows keep
    /// their provenance. Returns the number of rows updated.
    #[tracing::instrument(skip_all, fields(db.operation = "batch_update_percentage", count = updates.len()))]
    pub async fn batch_update_percentage(
        &self,
        updates: &[(i64, f64)],
        source_if_measured: DataSource,
    ) -> Result<u64> {
        let mut ids = Vec::with_capacity(updates.len());
        let mut percentages = Vec::with_capacity(updates.len());
        for &(id, percentage) in updates {
            ids.push(id);
            percentages.push(percentage);
        }

        let result = sqlx::query!(
            r#"
            UPDATE occupancy_logs AS o
            SET percentage = u.p,
                source = CASE WHEN o.source = 'measured' THEN $3 ELSE o.source END
            FROM UNNEST($1::int8[], $2::float8[]) AS u(id, p)
            WHERE o.id = u.id
            "#,
            &ids,
            &percentages,
            source_if_measured.as_str()
        )
        .execute(&self.pool)
        .await
        .context("Failed to batch update percentages")?;

        Ok(result.rows_affected())
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
            source: DataSource::Measured,
        }
    }

    #[test]
    fn test_minute_slot_truncates_seconds_and_subseconds() -> Result<()> {
        let ts = Utc
            .with_ymd_and_hms(2024, 6, 15, 14, 30, 59)
            .single()
            .context("valid time")?
            + chrono::Duration::milliseconds(999);
        let expected = Utc
            .with_ymd_and_hms(2024, 6, 15, 14, 30, 0)
            .single()
            .context("valid time")?;
        assert_eq!(minute_slot(ts), expected);
        assert_eq!(minute_slot(expected), expected);
        Ok(())
    }

    #[test]
    fn test_data_source_as_str_matches_serde() -> Result<()> {
        for source in [
            DataSource::Measured,
            DataSource::Interpolated,
            DataSource::Boundary,
            DataSource::Smoothed,
        ] {
            let mut wtr = csv::Writer::from_writer(Vec::new());
            wtr.serialize([source])?;
            let bytes = wtr.into_inner().context("flush csv")?;
            assert_eq!(String::from_utf8(bytes)?.trim(), source.as_str());
        }
        Ok(())
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
