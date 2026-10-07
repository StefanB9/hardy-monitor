//! Occupancy readings: storing, reading and the batch edits Data Repair makes.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use chrono_tz::Tz;

use super::{DataSource, Database, OccupancyLog, minute_slot};

impl Database {
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

    /// The newest stored row of any source.
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

    /// All rows with `start <= timestamp <= end`, oldest first.
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

    /// Deletes one row by id.
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
}
