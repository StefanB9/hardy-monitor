//! Average occupancy per gym-local weekday and hour.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use chrono_tz::Tz;

use super::{Database, HourlyAverage};

impl Database {
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
}
