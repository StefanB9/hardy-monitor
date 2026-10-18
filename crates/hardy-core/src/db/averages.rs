//! Average occupancy per gym-local weekday and hour or quarter hour.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use chrono_tz::Tz;

use super::{Database, HourlyAverage, QuarterHourAverage};

impl Database {
    /// Average occupancy per (weekday, hour) slot for records in `[start,
    /// end)`, with the number of days seen and the spread between them.
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
                SUM(day_sum) / SUM(day_n)::FLOAT8 as "avg_percentage!: f64",
                SUM(day_n)::BIGINT as "sample_count!: i64",
                COUNT(*) as "days!: i64",
                COALESCE(STDDEV_SAMP(day_sum / day_n), 0) as "std_dev!: f64"
            FROM (
                SELECT
                    weekday,
                    hour,
                    local_date,
                    SUM(percentage) as day_sum,
                    COUNT(*)::FLOAT8 as day_n
                FROM (
                    SELECT
                        (EXTRACT(ISODOW FROM local)::INTEGER - 1) as weekday,
                        EXTRACT(HOUR FROM local)::INTEGER as hour,
                        local::DATE as local_date,
                        percentage
                    FROM (
                        -- Converted to gym time once per reading.
                        SELECT timestamp AT TIME ZONE $3 as local, percentage
                        FROM occupancy_logs
                        WHERE timestamp >= $1 AND timestamp < $2
                          -- Repair's 0% opening/closing entries are not observations.
                          AND source <> 'boundary'
                    ) AS converted
                ) AS readings
                GROUP BY weekday, hour, local_date
            ) AS days
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

    /// Average occupancy per (weekday, quarter hour) slot for records in
    /// `[start, end)`, gym-local in `tz` like [`Self::get_averages_range`].
    /// Each day's readings are averaged first, so a day with more readings
    /// does not weigh more, and the spread is the spread between days.
    #[tracing::instrument(skip_all, fields(db.operation = "get_quarter_hour_averages", %start, %end, %tz))]
    pub async fn get_quarter_hour_averages(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        tz: Tz,
    ) -> Result<Vec<QuarterHourAverage>> {
        sqlx::query_as!(
            QuarterHourAverage,
            r#"
            SELECT
                weekday as "weekday!: i32",
                quarter as "quarter!: i32",
                AVG(day_mean) as "avg_percentage!: f64",
                COALESCE(STDDEV_SAMP(day_mean), 0) as "std_dev!: f64",
                COUNT(*) as "days!: i64"
            FROM (
                SELECT
                    (EXTRACT(ISODOW FROM local)::INTEGER - 1) as weekday,
                    (EXTRACT(HOUR FROM local)::INTEGER * 4
                        + EXTRACT(MINUTE FROM local)::INTEGER / 15) as quarter,
                    local::DATE as local_date,
                    AVG(percentage) as day_mean
                FROM (
                    SELECT timestamp AT TIME ZONE $3 as local, percentage
                    FROM occupancy_logs
                    WHERE timestamp >= $1 AND timestamp < $2
                      -- Repair's 0% opening/closing entries are not observations.
                      AND source <> 'boundary'
                ) AS readings
                GROUP BY weekday, quarter, local_date
            ) AS days
            GROUP BY weekday, quarter
            ORDER BY weekday, quarter
            "#,
            start,
            end,
            tz.name()
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch quarter-hour averages")
    }
}
