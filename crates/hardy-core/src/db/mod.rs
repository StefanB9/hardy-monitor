use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, DurationRound, TimeDelta, Utc};
use serde::Serialize;
use sqlx::{
    FromRow, PgPool,
    postgres::{PgConnectOptions, PgPoolOptions},
};

use crate::{config::DatabaseConfig, error::AppError};

mod alert_settings;
mod averages;
mod export;
mod forecast_log;
mod ml;
mod readings;
mod repair_state;
mod schema;

pub use forecast_log::{ForecastLogEntry, HorizonAccuracy};
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

/// Average occupancy of one gym-local weekday (0 = Monday) and hour.
#[derive(Debug, Clone)]
pub struct HourlyAverage {
    pub weekday: i32,
    pub hour: i32,
    pub avg_percentage: f64,
    pub sample_count: i64,
}

/// Connection pool to the occupancy database.
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
        let pool = Self::pool_options(config)
            .connect_with(Self::connect_options(config)?)
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

    /// Connection settings from the URL. Nothing else goes into the startup
    /// message: connection poolers such as `PgBouncer` (Neon's pooled endpoint)
    /// reject startup options.
    pub fn connect_options(config: &DatabaseConfig) -> Result<PgConnectOptions> {
        config
            .url
            .expose()
            .parse()
            .context("DATABASE_URL is not a valid PostgreSQL URL")
    }

    /// Pool limits, and the statement timeout set on each new connection.
    ///
    /// Through a transaction-mode pooler a session setting is not tied to
    /// this client's queries, so there the timeout is best effort; set it on
    /// the database role (`ALTER ROLE … SET statement_timeout`) to enforce it.
    pub fn pool_options(config: &DatabaseConfig) -> PgPoolOptions {
        let timeout = format!("{}s", config.statement_timeout_secs);
        PgPoolOptions::new()
            .max_connections(config.max_connections)
            .acquire_timeout(Duration::from_secs(config.acquire_timeout_secs))
            .after_connect(move |conn, _meta| {
                let timeout = timeout.clone();
                Box::pin(async move {
                    sqlx::query!("SELECT set_config('statement_timeout', $1, false)", timeout)
                        .fetch_one(conn)
                        .await?;
                    Ok(())
                })
            })
    }

    /// Closes the pool, waiting for open connections.
    pub async fn close(self) {
        self.pool.close().await;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::{Datelike, TimeZone, Timelike};

    use super::*;

    #[test]
    fn test_connect_options_send_no_startup_options() -> anyhow::Result<()> {
        // Connection poolers such as PgBouncer (Neon's pooled endpoint)
        // reject startup options, so settings must be applied after
        // connecting.
        let config = DatabaseConfig::with_url("postgres://hardy@db.example/hardy");
        let options = Database::connect_options(&config)?;
        assert_eq!(options.get_options(), None);
        Ok(())
    }

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
