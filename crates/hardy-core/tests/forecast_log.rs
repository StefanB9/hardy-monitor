//! Storing forecasts and scoring them against readings.

mod common;

use anyhow::{Context, Result};
use approx::assert_relative_eq;
use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use hardy_core::db::{DataSource, ForecastLogEntry};

/// 2024-06-17 (Monday) at `h:mi` UTC; Berlin is UTC+2.
fn utc(h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0)
        .single()
        .context("valid time")
}

fn entry(
    made_at: DateTime<Utc>,
    hours: u32,
    predicted: f64,
    baseline: f64,
    model: bool,
) -> ForecastLogEntry {
    ForecastLogEntry {
        made_at,
        target: made_at + TimeDelta::hours(i64::from(hours)),
        horizon_hours: hours,
        model_id: model.then_some(7),
        predicted,
        low: predicted - 5.0,
        high: predicted + 5.0,
        baseline,
    }
}

#[tokio::test]
async fn test_forecast_accuracy_scores_against_readings() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let tz = chrono_tz::Europe::Berlin;
    let db = &tdb.db;

    // Readings at 10:00 and 11:00 UTC; 12:00 only interpolated.
    db.insert_record(utc(10, 0)?, 30.0).await?;
    db.insert_record(utc(11, 0)?, 40.0).await?;
    db.insert_with_source(utc(12, 0)?, 50.0, DataSource::Interpolated)
        .await?;

    let made = utc(9, 0)?;
    let added = db
        .save_forecasts(&[
            entry(made, 1, 34.0, 20.0, true),        // target 10:00: |4|, |10|
            entry(made, 2, 40.0, 50.0, false),       // target 11:00: |0|, |10|
            entry(made, 3, 0.0, 0.0, true),          // target 12:00: no reading
            entry(utc(10, 0)?, 1, 46.0, 40.0, true), // target 11:00: |6|, |0|
        ])
        .await?;
    let again = db
        .save_forecasts(&[entry(made, 1, 99.0, 99.0, true)])
        .await?;

    let rows = db.forecast_accuracy(utc(0, 0)?, utc(23, 0)?, tz).await?;
    let deleted = db.delete_forecasts_before(utc(9, 30)?).await?;
    let after = db.forecast_accuracy(utc(0, 0)?, utc(23, 0)?, tz).await?;
    tdb.cleanup().await;

    assert_eq!(added, 4);
    assert_eq!(again, 0, "a stored forecast is never overwritten");

    let day = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    let h1 = rows
        .iter()
        .find(|r| r.horizon_hours == 1)
        .context("horizon 1")?;
    assert_eq!((h1.day, h1.scored, h1.from_model), (day, 2, 2));
    assert_relative_eq!(h1.forecast_mae, 5.0); // (4 + 6) / 2
    assert_relative_eq!(h1.baseline_mae, 5.0); // (10 + 0) / 2
    let h2 = rows
        .iter()
        .find(|r| r.horizon_hours == 2)
        .context("horizon 2")?;
    assert_eq!((h2.scored, h2.from_model), (1, 0));
    assert_relative_eq!(h2.forecast_mae, 0.0);
    assert_relative_eq!(h2.baseline_mae, 10.0);
    assert!(
        rows.iter().all(|r| r.horizon_hours != 3),
        "unscored horizons are omitted"
    );

    assert_eq!(deleted, 3);
    assert_eq!(after.len(), 1);
    Ok(())
}

#[tokio::test]
async fn test_forecast_accuracy_groups_by_gym_local_day() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let tz = chrono_tz::Europe::Berlin;
    // 22:30 UTC on the 17th is 00:30 on the 18th in Berlin.
    let late = utc(22, 30)?;
    tdb.db.insert_record(late, 10.0).await?;
    tdb.db
        .save_forecasts(&[entry(late - TimeDelta::hours(1), 1, 12.0, 15.0, true)])
        .await?;
    let rows = tdb
        .db
        .forecast_accuracy(utc(0, 0)?, utc(23, 59)?, tz)
        .await?;
    tdb.cleanup().await;

    assert_eq!(
        rows.iter().map(|r| r.day).collect::<Vec<_>>(),
        vec![NaiveDate::from_ymd_opt(2024, 6, 18).context("date")?]
    );
    Ok(())
}
