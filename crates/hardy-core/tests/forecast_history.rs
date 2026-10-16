//! Storing and reading the hour-ahead forecast of past slots.

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use hardy_core::db::ForecastHistoryEntry;

/// 2024-06-17 (Monday) at `h:mi` UTC.
fn utc(h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0)
        .single()
        .context("valid time")
}

fn past(target: DateTime<Utc>, value: f64, bounds: bool) -> ForecastHistoryEntry {
    ForecastHistoryEntry {
        target,
        made_at: target - TimeDelta::minutes(75),
        model: "naive".to_string(),
        value,
        low: bounds.then_some(value - 4.0),
        high: bounds.then_some(value + 6.0),
    }
}

#[tokio::test]
async fn test_forecast_history_round_trip_keeps_stored_rows() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let db = &tdb.db;
    assert!(!db.has_forecast_history().await?);

    let rows = [
        past(utc(10, 15)?, 30.0, true),
        past(utc(10, 0)?, 20.0, false),
        past(utc(11, 0)?, 40.0, true),
    ];
    assert_eq!(db.save_forecast_history(&rows).await?, 3);
    assert!(db.has_forecast_history().await?);
    // A second save keeps what is stored.
    let again = [past(utc(10, 0)?, 99.0, true), past(utc(12, 0)?, 50.0, true)];
    assert_eq!(db.save_forecast_history(&again).await?, 1);

    let read = db.forecast_history(utc(10, 0)?, utc(12, 0)?).await?;
    // [since, until), oldest first.
    assert_eq!(
        read,
        vec![rows[1].clone(), rows[0].clone(), rows[2].clone()]
    );
    Ok(())
}

#[tokio::test]
async fn test_forecast_history_saves_nothing_from_nothing() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    assert_eq!(tdb.db.save_forecast_history(&[]).await?, 0);
    assert_eq!(
        tdb.db.forecast_history(utc(0, 0)?, utc(23, 0)?).await?,
        Vec::new()
    );
    Ok(())
}
