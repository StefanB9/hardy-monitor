//! Integration tests for quarter-hour averages bucketed by gym-local time.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).

mod common;

use anyhow::{Context, Result};
use approx::assert_relative_eq;
use chrono::{DateTime, TimeZone, Utc};
use chrono_tz::Europe::Berlin;
use hardy_core::db::DataSource;

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0)
        .single()
        .context("valid UTC time")
}

#[tokio::test]
async fn test_quarter_hour_averages_average_the_days() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    // Mondays 2024-06-10 and 2024-06-17; 08:15 UTC is 10:15 in Berlin
    // (summer time), quarter 41 of the day.
    for (t, pct) in [
        (utc(2024, 6, 10, 8, 16)?, 20.0),
        (utc(2024, 6, 10, 8, 29)?, 30.0),
        (utc(2024, 6, 17, 8, 20)?, 35.0),
    ] {
        tdb.db.insert_record(t, pct).await.context("insert")?;
    }

    let averages = tdb
        .db
        .get_quarter_hour_averages(utc(2024, 6, 1, 0, 0)?, utc(2024, 7, 1, 0, 0)?, Berlin)
        .await
        .context("query")?;

    assert_eq!(averages.len(), 1, "{averages:?}");
    let slot = &averages[0];
    assert_eq!((slot.weekday, slot.quarter, slot.days), (0, 41, 2));
    // The days' means are 25 and 35: each day counts once, however many
    // readings it has.
    assert_relative_eq!(slot.avg_percentage, 30.0, epsilon = 1e-9);
    assert_relative_eq!(slot.std_dev, 50.0_f64.sqrt(), epsilon = 1e-9);

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_quarter_hour_seen_on_one_day_has_no_spread() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    tdb.db
        .insert_record(utc(2024, 6, 15, 16, 0)?, 12.0)
        .await
        .context("insert")?;

    let averages = tdb
        .db
        .get_quarter_hour_averages(utc(2024, 6, 1, 0, 0)?, utc(2024, 7, 1, 0, 0)?, Berlin)
        .await
        .context("query")?;

    assert_eq!(averages.len(), 1, "{averages:?}");
    // Saturday 18:00 in Berlin.
    assert_eq!((averages[0].weekday, averages[0].quarter), (5, 72));
    assert_eq!(averages[0].days, 1);
    assert_relative_eq!(averages[0].std_dev, 0.0);

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_quarter_hour_averages_skip_boundary_rows_and_other_periods() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    tdb.db
        .insert_with_source(utc(2024, 6, 10, 4, 0)?, 0.0, DataSource::Boundary)
        .await
        .context("insert boundary")?;
    tdb.db
        .insert_record(utc(2024, 5, 31, 23, 59)?, 50.0)
        .await
        .context("insert before")?;
    tdb.db
        .insert_record(utc(2024, 7, 1, 0, 0)?, 50.0)
        .await
        .context("insert at end")?;

    let averages = tdb
        .db
        .get_quarter_hour_averages(utc(2024, 6, 1, 0, 0)?, utc(2024, 7, 1, 0, 0)?, Berlin)
        .await
        .context("query")?;

    assert!(averages.is_empty(), "{averages:?}");

    tdb.cleanup().await;
    Ok(())
}
