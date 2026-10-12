//! Integration tests for the forecasting run's coordination state.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0)
        .single()
        .context("valid UTC time")
}

#[tokio::test]
async fn test_ml_state_retrain_request_lifecycle() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;

    let state = tdb.db.get_ml_state().await?;
    assert_eq!(state.retrain_requested_at, None);
    assert_eq!(state.last_attempt_at, None);
    assert_eq!(state.last_error, None);

    tdb.db.request_retrain(utc(2024, 6, 17, 10, 0)?).await?;
    assert_eq!(
        tdb.db.get_ml_state().await?.retrain_requested_at,
        Some(utc(2024, 6, 17, 10, 0)?)
    );

    // Starting an attempt consumes requests made before it.
    tdb.db
        .start_training_attempt(utc(2024, 6, 17, 10, 1)?)
        .await?;
    let started = tdb.db.get_ml_state().await?;
    assert_eq!(started.retrain_requested_at, None);
    assert_eq!(started.last_attempt_at, Some(utc(2024, 6, 17, 10, 1)?));

    tdb.db
        .finish_training_attempt(Some("not enough data"))
        .await?;
    assert_eq!(
        tdb.db.get_ml_state().await?.last_error.as_deref(),
        Some("not enough data")
    );
    tdb.db.finish_training_attempt(None).await?;
    assert_eq!(tdb.db.get_ml_state().await?.last_error, None);

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_ml_state_request_during_training_survives() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;

    tdb.db
        .start_training_attempt(utc(2024, 6, 17, 10, 0)?)
        .await?;
    tdb.db.request_retrain(utc(2024, 6, 17, 10, 5)?).await?;
    tdb.db.finish_training_attempt(None).await?;
    assert_eq!(
        tdb.db.get_ml_state().await?.retrain_requested_at,
        Some(utc(2024, 6, 17, 10, 5)?)
    );

    tdb.cleanup().await;
    Ok(())
}
