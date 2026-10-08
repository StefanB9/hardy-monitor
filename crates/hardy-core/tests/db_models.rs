//! Integration tests for stored ML models and the retrain state.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use hardy_core::db::NewModel;

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0)
        .single()
        .context("valid UTC time")
}

fn new_model(trained_at: DateTime<Utc>, feature_version: u32, bytes: &[u8]) -> NewModel<'_> {
    NewModel {
        trained_at,
        feature_version,
        algorithm: "Random Forest",
        training_samples: 1234,
        holdout_mae: 4.5,
        baseline_mae: 6.0,
        tuned: true,
        bytes,
    }
}

#[tokio::test]
async fn test_ml_models_save_latest_and_load() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;

    assert!(tdb.db.latest_model_info(2).await?.is_none());

    let older = tdb
        .db
        .save_model(&new_model(utc(2024, 6, 16, 22, 0)?, 2, b"old"), 3)
        .await?;
    let newer = tdb
        .db
        .save_model(&new_model(utc(2024, 6, 17, 22, 0)?, 2, b"new"), 3)
        .await?;
    // A model for another feature version is never returned for version 2.
    tdb.db
        .save_model(&new_model(utc(2024, 6, 18, 22, 0)?, 1, b"v1"), 3)
        .await?;

    let info = tdb
        .db
        .latest_model_info(2)
        .await?
        .context("a version-2 model is stored")?;
    assert_eq!(info.id, newer);
    assert_ne!(info.id, older);
    assert_eq!(info.training_samples, 1234);
    assert!(info.tuned);
    assert_eq!(info.algorithm, "Random Forest");
    assert_eq!(tdb.db.load_model(info.id).await?, b"new".to_vec());

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_ml_models_keeps_only_newest_per_version() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;

    for day in 10..15 {
        tdb.db
            .save_model(&new_model(utc(2024, 6, day, 22, 0)?, 2, b"m"), 3)
            .await?;
    }
    let ids = tdb.db.model_ids().await?;
    assert_eq!(ids.len(), 3);

    tdb.cleanup().await;
    Ok(())
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
