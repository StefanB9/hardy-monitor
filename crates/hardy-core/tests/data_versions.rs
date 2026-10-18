//! Integration tests for the cheap "has anything new been written"
//! versions of the published forecasts and the forecast suite.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).

mod common;

use anyhow::Result;
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use hardy_core::db::{BenchmarkRun, FitStatus, LiveForecast, ModelUpdate};

fn made(minute: i64) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0)
        .single()
        .unwrap_or_default()
        + TimeDelta::minutes(minute)
}

fn row(model: &str, made_at: DateTime<Utc>) -> LiveForecast {
    LiveForecast {
        model: model.to_string(),
        target: made_at + TimeDelta::minutes(15),
        made_at,
        value: 30.0,
        low: None,
        high: None,
        source: None,
    }
}

#[tokio::test]
async fn test_live_forecasts_version_changes_with_each_publication() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let models = vec!["naive".to_string(), "selected".to_string()];
    let empty = tdb.db.live_forecasts_version(&models).await?;

    tdb.db
        .replace_live_forecasts(&[row("naive", made(0)), row("selected", made(0))])
        .await?;
    let first = tdb.db.live_forecasts_version(&models).await?;
    assert_ne!(first, empty);
    assert_eq!(tdb.db.live_forecasts_version(&models).await?, first);

    tdb.db
        .replace_live_forecasts(&[row("naive", made(15)), row("selected", made(15))])
        .await?;
    let second = tdb.db.live_forecasts_version(&models).await?;
    assert_ne!(second, first);

    // A model dropping out changes it too, even at the same time.
    tdb.db
        .replace_live_forecasts(&[row("selected", made(15))])
        .await?;
    assert_ne!(tdb.db.live_forecasts_version(&models).await?, second);

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_suite_version_changes_with_a_run_request_and_attempt() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let start = tdb.db.suite_version().await?;
    assert_eq!(tdb.db.suite_version().await?, start);

    tdb.db.request_retrain(made(0)).await?;
    let requested = tdb.db.suite_version().await?;
    assert_ne!(requested, start);

    tdb.db.start_training_attempt(made(1)).await?;
    let attempted = tdb.db.suite_version().await?;
    assert_ne!(attempted, requested);

    tdb.db.finish_training_attempt(Some("failed")).await?;
    let finished = tdb.db.suite_version().await?;
    assert_ne!(finished, attempted);

    let update = ModelUpdate {
        model: "naive",
        status: FitStatus::Fitted,
        message: None,
        fitted_at: Some(made(2)),
        data_from: None,
        data_until: None,
        artifact: None,
    };
    tdb.db.save_model_states(&[update], made(2)).await?;
    let fitted = tdb.db.suite_version().await?;
    assert_ne!(fitted, finished);

    let run = BenchmarkRun {
        id: 0,
        started_at: made(3),
        finished_at: made(4),
        first_day: None,
        last_day: None,
        days: 1,
        origins: 1,
        models: Vec::new(),
        scores: Vec::new(),
    };
    tdb.db
        .save_benchmark(&run, &[(1, "naive".to_string())], 5)
        .await?;
    assert_ne!(tdb.db.suite_version().await?, fitted);

    tdb.cleanup().await;
    Ok(())
}
