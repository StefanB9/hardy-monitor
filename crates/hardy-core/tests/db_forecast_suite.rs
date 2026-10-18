//! Integration tests for the forecasting suite's tables: model states,
//! benchmark runs with the selection, and live forecasts.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).
#![allow(clippy::float_cmp)]

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use hardy_core::db::{
    BenchmarkModel, BenchmarkRun, BenchmarkScore, FitStatus, LiveForecast, ModelUpdate,
};

fn utc(d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(2024, 6, d, h, mi, 0)
        .single()
        .context("valid UTC time")
}

fn fitted<'a>(model: &'a str, at: DateTime<Utc>, bytes: &'a [u8]) -> ModelUpdate<'a> {
    ModelUpdate {
        model,
        status: FitStatus::Fitted,
        message: None,
        fitted_at: Some(at),
        data_from: Some(at - TimeDelta::days(30)),
        data_until: Some(at),
        artifact: Some((1, bytes)),
    }
}

fn run(at: DateTime<Utc>, models: &[(&str, u32, [f64; 2])]) -> BenchmarkRun {
    BenchmarkRun {
        id: 0,
        started_at: at,
        finished_at: at + TimeDelta::minutes(5),
        first_day: NaiveDate::from_ymd_opt(2024, 5, 1),
        last_day: NaiveDate::from_ymd_opt(2024, 5, 28),
        days: 28,
        origins: 500,
        models: models
            .iter()
            .map(|(model, origins, _)| BenchmarkModel {
                model: (*model).to_string(),
                origins: *origins,
                fitted_days: 20,
                eligible: *origins >= 24,
                failure: (*origins < 24).then(|| "too few origins".to_string()),
            })
            .collect(),
        scores: models
            .iter()
            .flat_map(|(model, _, maes)| {
                [(0_i16, maes[0]), (1, maes[1])].map(|(bucket, mae)| BenchmarkScore {
                    model: (*model).to_string(),
                    bucket,
                    mae,
                    rmse: mae * 1.5,
                    mase: Some(mae / 4.0),
                    coverage: (bucket > 0).then_some(0.8),
                    pinball: None,
                    errors: 100,
                    interval: (bucket > 0).then_some((-3.0, 4.0)),
                })
            })
            .collect(),
    }
}

#[tokio::test]
async fn test_model_states_upsert_and_keep_artifacts_by_version() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let at = utc(3, 23, 30)?;
    tdb.db
        .save_model_states(
            &[
                fitted("naive", at, b"n1"),
                ModelUpdate {
                    model: "weekly_seasonal_naive",
                    status: FitStatus::Waiting,
                    message: Some("needs 7 days, has 3"),
                    fitted_at: None,
                    data_from: None,
                    data_until: None,
                    artifact: None,
                },
            ],
            at,
        )
        .await?;
    // A later fit replaces the earlier one.
    let later = at + TimeDelta::days(1);
    tdb.db
        .save_model_states(&[fitted("naive", later, b"n2")], later)
        .await?;

    let records = tdb.db.model_records().await?;
    let artifacts = tdb.db.model_artifacts(1).await?;
    let other_version = tdb.db.model_artifacts(2).await?;
    tdb.cleanup().await;

    assert_eq!(records.len(), 2);
    let naive = records
        .iter()
        .find(|r| r.model == "naive")
        .context("naive stored")?;
    assert_eq!(naive.status, FitStatus::Fitted);
    assert_eq!(naive.fitted_at, Some(later));
    assert_eq!(naive.updated_at, later);
    let weekly = records
        .iter()
        .find(|r| r.model == "weekly_seasonal_naive")
        .context("weekly stored")?;
    assert_eq!(weekly.status, FitStatus::Waiting);
    assert_eq!(weekly.message.as_deref(), Some("needs 7 days, has 3"));
    assert_eq!(artifacts, vec![("naive".to_string(), b"n2".to_vec())]);
    assert_eq!(other_version, Vec::new());
    Ok(())
}

#[tokio::test]
async fn test_benchmark_round_trips_with_its_selection() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let empty = tdb.db.latest_benchmark().await?;
    let at = utc(3, 23, 30)?;
    let id = tdb
        .db
        .save_benchmark(
            &run(
                at,
                &[("naive", 400, [6.0, 2.0]), ("weekly", 10, [5.0, 3.0])],
            ),
            &[(1, "naive".to_string())],
            5,
        )
        .await?;
    let latest = tdb.db.latest_benchmark().await?.context("run stored")?;
    let selection = tdb.db.forecast_selection().await?;
    tdb.cleanup().await;

    assert!(empty.is_none());
    assert_eq!(latest.id, id);
    assert_eq!(latest.started_at, at);
    assert_eq!(latest.days, 28);
    assert_eq!(latest.models.len(), 2);
    let weekly = latest
        .models
        .iter()
        .find(|m| m.model == "weekly")
        .context("weekly")?;
    assert!(!weekly.eligible);
    assert_eq!(latest.scores.len(), 4);
    let next_hour = latest
        .scores
        .iter()
        .find(|s| s.model == "naive" && s.bucket == 1)
        .context("naive next hour")?;
    assert_eq!(next_hour.mae, 2.0);
    assert_eq!(next_hour.interval, Some((-3.0, 4.0)));
    assert_eq!(selection, vec![(1, "naive".to_string())]);
    Ok(())
}

#[tokio::test]
async fn test_benchmark_keeps_only_the_newest_runs_and_replaces_the_selection() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let mut ids = Vec::new();
    for day in 1..=4 {
        let selected = if day % 2 == 0 { "naive" } else { "weekly" };
        ids.push(
            tdb.db
                .save_benchmark(
                    &run(utc(day, 23, 30)?, &[("naive", 400, [6.0, 2.0])]),
                    &[(1, selected.to_string()), (2, "naive".to_string())],
                    2,
                )
                .await?,
        );
    }
    let pool = sqlx::PgPool::connect(&tdb.url).await?;
    let runs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM benchmark_runs")
        .fetch_one(&pool)
        .await?;
    pool.close().await;
    let latest = tdb.db.latest_benchmark().await?.context("run stored")?;
    let selection = tdb.db.forecast_selection().await?;
    tdb.cleanup().await;

    assert_eq!(runs, 2);
    assert_eq!(Some(&latest.id), ids.last());
    assert_eq!(
        selection,
        vec![(1, "naive".to_string()), (2, "naive".to_string())]
    );
    Ok(())
}

#[tokio::test]
async fn test_live_forecasts_are_replaced_as_a_whole() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let made = utc(3, 10, 0)?;
    let row = |model: &str, minutes: i64, value: f64, made_at: DateTime<Utc>| LiveForecast {
        model: model.to_string(),
        target: made_at + TimeDelta::minutes(minutes),
        made_at,
        value,
        low: Some(value - 5.0),
        high: Some(value + 5.0),
        source: (model == "selected").then(|| "naive".to_string()),
    };
    tdb.db
        .replace_live_forecasts(&[
            row("naive", 15, 30.0, made),
            row("naive", 30, 31.0, made),
            row("selected", 15, 30.0, made),
            row("weekly", 15, 40.0, made),
        ])
        .await?;
    // Fifteen minutes later the weekly model has no forecast any more.
    let later = made + TimeDelta::minutes(15);
    tdb.db
        .replace_live_forecasts(&[
            row("naive", 15, 33.0, later),
            row("selected", 15, 33.0, later),
        ])
        .await?;

    let naive = tdb.db.live_forecasts(&["naive".to_string()]).await?;
    let both = tdb
        .db
        .live_forecasts(&["selected".to_string(), "weekly".to_string()])
        .await?;
    tdb.cleanup().await;

    assert_eq!(naive.len(), 1, "older rows of a republished model go");
    assert_eq!(naive[0].value, 33.0);
    assert_eq!(naive[0].made_at, later);
    assert_eq!(both.len(), 1, "a model missing from the new set is gone");
    assert_eq!(both[0].model, "selected");
    assert_eq!(both[0].source.as_deref(), Some("naive"));
    Ok(())
}

#[test]
fn test_fit_status_round_trips_through_its_text() {
    for status in [FitStatus::Fitted, FitStatus::Waiting, FitStatus::Failed] {
        assert_eq!(FitStatus::parse(status.as_str()), Some(status));
    }
    assert_eq!(FitStatus::parse("unknown"), None);
}
