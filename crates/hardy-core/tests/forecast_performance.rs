//! The data behind the forecast performance page: every model's logged
//! live forecasts, the scored forecast history and the benchmark history.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).

mod common;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use chrono_tz::Europe::Berlin;
use hardy_core::{
    accuracy::{ErrorSums, Pair},
    db::{DataSource, ForecastHistoryEntry, ModelForecast},
};

/// Steps logged per forecast (6 h of quarter hours).
const STEPS: usize = 24;

/// 2024-06-17 (Monday, CEST) at `h:mi` UTC.
fn utc(h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0)
        .single()
        .context("valid time")
}

/// A logged forecast with `value` (and `band`) at element `index` only.
fn logged(model: &str, index: usize, value: f64, band: Option<(f64, f64)>) -> ModelForecast {
    let mut forecast = vec![None; STEPS];
    let mut low = vec![None; STEPS];
    let mut high = vec![None; STEPS];
    forecast[index] = Some(value);
    low[index] = band.map(|(l, _)| l);
    high[index] = band.map(|(_, h)| h);
    ModelForecast {
        model: model.to_string(),
        forecast,
        low,
        high,
    }
}

/// Readings for the target 10:00 UTC: 40 then, and 20 and 30 one and two
/// weeks before (so plain averages say 25 and last week says 20).
async fn readings(tdb: &common::TestDatabase) -> Result<DateTime<Utc>> {
    let target = utc(10, 0)?;
    tdb.db.insert_record(target, 40.0).await?;
    tdb.db
        .insert_record(target - TimeDelta::days(7), 20.0)
        .await?;
    tdb.db
        .insert_record(target - TimeDelta::days(14), 30.0)
        .await?;
    // Interpolated readings are not scored.
    tdb.db
        .insert_with_source(utc(10, 30)?, 10.0, DataSource::Interpolated)
        .await?;
    Ok(target)
}

fn sums(pair: &Pair) -> ErrorSums {
    let mut sums = ErrorSums::default();
    sums.add(pair);
    sums
}

#[tokio::test]
async fn test_model_forecast_errors_by_step_and_by_day() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    readings(&tdb).await?;
    let made_at = utc(9, 0)?;
    // Element 3 is 10:00; element 5 (10:30) has only an interpolated
    // reading and element 4 (10:15) none.
    let mut naive = logged("naive", 3, 36.0, Some((30.0, 44.0)));
    naive.forecast[4] = Some(50.0);
    naive.forecast[5] = Some(50.0);
    let rows = [naive, logged("selected", 3, 50.0, None)];
    let added = tdb.db.save_model_forecasts(made_at, &rows).await?;
    let again = tdb.db.save_model_forecasts(made_at, &rows).await?;

    let errors = tdb
        .db
        .model_forecast_errors(utc(0, 0)?, utc(23, 0)?, Berlin)
        .await?;
    // Nothing for targets outside the period.
    let outside = tdb
        .db
        .model_forecast_errors(utc(10, 15)?, utc(23, 0)?, Berlin)
        .await?;
    tdb.cleanup().await;

    assert_eq!((added, again), (2, 0), "stored rows are kept");
    let naive_pair = Pair {
        actual: 40.0,
        forecast: 36.0,
        band: Some((30.0, 44.0)),
        baseline: Some(25.0),
        last_week: Some(20.0),
    };
    let selected_pair = Pair {
        forecast: 50.0,
        band: None,
        ..naive_pair
    };
    assert_eq!(
        errors.by_step,
        vec![
            ("naive".to_string(), 4, sums(&naive_pair)),
            ("selected".to_string(), 4, sums(&selected_pair)),
        ]
    );
    let monday = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    assert_eq!(
        errors.by_day,
        vec![
            ("naive".to_string(), monday, sums(&naive_pair)),
            ("selected".to_string(), monday, sums(&selected_pair)),
        ]
    );
    assert!(outside.by_step.is_empty() && outside.by_day.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_model_forecasts_must_have_equal_lengths() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let mut short = logged("naive", 3, 36.0, None);
    short.low.pop();
    let result = tdb.db.save_model_forecasts(utc(9, 0)?, &[short]).await;
    tdb.cleanup().await;
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_scored_forecast_history_pairs_with_readings() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let target = readings(&tdb).await?;
    let entry = |target: DateTime<Utc>, value: f64| ForecastHistoryEntry {
        target,
        made_at: target - TimeDelta::hours(1),
        model: "naive".to_string(),
        value,
        low: Some(value - 4.0),
        high: Some(value + 6.0),
    };
    tdb.db
        .save_forecast_history(&[
            entry(target, 38.0),
            // No reading, and only an interpolated one: not scored.
            entry(utc(10, 15)?, 30.0),
            entry(utc(10, 30)?, 30.0),
        ])
        .await?;

    let scored = tdb
        .db
        .scored_forecast_history(utc(0, 0)?, utc(23, 0)?, Berlin)
        .await?;
    tdb.cleanup().await;

    assert_eq!(scored.len(), 1, "{scored:?}");
    let first = &scored[0];
    assert_eq!(first.target, target);
    assert_eq!(first.model, "naive");
    assert_eq!(
        first.pair,
        Pair {
            actual: 40.0,
            forecast: 38.0,
            band: Some((34.0, 44.0)),
            baseline: Some(25.0),
            last_week: Some(20.0),
        }
    );
    Ok(())
}
