//! Integration tests for `DataRepairer` against a real database.
#![allow(clippy::float_cmp)]

mod common;

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use hardy_core::{DataRepairer, GymSchedule, db::DataSource};

/// Plenty of time for a nightly repair in tests.
const BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

fn utc(h: u32, mi: u32) -> Result<DateTime<Utc>> {
    // Monday 2024-06-17; CEST = UTC+2, so the gym is open 04:00–21:00 UTC.
    Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0)
        .single()
        .context("valid UTC time")
}

#[tokio::test]
async fn test_repair_labels_provenance_and_is_idempotent() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let db = Arc::new(tdb.db.clone());

    // Measured at 10:00 and 10:04 local with a 3-minute gap between, plus a
    // reading after closing time (23:30 local).
    db.insert_record(utc(8, 0)?, 40.0).await?;
    db.insert_record(utc(8, 4)?, 60.0).await?;
    db.insert_record(utc(21, 30)?, 10.0).await?;

    let date = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    let repairer = DataRepairer::new(db.clone(), GymSchedule::default());
    let summary = repairer
        .repair_date_range(date, date, None)
        .await
        .context("repair should succeed")?;
    assert_eq!(summary.gaps_filled, 3);
    assert_eq!(summary.boundary_entries_added, 2);
    assert_eq!(summary.records_deleted, 1);

    let rows = db.get_history_range(utc(0, 0)?, utc(23, 59)?).await?;
    let by_time = |t: DateTime<Utc>| {
        rows.iter()
            .find(|r| r.timestamp == t)
            .with_context(|| format!("row at {t}"))
    };

    assert_eq!(by_time(utc(4, 0)?)?.source, DataSource::Boundary);
    assert_eq!(by_time(utc(21, 0)?)?.source, DataSource::Boundary);
    for minute in 1..=3 {
        assert_eq!(by_time(utc(8, minute)?)?.source, DataSource::Interpolated);
    }
    for t in [utc(8, 0)?, utc(8, 4)?] {
        assert!(
            matches!(
                by_time(t)?.source,
                DataSource::Measured | DataSource::Smoothed
            ),
            "measured rows stay measured unless repair changed them"
        );
    }
    let after_closing = utc(21, 30)?;
    assert!(
        rows.iter().all(|r| r.timestamp != after_closing),
        "reading after closing time is removed"
    );

    let rows_before = rows.len();
    let again = repairer
        .repair_date_range(date, date, None)
        .await
        .context("second repair should succeed")?;
    assert_eq!(again.gaps_filled + again.boundary_entries_added, 0);
    let rows_after = db.get_history_range(utc(0, 0)?, utc(23, 59)?).await?;
    assert_eq!(rows_after.len(), rows_before, "repair is idempotent");

    drop(repairer);
    drop(db);
    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_nightly_repair_runs_once_per_closed_day() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let schedule = GymSchedule::default();
    let monday = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    // A 3-minute gap on Monday; 23:20 local is after closing + 15 min.
    tdb.db.insert_record(utc(8, 0)?, 40.0).await?;
    tdb.db.insert_record(utc(8, 4)?, 60.0).await?;
    let after_close = utc(21, 20)?;

    let first = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        hardy_core::repair::run_nightly_repair(&tdb.db, &schedule, after_close, BUDGET),
    )
    .await
    .context("repair timed out")??;
    let again =
        hardy_core::repair::run_nightly_repair(&tdb.db, &schedule, after_close, BUDGET).await?;
    let state = tdb.db.get_repair_state().await?;
    let rows = tdb.db.get_history_range(utc(8, 0)?, utc(8, 4)?).await?;
    tdb.cleanup().await;

    let (days, summary) = first.context("Monday was due")?;
    assert_eq!(days, (monday, monday));
    assert_eq!(summary.gaps_filled, 3);
    assert!(again.is_none(), "a repaired day is not repaired twice");
    assert_eq!(state.repaired_through, Some(monday));
    assert_eq!(state.last_error, None);
    assert_eq!(rows.len(), 5);
    Ok(())
}

#[tokio::test]
async fn test_repair_state_records_progress_and_failures() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let initial = tdb.db.get_repair_state().await?;

    let at = Utc
        .with_ymd_and_hms(2024, 6, 18, 21, 30, 0)
        .single()
        .context("time")?;
    let day = NaiveDate::from_ymd_opt(2024, 6, 18).context("date")?;
    tdb.db.record_repair_failure(at, "boom").await?;
    let failed = tdb.db.get_repair_state().await?;
    tdb.db
        .record_repair_success(day, at + Duration::minutes(30))
        .await?;
    let done = tdb.db.get_repair_state().await?;
    tdb.cleanup().await;

    assert_eq!(initial, hardy_core::db::RepairState::default());
    assert_eq!(failed.repaired_through, None);
    assert_eq!(failed.last_attempt_at, Some(at));
    assert_eq!(failed.last_error.as_deref(), Some("boom"));
    assert_eq!(done.repaired_through, Some(day));
    assert_eq!(done.last_attempt_at, Some(at + Duration::minutes(30)));
    assert_eq!(done.last_error, None);
    Ok(())
}

#[tokio::test]
async fn test_nightly_repair_out_of_budget_counts_as_failure() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let schedule = GymSchedule::default();
    tdb.db.insert_record(utc(8, 0)?, 40.0).await?;
    tdb.db.insert_record(utc(8, 4)?, 60.0).await?;
    let after_close = utc(21, 20)?;

    let out_of_time = hardy_core::repair::run_nightly_repair(
        &tdb.db,
        &schedule,
        after_close,
        std::time::Duration::ZERO,
    )
    .await;
    let state = tdb.db.get_repair_state().await?;
    let retried = hardy_core::repair::run_nightly_repair(
        &tdb.db,
        &schedule,
        after_close + Duration::minutes(10),
        BUDGET,
    )
    .await?;
    tdb.cleanup().await;

    let error = out_of_time.err().context("no time means no repair")?;
    assert!(format!("{error:#}").contains("time"), "{error:#}");
    assert_eq!(state.repaired_through, None);
    assert_eq!(state.last_attempt_at, Some(after_close));
    assert!(state.last_error.is_some_and(|e| e.contains("time")));
    assert!(retried.is_none(), "the usual retry delay applies");
    Ok(())
}
