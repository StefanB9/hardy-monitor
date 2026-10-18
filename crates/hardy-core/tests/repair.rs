//! Integration tests for `DataRepairer` against a real database.
#![allow(clippy::float_cmp)]

mod common;

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use hardy_core::{DataRepairer, GymSchedule, db::DataSource, repair::NightlyRepair};

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
        .repair_date_range(date, date, Utc::now(), None)
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
        .repair_date_range(date, date, Utc::now(), None)
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
async fn test_nightly_repair_reads_its_state_only_when_a_repair_is_due() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let schedule = GymSchedule::default();
    let monday = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    let tuesday = monday.succ_opt().context("date")?;
    let after_close = utc(21, 20)?;
    let mut nightly = NightlyRepair::default();
    let first = nightly.run(&tdb.db, &schedule, after_close, BUDGET).await?;

    // Forgetting Monday behind the daemon's back: a remembered state that
    // says nothing is due must not look again.
    let pool = sqlx::PgPool::connect(&tdb.url).await?;
    sqlx::query("UPDATE repair_state SET repaired_through = NULL WHERE id = 1")
        .execute(&pool)
        .await?;
    pool.close().await;
    let remembered = nightly
        .run(
            &tdb.db,
            &schedule,
            after_close + Duration::minutes(5),
            BUDGET,
        )
        .await?;
    let fresh = NightlyRepair::default()
        .run(
            &tdb.db,
            &schedule,
            after_close + Duration::minutes(5),
            BUDGET,
        )
        .await?;
    // Once Tuesday has closed, the remembered state says a repair is due.
    let next_night = nightly
        .run(&tdb.db, &schedule, after_close + Duration::days(1), BUDGET)
        .await?;
    tdb.cleanup().await;

    assert_eq!(first.map(|(days, _)| days), Some((monday, monday)));
    assert!(remembered.is_none(), "nothing due by the remembered state");
    assert_eq!(fresh.map(|(days, _)| days), Some((monday, monday)));
    assert_eq!(next_night.map(|(days, _)| days), Some((tuesday, tuesday)));
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

#[tokio::test]
async fn test_repair_leaves_a_day_without_readings_empty() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let repairer = DataRepairer::new(Arc::new(tdb.db.clone()), GymSchedule::default());
    let day = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;

    let summary = repairer
        .repair_date_range(day, day, Utc::now(), None)
        .await?;
    let rows = tdb.db.get_history_range(utc(0, 0)?, utc(23, 59)?).await?;
    tdb.cleanup().await;

    assert_eq!(summary.boundary_entries_added, 0);
    assert!(rows.is_empty(), "no anchors on an empty day: {rows:?}");
    Ok(())
}

#[tokio::test]
async fn test_repair_removes_anchors_left_on_a_day_without_readings() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    // What an earlier repair left on a day the daemon never collected.
    tdb.db
        .insert_with_source(utc(4, 0)?, 0.0, DataSource::Boundary)
        .await?;
    tdb.db
        .insert_with_source(utc(21, 0)?, 0.0, DataSource::Boundary)
        .await?;
    let repairer = DataRepairer::new(Arc::new(tdb.db.clone()), GymSchedule::default());
    let day = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;

    let summary = repairer
        .repair_date_range(day, day, Utc::now(), None)
        .await?;
    let rows = tdb.db.get_history_range(utc(0, 0)?, utc(23, 59)?).await?;
    tdb.cleanup().await;

    assert_eq!(summary.records_deleted, 2);
    assert!(rows.is_empty(), "stale anchors removed: {rows:?}");
    Ok(())
}

#[tokio::test]
async fn test_repair_leaves_a_day_alone_until_it_has_closed() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    tdb.db.insert_record(utc(8, 0)?, 40.0).await?;
    tdb.db.insert_record(utc(8, 4)?, 60.0).await?;
    let repairer = DataRepairer::new(Arc::new(tdb.db.clone()), GymSchedule::default());
    let day = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;

    // Noon local: the gym is still open, so no closing anchor yet.
    let open = repairer
        .repair_date_range(day, day, utc(10, 0)?, None)
        .await?;
    let during = tdb.db.get_history_range(utc(0, 0)?, utc(23, 59)?).await?;
    // After closing the day is repaired as usual.
    let closed = repairer
        .repair_date_range(day, day, utc(21, 30)?, None)
        .await?;
    tdb.cleanup().await;

    assert_eq!(open.days_processed, 0);
    assert_eq!(during.len(), 2, "untouched while open: {during:?}");
    assert_eq!(closed.days_processed, 1);
    assert_eq!(closed.boundary_entries_added, 2);
    Ok(())
}
