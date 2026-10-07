//! Integration tests for `DataRepairer` against a real database.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]

mod common;

use std::sync::Arc;

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use hardy_core::{DataRepairer, GymSchedule, db::DataSource};

fn utc(h: u32, mi: u32) -> DateTime<Utc> {
    // Monday 2024-06-17; CEST = UTC+2, so the gym is open 04:00–21:00 UTC.
    Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0).unwrap()
}

#[tokio::test]
async fn test_repair_labels_provenance_and_is_idempotent() {
    let tdb = common::TestDatabase::new().await;
    let db = Arc::new(tdb.db.clone());

    // Measured at 10:00 and 10:04 local with a 3-minute gap between, plus a
    // reading after closing time (23:30 local).
    db.insert_record(utc(8, 0), 40.0).await.unwrap();
    db.insert_record(utc(8, 4), 60.0).await.unwrap();
    db.insert_record(utc(21, 30), 10.0).await.unwrap();

    let date = NaiveDate::from_ymd_opt(2024, 6, 17).unwrap();
    let repairer = DataRepairer::new(db.clone(), GymSchedule::default());
    let summary = repairer
        .repair_date_range(date, date, None)
        .await
        .expect("repair should succeed");
    assert_eq!(summary.gaps_filled, 3);
    assert_eq!(summary.boundary_entries_added, 2);
    assert_eq!(summary.records_deleted, 1);

    let rows = db.get_history_range(utc(0, 0), utc(23, 59)).await.unwrap();
    let by_time = |t: DateTime<Utc>| rows.iter().find(|r| r.timestamp == t).unwrap();

    assert_eq!(by_time(utc(4, 0)).source, DataSource::Boundary);
    assert_eq!(by_time(utc(21, 0)).source, DataSource::Boundary);
    for minute in 1..=3 {
        assert_eq!(by_time(utc(8, minute)).source, DataSource::Interpolated);
    }
    for t in [utc(8, 0), utc(8, 4)] {
        assert!(
            matches!(
                by_time(t).source,
                DataSource::Measured | DataSource::Smoothed
            ),
            "measured rows stay measured unless repair changed them"
        );
    }
    assert!(
        rows.iter().all(|r| r.timestamp != utc(21, 30)),
        "reading after closing time is removed"
    );

    let rows_before = rows.len();
    let again = repairer
        .repair_date_range(date, date, None)
        .await
        .expect("second repair should succeed");
    assert_eq!(again.gaps_filled + again.boundary_entries_added, 0);
    let rows_after = db.get_history_range(utc(0, 0), utc(23, 59)).await.unwrap();
    assert_eq!(rows_after.len(), rows_before, "repair is idempotent");

    drop(repairer);
    drop(db);
    tdb.cleanup().await;
}

#[tokio::test]
async fn test_nightly_repair_runs_once_per_closed_day() -> anyhow::Result<()> {
    use anyhow::Context;
    let tdb = common::TestDatabase::new().await;
    let schedule = GymSchedule::default();
    let monday = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
    // A 3-minute gap on Monday; 23:20 local is after closing + 15 min.
    tdb.db.insert_record(utc(8, 0), 40.0).await?;
    tdb.db.insert_record(utc(8, 4), 60.0).await?;
    let after_close = utc(21, 20);

    let first = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        hardy_core::repair::run_nightly_repair(&tdb.db, &schedule, after_close),
    )
    .await
    .context("repair timed out")??;
    let again = hardy_core::repair::run_nightly_repair(&tdb.db, &schedule, after_close).await?;
    let state = tdb.db.get_repair_state().await?;
    let rows = tdb.db.get_history_range(utc(8, 0), utc(8, 4)).await?;
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
async fn test_repair_state_records_progress_and_failures() -> anyhow::Result<()> {
    use anyhow::Context;
    let tdb = common::TestDatabase::new().await;
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
