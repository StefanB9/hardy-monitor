//! Integration tests for database operations.
//!
//! Each test creates and drops its own isolated `PostgreSQL` database via
//! `common::TestDatabase`, ensuring tests never read or write production data
//! and run deterministically regardless of pre-existing state.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]

mod common;

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use chrono_tz::{America::New_York, Europe::Berlin};
use hardy_core::MockClock;

#[tokio::test]
async fn test_database_creation() {
    let tdb = common::TestDatabase::new().await;
    tdb.cleanup().await;
}

#[tokio::test]
async fn test_insert_record() {
    let tdb = common::TestDatabase::new().await;

    let id = tdb
        .db
        .insert_record(Utc::now(), 50.0)
        .await
        .expect("insert should succeed");

    assert!(id > 0, "INSERT should return a positive ID");

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_insert_and_get_history() {
    let tdb = common::TestDatabase::new().await;

    let now = Utc::now();
    for i in 0..5i64 {
        tdb.db
            .insert_record(now - Duration::hours(i), (i as f64) * 10.0)
            .await
            .expect("insert should succeed");
    }

    let history = tdb
        .db
        .get_history(1)
        .await
        .expect("get_history should succeed");

    assert_eq!(
        history.len(),
        5,
        "clean DB should contain exactly 5 records"
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_history_range() {
    let tdb = common::TestDatabase::new().await;

    let now = Utc::now();
    for i in 0..6i64 {
        tdb.db
            .insert_record(now - Duration::hours(i), 50.0)
            .await
            .expect("insert should succeed");
    }

    let history = tdb
        .db
        .get_history_range(now - Duration::hours(2), now + Duration::hours(1))
        .await
        .expect("range query should succeed");

    assert_eq!(
        history.len(),
        3,
        "window [now-2h, now+1h] should capture exactly 3 records"
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_averages_range() {
    let tdb = common::TestDatabase::new().await;

    let base_time = Utc.with_ymd_and_hms(2024, 6, 15, 10, 30, 0).unwrap();

    for i in 0..3i64 {
        tdb.db
            .insert_record(
                base_time - Duration::minutes(i * 10),
                30.0 + (i as f64) * 10.0,
            )
            .await
            .expect("insert should succeed");
    }

    let averages = tdb
        .db
        .get_averages_range(
            base_time - Duration::hours(1),
            base_time + Duration::hours(1),
            Berlin,
        )
        .await
        .expect("averages query should succeed");

    assert_eq!(
        averages.len(),
        1,
        "all three records fall in hour 10, so exactly one hourly bucket expected"
    );
    assert!(
        (averages[0].avg_percentage - 40.0).abs() < 0.001,
        "average of 30, 40, 50 should be 40.0; got {:.4}",
        averages[0].avg_percentage
    );

    tdb.cleanup().await;
}

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
}

/// Returns `(weekday, hour, avg)` triples for compact assertions.
async fn bucket_averages(
    tdb: &common::TestDatabase,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    tz: chrono_tz::Tz,
) -> Vec<(i32, i32, f64)> {
    tdb.db
        .get_averages_range(start, end, tz)
        .await
        .expect("averages query should succeed")
        .into_iter()
        .map(|a| (a.weekday, a.hour, a.avg_percentage))
        .collect()
}

#[tokio::test]
async fn test_get_averages_range_buckets_by_gym_local_hour() {
    let tdb = common::TestDatabase::new().await;

    // Saturday 10:30 UTC is 12:30 CEST in the gym.
    tdb.db
        .insert_record(utc(2024, 6, 15, 10, 30), 42.0)
        .await
        .expect("insert should succeed");

    let buckets =
        bucket_averages(&tdb, utc(2024, 6, 15, 0, 0), utc(2024, 6, 16, 0, 0), Berlin).await;
    assert_eq!(buckets, vec![(5, 12, 42.0)]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_averages_range_merges_same_local_hour_across_dst() {
    let tdb = common::TestDatabase::new().await;

    // Both are Mondays at 10:15 gym-local time: CET (UTC+1) and CEST (UTC+2).
    tdb.db
        .insert_record(utc(2024, 1, 15, 9, 15), 20.0)
        .await
        .expect("insert should succeed");
    tdb.db
        .insert_record(utc(2024, 7, 15, 8, 15), 40.0)
        .await
        .expect("insert should succeed");

    let buckets =
        bucket_averages(&tdb, utc(2024, 1, 1, 0, 0), utc(2024, 12, 31, 0, 0), Berlin).await;
    assert_eq!(buckets, vec![(0, 10, 30.0)]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_averages_range_local_weekday_rolls_over_at_local_midnight() {
    let tdb = common::TestDatabase::new().await;

    // Sunday 22:30 UTC is already Monday 00:30 CEST.
    tdb.db
        .insert_record(utc(2024, 6, 16, 22, 30), 5.0)
        .await
        .expect("insert should succeed");

    let buckets =
        bucket_averages(&tdb, utc(2024, 6, 16, 0, 0), utc(2024, 6, 17, 6, 0), Berlin).await;
    assert_eq!(buckets, vec![(0, 0, 5.0)]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_averages_range_respects_requested_timezone() {
    let tdb = common::TestDatabase::new().await;

    // Saturday 10:30 UTC is 06:30 EDT.
    tdb.db
        .insert_record(utc(2024, 6, 15, 10, 30), 42.0)
        .await
        .expect("insert should succeed");

    let buckets = bucket_averages(
        &tdb,
        utc(2024, 6, 15, 0, 0),
        utc(2024, 6, 16, 0, 0),
        New_York,
    )
    .await;
    assert_eq!(buckets, vec![(5, 6, 42.0)]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_records_for_date_uses_gym_local_day() {
    let tdb = common::TestDatabase::new().await;

    let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
    // 00:30 local on the 15th (previous UTC day) — inside.
    tdb.db
        .insert_record(utc(2024, 6, 14, 22, 30), 1.0)
        .await
        .expect("insert should succeed");
    // 23:59:59.5 local on the 15th — inside (sub-second before midnight).
    tdb.db
        .insert_record(
            utc(2024, 6, 15, 21, 59) + Duration::milliseconds(59_500),
            2.0,
        )
        .await
        .expect("insert should succeed");
    // 00:30 local on the 16th (same UTC day) — outside.
    tdb.db
        .insert_record(utc(2024, 6, 15, 22, 30), 3.0)
        .await
        .expect("insert should succeed");

    let records = tdb
        .db
        .get_records_for_date(date, Berlin)
        .await
        .expect("query should succeed");
    let values: Vec<f64> = records.iter().map(|r| r.percentage).collect();
    assert_eq!(values, vec![1.0, 2.0]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_inserts() {
    let tdb = common::TestDatabase::new().await;

    let now = Utc::now();
    let mut handles = Vec::new();
    for i in 0..10i64 {
        let db_clone = tdb.db.clone();
        let ts = now - Duration::seconds(i);
        handles.push(tokio::spawn(async move {
            db_clone.insert_record(ts, i as f64).await
        }));
    }

    for handle in handles {
        handle
            .await
            .expect("task should not panic")
            .expect("insert should succeed");
    }

    let history = tdb
        .db
        .get_history(1)
        .await
        .expect("history query should succeed");

    assert_eq!(
        history.len(),
        10,
        "all 10 concurrent inserts should be present in a clean DB"
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_occupancy_log_datetime_parsing() {
    let tdb = common::TestDatabase::new().await;

    tdb.db
        .insert_record(Utc::now(), 75.5)
        .await
        .expect("insert should succeed");

    let history = tdb
        .db
        .get_history(1)
        .await
        .expect("get_history should succeed");

    assert_eq!(history.len(), 1, "clean DB should contain exactly 1 record");
    let stored = history[0].timestamp;
    assert!(
        DateTime::parse_from_rfc3339(&stored.to_rfc3339()).is_ok(),
        "stored timestamp should survive an RFC3339 round-trip"
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_csv_export_with_mock_clock() {
    let tdb = common::TestDatabase::new().await;

    let now = Utc::now();
    for i in 0..3i64 {
        tdb.db
            .insert_record(now - Duration::hours(i), (i as f64) * 20.0)
            .await
            .expect("insert should succeed");
    }

    let fixed_time = Utc.with_ymd_and_hms(2024, 6, 15, 10, 30, 45).unwrap();
    let clock = MockClock::new(fixed_time);

    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let csv_path = tdb
        .db
        .export_to_csv(temp_dir.path(), &clock)
        .await
        .expect("CSV export should succeed");

    assert!(csv_path.exists(), "exported CSV file should exist on disk");

    let filename = csv_path
        .file_name()
        .expect("path should have a filename")
        .to_str()
        .expect("filename should be valid UTF-8");

    assert!(
        filename.contains("20240615_103045"),
        "filename should embed the mock clock timestamp; got: {filename}"
    );
    assert!(filename.starts_with("hardy_monitor_export_"));
    assert!(
        std::path::Path::new(filename)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("csv")),
        "filename should end with .csv (case-insensitive)"
    );

    let content = std::fs::read_to_string(&csv_path).expect("should be able to read the CSV");
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(
        lines.len(),
        4,
        "expected 1 header + 3 data rows; got {} lines",
        lines.len()
    );
    assert!(lines[0].contains("id"), "header should contain 'id'");
    assert!(
        lines[0].contains("timestamp"),
        "header should contain 'timestamp'"
    );
    assert!(
        lines[0].contains("percentage"),
        "header should contain 'percentage'"
    );

    tdb.cleanup().await;
}
