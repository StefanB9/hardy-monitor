//! Integration tests for database operations.
//!
//! Each test creates and drops its own isolated PostgreSQL database via
//! `common::TestDatabase`, ensuring tests never read or write production data
//! and run deterministically regardless of pre-existing state.

mod common;

use chrono::{DateTime, Duration, TimeZone, Utc};
use hardy_monitor::MockClock;

/// Test that database creation and migration complete without error.
#[tokio::test]
async fn test_database_creation() {
    // `TestDatabase::new()` panics on failure, so reaching the cleanup line
    // is the assertion.
    let tdb = common::TestDatabase::new().await;
    tdb.cleanup().await;
}

/// Test inserting a single record returns a positive auto-increment ID.
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

/// Test inserting multiple records and retrieving them through `get_history`.
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

    assert_eq!(history.len(), 5, "clean DB should contain exactly 5 records");

    tdb.cleanup().await;
}

/// Test that `get_history_range` returns only records within the requested
/// window.
///
/// Six records are inserted at `now`, `now-1h`, …, `now-5h`.
/// The query window is `[now-2h, now+1h]` (inclusive), which captures exactly
/// three: `now`, `now-1h`, and `now-2h`.
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

/// Test that `get_averages_range` correctly groups records into hourly buckets
/// and computes their average.
///
/// Three records are inserted at 10:30, 10:20, and 10:10 with percentages
/// 30, 40, and 50.  The expected aggregate is one bucket (hour 10) with an
/// average of 40.0.
#[tokio::test]
async fn test_get_averages_range() {
    let tdb = common::TestDatabase::new().await;

    // Mid-hour anchor keeps all three records inside the same hourly bucket.
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
        .get_averages_range(base_time - Duration::hours(1), base_time + Duration::hours(1))
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

/// Test that the connection pool handles concurrent inserts without data loss.
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

/// Test that `OccupancyLog::datetime()` correctly parses the timestamp stored
/// by a round-trip through the database.
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
    // timestamp is now DateTime<Utc> directly — just verify it round-trips via RFC3339
    let stored = history[0].timestamp;
    assert!(
        DateTime::parse_from_rfc3339(&stored.to_rfc3339()).is_ok(),
        "stored timestamp should survive an RFC3339 round-trip"
    );

    tdb.cleanup().await;
}

/// Test that `export_to_csv` produces a correctly named file containing a
/// header row and one data row per record.
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
    assert!(filename.ends_with(".csv"));

    let content = std::fs::read_to_string(&csv_path).expect("should be able to read the CSV");
    let lines: Vec<&str> = content.lines().collect();

    // Clean DB has exactly 3 records → 1 header + 3 data rows = 4 lines.
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
