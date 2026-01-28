//! Integration tests for database operations.
//!
//! These tests require a running PostgreSQL database.
//! Set DATABASE_URL environment variable to run these tests.
//!
//! Example: DATABASE_URL=postgres://hardy:devpassword@localhost:5432/hardy_monitor_test

use chrono::{Duration, TimeZone, Utc};
use hardy_monitor::{MockClock, db::Database};

/// Get the database URL from environment, or skip the test.
fn get_database_url() -> Option<String> {
    // Load .env if present
    let _ = dotenvy::dotenv();
    std::env::var("DATABASE_URL").ok()
}

/// Helper macro to skip tests if DATABASE_URL is not set.
macro_rules! require_db {
    () => {
        match get_database_url() {
            Some(url) => url,
            None => {
                eprintln!("Skipping test: DATABASE_URL not set");
                return;
            }
        }
    };
}

/// Test database creation and migration.
#[tokio::test]
async fn test_database_creation() {
    let db_url = require_db!();
    let result = Database::new(&db_url).await;
    assert!(result.is_ok(), "Database creation should succeed: {:?}", result.err());
}

/// Test inserting a single record.
#[tokio::test]
async fn test_insert_record() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    let timestamp = Utc::now();
    let result = db.insert_record(timestamp, 50.0).await;

    assert!(result.is_ok());
    let id = result.unwrap();
    assert!(id > 0, "Insert should return a positive ID");
}

/// Test inserting multiple records and retrieving history.
#[tokio::test]
async fn test_insert_and_get_history() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    let now = Utc::now();

    // Insert 5 records
    for i in 0..5 {
        let timestamp = now - Duration::hours(i);
        db.insert_record(timestamp, (i as f64) * 10.0)
            .await
            .expect("Insert should succeed");
    }

    // Retrieve history for last 1 day
    let history = db.get_history(1).await.expect("Get history should succeed");

    // Note: In a shared test database, there might be more records
    assert!(history.len() >= 5, "Should retrieve at least 5 records");
}

/// Test retrieving history with date range.
#[tokio::test]
async fn test_get_history_range() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    let now = Utc::now();

    // Insert records over 3 hours
    for i in 0..6 {
        let timestamp = now - Duration::hours(i);
        db.insert_record(timestamp, 50.0)
            .await
            .expect("Insert should succeed");
    }

    // Query only the last 2 hours
    let start = now - Duration::hours(2);
    let end = now + Duration::hours(1); // Include current time

    let history = db
        .get_history_range(start, end)
        .await
        .expect("Range query should succeed");

    // Should get records from hours 0, 1, 2
    assert!(history.len() >= 2, "Should have at least 2 records in range");
}

/// Test aggregation of hourly averages.
#[tokio::test]
async fn test_get_averages_range() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    // Use a fixed timestamp to ensure all records fall in the same hour
    // Use middle of an hour (e.g., 10:30) so +/-20 minutes stays within the same hour
    let base_time = Utc.with_ymd_and_hms(2024, 6, 15, 10, 30, 0).unwrap();

    // Insert multiple records in the same hour: 10:10, 10:20, 10:30
    for i in 0..3 {
        let timestamp = base_time - Duration::minutes(i * 10);
        db.insert_record(timestamp, 30.0 + (i as f64) * 10.0) // 30, 40, 50
            .await
            .expect("Insert should succeed");
    }

    let start = base_time - Duration::hours(1);
    let end = base_time + Duration::hours(1);

    let averages = db
        .get_averages_range(start, end)
        .await
        .expect("Averages query should succeed");

    // Should have at least one hourly average (hour 10)
    assert!(!averages.is_empty(), "Should have at least one hour of data");
}

/// Test database handles concurrent writes.
#[tokio::test]
async fn test_concurrent_inserts() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    let now = Utc::now();

    // Spawn multiple concurrent inserts
    let mut handles = Vec::new();
    for i in 0..10 {
        let db_clone = db.clone();
        let ts = now - Duration::seconds(i);
        handles.push(tokio::spawn(async move {
            db_clone.insert_record(ts, i as f64).await
        }));
    }

    // Wait for all inserts
    for handle in handles {
        let result = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Insert should succeed");
    }

    // Verify records were inserted (may have more in shared db)
    let history = db.get_history(1).await.expect("Query should succeed");
    assert!(history.len() >= 10, "At least 10 records should be inserted");
}

/// Test OccupancyLog datetime parsing from database records.
#[tokio::test]
async fn test_occupancy_log_datetime_parsing() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    let now = Utc::now();
    db.insert_record(now, 75.5)
        .await
        .expect("Insert should succeed");

    let history = db.get_history(1).await.expect("Query should succeed");
    assert!(!history.is_empty(), "Should have at least one record");

    // Find the record we just inserted
    let log = history.iter().find(|l| (l.percentage - 75.5).abs() < 0.01);
    assert!(log.is_some(), "Should find our inserted record");

    let log = log.unwrap();

    // Verify datetime parsing works
    let parsed = log.datetime();
    assert!(parsed.is_some(), "datetime() should parse the timestamp");
}

/// Test CSV export functionality with MockClock.
#[tokio::test]
async fn test_csv_export_with_mock_clock() {
    let db_url = require_db!();
    let db = Database::new(&db_url).await.expect("DB creation failed");

    // Insert some test data
    let now = Utc::now();
    for i in 0..3 {
        let timestamp = now - Duration::hours(i);
        db.insert_record(timestamp, (i as f64) * 20.0)
            .await
            .expect("Insert should succeed");
    }

    // Create a mock clock with a fixed time for deterministic filename
    let fixed_time = Utc.with_ymd_and_hms(2024, 6, 15, 10, 30, 45).unwrap();
    let clock = MockClock::new(fixed_time);

    // Export to CSV using a temp directory
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let output_dir = temp_dir.path();
    let result = db.export_to_csv(output_dir, &clock).await;

    assert!(result.is_ok(), "CSV export should succeed");

    let csv_path = result.unwrap();
    assert!(csv_path.exists(), "CSV file should exist");

    // Verify filename format includes the mock time
    let filename = csv_path.file_name().unwrap().to_str().unwrap();
    assert!(
        filename.contains("20240615_103045"),
        "Filename should contain mock timestamp, got: {}",
        filename
    );
    assert!(filename.starts_with("hardy_monitor_export_"));
    assert!(filename.ends_with(".csv"));

    // Verify CSV content has data
    let content = std::fs::read_to_string(&csv_path).expect("Should read CSV");
    let lines: Vec<&str> = content.lines().collect();

    // Header + data rows
    assert!(lines.len() >= 2, "Should have header + at least 1 record");

    // Verify header contains expected columns
    let header = lines[0];
    assert!(header.contains("id"));
    assert!(header.contains("timestamp"));
    assert!(header.contains("percentage"));
}
