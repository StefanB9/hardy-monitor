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
use hardy_core::{
    GymSchedule, MockClock,
    alert::{AlertDuration, AlertSettings, SettingsSource},
    db::{DataSource, NewModel},
};

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
        .expect("insert should succeed")
        .expect("empty slot should be filled");

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

    // Minute-aligned so stored slots equal the inserted instants.
    let now = utc(2024, 6, 15, 12, 0);
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
        let ts = now - Duration::minutes(i);
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
async fn test_insert_record_stores_minute_slot_as_measured() {
    let tdb = common::TestDatabase::new().await;

    tdb.db
        .insert_record(
            utc(2024, 6, 15, 10, 30) + Duration::milliseconds(42_700),
            50.0,
        )
        .await
        .expect("insert should succeed");

    let rows = tdb
        .db
        .get_history_range(utc(2024, 6, 15, 10, 0), utc(2024, 6, 15, 11, 0))
        .await
        .expect("query should succeed");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].timestamp, utc(2024, 6, 15, 10, 30));
    assert_eq!(rows[0].source, DataSource::Measured);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_insert_record_same_minute_keeps_first() {
    let tdb = common::TestDatabase::new().await;

    let first = tdb
        .db
        .insert_record(utc(2024, 6, 15, 10, 30), 40.0)
        .await
        .expect("insert should succeed");
    let second = tdb
        .db
        .insert_record(utc(2024, 6, 15, 10, 30) + Duration::seconds(20), 60.0)
        .await
        .expect("duplicate slot is not an error");

    assert!(first.is_some());
    assert_eq!(second, None, "occupied slot should be skipped");
    let rows = tdb
        .db
        .get_history_range(utc(2024, 6, 15, 10, 0), utc(2024, 6, 15, 11, 0))
        .await
        .expect("query should succeed");
    let values: Vec<f64> = rows.iter().map(|r| r.percentage).collect();
    assert_eq!(values, vec![40.0]);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_inserts_same_minute_store_one_row() {
    let tdb = common::TestDatabase::new().await;

    let slot = utc(2024, 6, 15, 10, 30);
    let mut handles = Vec::new();
    for i in 0..10i64 {
        let db_clone = tdb.db.clone();
        handles.push(tokio::spawn(async move {
            db_clone
                .insert_record(slot + Duration::seconds(i), i as f64)
                .await
        }));
    }
    let mut stored = 0;
    for handle in handles {
        if handle
            .await
            .expect("task should not panic")
            .expect("insert should not error")
            .is_some()
        {
            stored += 1;
        }
    }

    assert_eq!(stored, 1, "exactly one concurrent insert wins the slot");

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_insert_record_rejects_out_of_range_percentage() {
    let tdb = common::TestDatabase::new().await;

    for pct in [-0.1, 100.1, f64::NAN] {
        let result = tdb.db.insert_record(utc(2024, 6, 15, 10, 30), pct).await;
        assert!(result.is_err(), "{pct} should be rejected");
    }

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_batch_insert_skips_occupied_slots_and_sets_source() {
    let tdb = common::TestDatabase::new().await;

    tdb.db
        .insert_record(utc(2024, 6, 15, 10, 1), 30.0)
        .await
        .expect("insert should succeed");

    let inserted = tdb
        .db
        .batch_insert(
            &[
                (utc(2024, 6, 15, 10, 0), 10.0),
                (utc(2024, 6, 15, 10, 1), 99.0),
                (utc(2024, 6, 15, 10, 2), 20.0),
            ],
            DataSource::Interpolated,
        )
        .await
        .expect("batch insert should succeed");
    assert_eq!(inserted, 2);

    let rows = tdb
        .db
        .get_history_range(utc(2024, 6, 15, 10, 0), utc(2024, 6, 15, 11, 0))
        .await
        .expect("query should succeed");
    let summary: Vec<(f64, DataSource)> = rows.iter().map(|r| (r.percentage, r.source)).collect();
    assert_eq!(
        summary,
        vec![
            (10.0, DataSource::Interpolated),
            (30.0, DataSource::Measured),
            (20.0, DataSource::Interpolated),
        ]
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_insert_with_source_records_provenance() {
    let tdb = common::TestDatabase::new().await;

    tdb.db
        .insert_with_source(utc(2024, 6, 15, 6, 0), 0.0, DataSource::Boundary)
        .await
        .expect("insert should succeed")
        .expect("slot should be free");

    let rows = tdb
        .db
        .get_history_range(utc(2024, 6, 15, 0, 0), utc(2024, 6, 16, 0, 0))
        .await
        .expect("query should succeed");
    assert_eq!(rows[0].source, DataSource::Boundary);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_batch_update_percentage_relabels_only_measured_rows() {
    let tdb = common::TestDatabase::new().await;

    let measured = tdb
        .db
        .insert_record(utc(2024, 6, 15, 10, 0), 30.0)
        .await
        .expect("insert should succeed")
        .expect("slot should be free");
    let interpolated = tdb
        .db
        .insert_with_source(utc(2024, 6, 15, 10, 1), 35.0, DataSource::Interpolated)
        .await
        .expect("insert should succeed")
        .expect("slot should be free");

    let updated = tdb
        .db
        .batch_update_percentage(
            &[(measured, 31.0), (interpolated, 36.0)],
            DataSource::Smoothed,
        )
        .await
        .expect("update should succeed");
    assert_eq!(updated, 2);

    let rows = tdb
        .db
        .get_history_range(utc(2024, 6, 15, 10, 0), utc(2024, 6, 15, 11, 0))
        .await
        .expect("query should succeed");
    let summary: Vec<(f64, DataSource)> = rows.iter().map(|r| (r.percentage, r.source)).collect();
    assert_eq!(
        summary,
        vec![
            (31.0, DataSource::Smoothed),
            (36.0, DataSource::Interpolated)
        ]
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
    assert!(
        lines[0].contains("source"),
        "header should contain 'source'"
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_settings_default_row_is_disarmed() {
    let tdb = common::TestDatabase::new().await;

    let settings = tdb
        .db
        .get_alert_settings()
        .await
        .expect("default settings row exists");
    assert!(!settings.enabled());
    assert_eq!(settings.threshold_percent(), 30.0);
    assert_eq!(settings.active_until(), None);
    assert_eq!(settings.updated_by(), SettingsSource::Migration);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_settings_round_trip() {
    let tdb = common::TestDatabase::new().await;

    let now = utc(2024, 6, 17, 8, 0);
    let armed = AlertSettings::armed(
        25.0,
        AlertDuration::UntilClosing,
        now,
        &GymSchedule::default(),
        SettingsSource::Phone,
    )
    .unwrap();
    tdb.db
        .save_alert_settings(&armed)
        .await
        .expect("save should succeed");

    let loaded = tdb.db.get_alert_settings().await.expect("load");
    assert_eq!(loaded, armed);
    assert!(loaded.is_active(now));

    let off = loaded.disarmed(utc(2024, 6, 17, 9, 0), SettingsSource::Gui);
    tdb.db.save_alert_settings(&off).await.expect("save");
    assert_eq!(tdb.db.get_alert_settings().await.expect("load"), off);

    tdb.cleanup().await;
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
async fn test_ml_models_save_latest_and_load() {
    let tdb = common::TestDatabase::new().await;

    assert!(tdb.db.latest_model_info(2).await.unwrap().is_none());

    let older = tdb
        .db
        .save_model(&new_model(utc(2024, 6, 16, 22, 0), 2, b"old"), 3)
        .await
        .unwrap();
    let newer = tdb
        .db
        .save_model(&new_model(utc(2024, 6, 17, 22, 0), 2, b"new"), 3)
        .await
        .unwrap();
    // A model for another feature version is never returned for version 2.
    tdb.db
        .save_model(&new_model(utc(2024, 6, 18, 22, 0), 1, b"v1"), 3)
        .await
        .unwrap();

    let info = tdb.db.latest_model_info(2).await.unwrap().unwrap();
    assert_eq!(info.id, newer);
    assert_ne!(info.id, older);
    assert_eq!(info.training_samples, 1234);
    assert!(info.tuned);
    assert_eq!(info.algorithm, "Random Forest");
    assert_eq!(tdb.db.load_model(info.id).await.unwrap(), b"new".to_vec());

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_ml_models_keeps_only_newest_per_version() {
    let tdb = common::TestDatabase::new().await;

    for day in 10..15 {
        tdb.db
            .save_model(&new_model(utc(2024, 6, day, 22, 0), 2, b"m"), 3)
            .await
            .unwrap();
    }
    let ids = tdb.db.model_ids().await.unwrap();
    assert_eq!(ids.len(), 3);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_ml_state_retrain_request_lifecycle() {
    let tdb = common::TestDatabase::new().await;

    let state = tdb.db.get_ml_state().await.unwrap();
    assert_eq!(state.retrain_requested_at, None);
    assert_eq!(state.last_attempt_at, None);
    assert_eq!(state.last_error, None);

    tdb.db
        .request_retrain(utc(2024, 6, 17, 10, 0))
        .await
        .unwrap();
    assert_eq!(
        tdb.db.get_ml_state().await.unwrap().retrain_requested_at,
        Some(utc(2024, 6, 17, 10, 0))
    );

    // Starting an attempt consumes requests made before it.
    tdb.db
        .start_training_attempt(utc(2024, 6, 17, 10, 1))
        .await
        .unwrap();
    let started = tdb.db.get_ml_state().await.unwrap();
    assert_eq!(started.retrain_requested_at, None);
    assert_eq!(started.last_attempt_at, Some(utc(2024, 6, 17, 10, 1)));

    tdb.db
        .finish_training_attempt(Some("not enough data"))
        .await
        .unwrap();
    assert_eq!(
        tdb.db.get_ml_state().await.unwrap().last_error.as_deref(),
        Some("not enough data")
    );
    tdb.db.finish_training_attempt(None).await.unwrap();
    assert_eq!(tdb.db.get_ml_state().await.unwrap().last_error, None);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_ml_state_request_during_training_survives() {
    let tdb = common::TestDatabase::new().await;

    tdb.db
        .start_training_attempt(utc(2024, 6, 17, 10, 0))
        .await
        .unwrap();
    tdb.db
        .request_retrain(utc(2024, 6, 17, 10, 5))
        .await
        .unwrap();
    tdb.db.finish_training_attempt(None).await.unwrap();
    assert_eq!(
        tdb.db.get_ml_state().await.unwrap().retrain_requested_at,
        Some(utc(2024, 6, 17, 10, 5))
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_get_averages_range_ignores_repair_boundary_rows() {
    let tdb = common::TestDatabase::new().await;

    // Saturday 21:00 CEST closing: a measured 40% at 20:30, plus repair's
    // artificial 0% boundary entry at 21:00 and an interpolated value.
    tdb.db
        .insert_record(utc(2024, 6, 15, 18, 30), 40.0)
        .await
        .unwrap();
    tdb.db
        .insert_with_source(utc(2024, 6, 15, 19, 0), 0.0, DataSource::Boundary)
        .await
        .unwrap();
    tdb.db
        .insert_with_source(utc(2024, 6, 15, 18, 31), 20.0, DataSource::Interpolated)
        .await
        .unwrap();

    let buckets =
        bucket_averages(&tdb, utc(2024, 6, 15, 0, 0), utc(2024, 6, 16, 0, 0), Berlin).await;
    // Only the 20:00 slot remains (measured + interpolated); no 21:00 slot.
    assert_eq!(buckets, vec![(5, 20, 30.0)]);

    tdb.cleanup().await;
}
