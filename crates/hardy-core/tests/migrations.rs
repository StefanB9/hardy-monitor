//! Integration tests for schema migrations that rewrite existing data.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]

mod common;

use std::borrow::Cow;

use chrono::{DateTime, TimeZone, Utc};
use sqlx::{PgPool, migrate::Migrator};

const DATA_INTEGRITY_VERSION: i64 = 20_261_006_164_725;

fn utc(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
}

/// Applies every migration older than `version`.
async fn migrate_before(pool: &PgPool, version: i64) {
    let mut migrator: Migrator = sqlx::migrate!("../../migrations");
    let older: Vec<_> = migrator
        .migrations
        .iter()
        .filter(|m| m.version < version)
        .cloned()
        .collect();
    migrator.migrations = Cow::Owned(older);
    migrator.run(pool).await.expect("older migrations apply");
}

async fn insert_raw(pool: &PgPool, ts: DateTime<Utc>, pct: f64) {
    sqlx::query("INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2)")
        .bind(ts)
        .bind(pct)
        .execute(pool)
        .await
        .expect("raw insert");
}

#[tokio::test]
async fn test_data_integrity_migration_quarantines_and_normalizes() {
    let raw = common::RawTestDatabase::new().await;
    migrate_before(&raw.pool, DATA_INTEGRITY_VERSION).await;

    // id 1: measured at 10:00:05 — kept, truncated to 10:00.
    insert_raw(&raw.pool, utc("2024-06-15T10:00:05Z"), 40.0).await;
    // id 2: repair row in the same minute — duplicate.
    insert_raw(&raw.pool, utc("2024-06-15T10:00:00Z"), 0.0).await;
    // id 3, 4: impossible values.
    insert_raw(&raw.pool, utc("2024-06-15T10:01:30Z"), 120.0).await;
    insert_raw(&raw.pool, utc("2024-06-15T10:02:00Z"), f64::NAN).await;
    // id 5: valid, sub-minute.
    insert_raw(&raw.pool, utc("2024-06-15T10:03:12.5Z"), 55.0).await;

    sqlx::migrate!("../../migrations")
        .run(&raw.pool)
        .await
        .expect("data_integrity migration applies to dirty data");

    let kept: Vec<(i64, DateTime<Utc>, f64, String)> =
        sqlx::query_as("SELECT id, timestamp, percentage, source FROM occupancy_logs ORDER BY id")
            .fetch_all(&raw.pool)
            .await
            .unwrap();
    assert_eq!(
        kept,
        vec![
            (
                1,
                Utc.with_ymd_and_hms(2024, 6, 15, 10, 0, 0).unwrap(),
                40.0,
                "measured".to_string()
            ),
            (
                5,
                Utc.with_ymd_and_hms(2024, 6, 15, 10, 3, 0).unwrap(),
                55.0,
                "measured".to_string()
            ),
        ]
    );

    let quarantined: Vec<(i64, String)> =
        sqlx::query_as("SELECT id, reason FROM occupancy_logs_quarantine ORDER BY id")
            .fetch_all(&raw.pool)
            .await
            .unwrap();
    assert_eq!(
        quarantined,
        vec![
            (2, "duplicate_minute".to_string()),
            (3, "out_of_range".to_string()),
            (4, "out_of_range".to_string()),
        ]
    );

    raw.cleanup().await;
}

#[tokio::test]
async fn test_data_integrity_constraints_reject_invalid_rows() {
    let raw = common::RawTestDatabase::new().await;
    sqlx::migrate!("../../migrations")
        .run(&raw.pool)
        .await
        .unwrap();

    insert_raw(&raw.pool, utc("2024-06-15T10:00:00Z"), 40.0).await;

    for (ts, pct, why) in [
        ("2024-06-15T10:00:00Z", 41.0, "duplicate minute"),
        ("2024-06-15T10:01:30Z", 41.0, "not minute-aligned"),
        ("2024-06-15T10:02:00Z", 100.5, "above 100"),
        ("2024-06-15T10:03:00Z", -0.5, "below 0"),
    ] {
        let result =
            sqlx::query("INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2)")
                .bind(utc(ts))
                .bind(pct)
                .execute(&raw.pool)
                .await;
        assert!(result.is_err(), "insert should be rejected: {why}");
    }

    let bad_source = sqlx::query(
        "INSERT INTO occupancy_logs (timestamp, percentage, source) VALUES ($1, 10, 'guessed')",
    )
    .bind(utc("2024-06-15T10:04:00Z"))
    .execute(&raw.pool)
    .await;
    assert!(bad_source.is_err(), "unknown source should be rejected");

    raw.cleanup().await;
}

#[tokio::test]
async fn test_data_integrity_migration_reverts_and_restores_quarantine() {
    let raw = common::RawTestDatabase::new().await;
    migrate_before(&raw.pool, DATA_INTEGRITY_VERSION).await;
    insert_raw(&raw.pool, utc("2024-06-15T10:00:05Z"), 40.0).await;
    insert_raw(&raw.pool, utc("2024-06-15T10:00:00Z"), 0.0).await;

    let migrator = sqlx::migrate!("../../migrations");
    migrator.run(&raw.pool).await.unwrap();
    migrator
        .undo(&raw.pool, DATA_INTEGRITY_VERSION - 1)
        .await
        .expect("down migration applies");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM occupancy_logs")
        .fetch_one(&raw.pool)
        .await
        .unwrap();
    assert_eq!(count, 2, "quarantined row is restored on revert");

    raw.cleanup().await;
}
