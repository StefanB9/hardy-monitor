//! Integration tests for hourly averages bucketed by gym-local time.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]

mod common;

use chrono::{DateTime, Duration, TimeZone, Utc};
use chrono_tz::{America::New_York, Europe::Berlin};
use hardy_core::db::DataSource;

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
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
