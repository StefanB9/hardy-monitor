//! Integration tests for the stored alert settings.
//!
//! Each test uses its own isolated database (`common::TestDatabase`).
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_precision_loss)]

mod common;

use chrono::{DateTime, TimeZone, Utc};
use hardy_core::{
    GymSchedule,
    alert::{AlertDuration, AlertSettings, SettingsSource},
};

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
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
