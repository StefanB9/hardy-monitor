//! Integration tests for `AlertService`: phone commands and alert delivery
//! against a real database and a mocked ntfy server.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]

mod common;

use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use hardy_core::{
    GymSchedule,
    alert::{AlertDuration, AlertService, AlertSettings, SettingsSource},
    config::{NetworkConfig, NotificationConfig},
    health::{HEALTH_TITLE, HealthEvent},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string_contains, header, method, path, query_param},
};

/// Monday 2024-06-17 at `h:mi` CEST, as UTC.
fn local(h: u32, mi: u32) -> DateTime<Utc> {
    chrono_tz::Europe::Berlin
        .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
        .unwrap()
        .with_timezone(&Utc)
}

fn config(server: &MockServer, control: bool) -> NotificationConfig {
    NotificationConfig {
        ntfy_topic: Some("alerts".to_string()),
        ntfy_server: server.uri(),
        control_topic: control.then(|| "ctl".to_string()),
        cooldown_secs: 0,
        opening_grace_minutes: 60,
        ..NotificationConfig::default()
    }
}

fn service(server: &MockServer, control: bool, now: DateTime<Utc>) -> AlertService {
    AlertService::new(
        &config(server, control),
        &NetworkConfig::default(),
        GymSchedule::default(),
        now,
    )
    .unwrap()
}

fn ndjson(messages: &[(&str, &str)]) -> String {
    messages
        .iter()
        .map(|(id, text)| {
            format!(r#"{{"id":"{id}","time":1718611200,"event":"message","topic":"ctl","message":"{text}"}}"#)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn mock_poll(server: &MockServer, since: &str, messages: &[(&str, &str)]) {
    Mock::given(method("GET"))
        .and(path("/ctl/json"))
        .and(query_param("since", since))
        .respond_with(ResponseTemplate::new(200).set_body_string(ndjson(messages)))
        .expect(1)
        .mount(server)
        .await;
}

#[tokio::test]
async fn test_alert_service_on_command_arms_and_replies() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;
    let start = local(10, 0);
    mock_poll(
        &server,
        &start.timestamp().to_string(),
        &[("m1", "on 25 2h")],
    )
    .await;
    Mock::given(method("POST"))
        .and(path("/alerts"))
        .and(body_string_contains("On below 25% until 12:01"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let mut svc = service(&server, true, start);
    let applied = svc
        .process_commands(&tdb.db, local(10, 1))
        .await
        .expect("commands processed");
    assert_eq!(applied, 1);

    let settings = tdb.db.get_alert_settings().await.unwrap();
    assert!(settings.is_active(local(12, 0)));
    assert!(!settings.is_active(local(12, 1)));
    assert_eq!(settings.threshold_percent(), 25.0);
    assert_eq!(settings.updated_by(), SettingsSource::Phone);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_off_and_status_commands() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;
    let start = local(10, 0);
    let armed = AlertSettings::armed(
        40.0,
        AlertDuration::Always,
        start,
        &GymSchedule::default(),
        SettingsSource::Gui,
    )
    .unwrap();
    tdb.db.save_alert_settings(&armed).await.unwrap();

    mock_poll(
        &server,
        &start.timestamp().to_string(),
        &[("m1", "status"), ("m2", "off")],
    )
    .await;
    Mock::given(method("POST"))
        .and(body_string_contains("On below 40% (no expiry)"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_string_contains("Off (threshold 40%)"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let mut svc = service(&server, true, start);
    assert_eq!(
        svc.process_commands(&tdb.db, local(10, 1)).await.unwrap(),
        2
    );
    assert!(!tdb.db.get_alert_settings().await.unwrap().enabled());

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_invalid_command_replies_usage_and_keeps_settings() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;
    let start = local(10, 0);
    mock_poll(
        &server,
        &start.timestamp().to_string(),
        &[("m1", "turn it on")],
    )
    .await;
    Mock::given(method("POST"))
        .and(body_string_contains("Commands:"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let before = tdb.db.get_alert_settings().await.unwrap();
    let mut svc = service(&server, true, start);
    assert_eq!(
        svc.process_commands(&tdb.db, local(10, 1)).await.unwrap(),
        0
    );
    assert_eq!(tdb.db.get_alert_settings().await.unwrap(), before);

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_polls_from_last_seen_message() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;
    let start = local(10, 0);
    mock_poll(&server, &start.timestamp().to_string(), &[("m1", "status")]).await;
    mock_poll(&server, "m1", &[]).await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let mut svc = service(&server, true, start);
    svc.process_commands(&tdb.db, local(10, 1)).await.unwrap();
    svc.process_commands(&tdb.db, local(10, 2)).await.unwrap();

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_without_control_topic_does_not_poll() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;

    let mut svc = service(&server, false, local(10, 0));
    assert_eq!(
        svc.process_commands(&tdb.db, local(10, 1)).await.unwrap(),
        0
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_publishes_alert_once_per_dip() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;
    let armed = AlertSettings::armed(
        30.0,
        AlertDuration::UntilClosing,
        local(9, 0),
        &GymSchedule::default(),
        SettingsSource::Gui,
    )
    .unwrap();
    tdb.db.save_alert_settings(&armed).await.unwrap();
    Mock::given(method("POST"))
        .and(path("/alerts"))
        .and(body_string_contains("Quiet now: 20% (below 30%)"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let mut svc = service(&server, false, local(9, 0));
    let now = local(10, 0);
    assert!(
        svc.process_reading(&tdb.db, 45.0, now)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        svc.process_reading(&tdb.db, 20.0, now + TimeDelta::minutes(1))
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        svc.process_reading(&tdb.db, 18.0, now + TimeDelta::minutes(2))
            .await
            .unwrap()
            .is_none()
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_no_alert_when_disarmed() {
    let tdb = common::TestDatabase::new().await;
    let server = MockServer::start().await;

    let mut svc = service(&server, false, local(9, 0));
    assert!(
        svc.process_reading(&tdb.db, 1.0, local(12, 0))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );

    tdb.cleanup().await;
}

#[tokio::test]
async fn test_alert_service_publishes_health_events_in_gym_time() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/alerts"))
        .and(header("Title", HEALTH_TITLE))
        .and(body_string_contains("No readings stored since 20:39"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let svc = service(&server, false, local(20, 0));
    svc.publish_health(&HealthEvent::Down {
        since: local(20, 39),
        reason: "insert failed".to_string(),
    })
    .await;
}
