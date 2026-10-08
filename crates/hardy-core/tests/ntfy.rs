//! wiremock-based tests for the ntfy client.

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use hardy_core::{
    AppError, Notifier,
    config::NetworkConfig,
    ntfy::{NtfyClient, NtfyMessage, NtfyNotifier, PollSince},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string, header, header_exists, method, path, query_param},
};

fn client(server: &MockServer, token: Option<&str>) -> Result<NtfyClient> {
    NtfyClient::new(
        &server.uri(),
        token.map(str::to_string),
        &NetworkConfig::default(),
    )
    .context("ntfy client should build")
}

#[tokio::test]
async fn test_ntfy_publish_posts_title_and_body() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/alerts"))
        .and(header("Title", "Hardy"))
        .and(body_string("Quiet now"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    client(&server, None)?
        .publish("alerts", "Hardy", "Quiet now")
        .await
        .context("publish should succeed")?;
    Ok(())
}

#[tokio::test]
async fn test_ntfy_sends_bearer_token_when_configured() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/alerts"))
        .and(header("Authorization", "Bearer tk_secret"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    client(&server, Some("tk_secret"))?
        .publish("alerts", "t", "b")
        .await
        .context("authorized publish should succeed")?;
    Ok(())
}

#[tokio::test]
async fn test_ntfy_omits_auth_header_without_token() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header_exists("Authorization"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    client(&server, Some(""))?
        .publish("alerts", "t", "b")
        .await
        .context("empty token is treated as no token")?;
    Ok(())
}

#[tokio::test]
async fn test_ntfy_publish_error_status_is_api_error() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&server)
        .await;

    let result = client(&server, None)?.publish("alerts", "t", "b").await;
    assert!(matches!(
        result,
        Err(AppError::Api {
            status_code: 403,
            ..
        })
    ));
    Ok(())
}

#[tokio::test]
async fn test_ntfy_poll_returns_messages_only() -> Result<()> {
    let server = MockServer::start().await;
    let body = concat!(
        r#"{"id":"open1","time":1718611200,"event":"open","topic":"ctl"}"#,
        "\n",
        r#"{"id":"m1","time":1718611260,"event":"message","topic":"ctl","message":"on 25"}"#,
        "\n",
        r#"{"id":"k1","time":1718611270,"event":"keepalive","topic":"ctl"}"#,
        "\n",
        r#"{"id":"m2","time":1718611320,"event":"message","topic":"ctl","message":"off"}"#,
        "\n"
    );
    Mock::given(method("GET"))
        .and(path("/ctl/json"))
        .and(query_param("poll", "1"))
        .and(query_param("since", "m0"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let messages = client(&server, None)?
        .poll("ctl", &PollSince::Id("m0".to_string()))
        .await
        .context("poll should succeed")?;
    assert_eq!(
        messages,
        vec![
            NtfyMessage {
                id: "m1".to_string(),
                time: 1_718_611_260,
                message: "on 25".to_string()
            },
            NtfyMessage {
                id: "m2".to_string(),
                time: 1_718_611_320,
                message: "off".to_string()
            },
        ]
    );
    Ok(())
}

#[tokio::test]
async fn test_ntfy_poll_since_time_uses_unix_seconds() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/ctl/json"))
        .and(query_param("since", "1718611200"))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .expect(1)
        .mount(&server)
        .await;

    let since = PollSince::Time(
        Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0)
            .single()
            .context("valid timestamp")?,
    );
    let messages = client(&server, None)?.poll("ctl", &since).await?;
    assert_eq!(messages, Vec::new());
    Ok(())
}

#[tokio::test]
async fn test_ntfy_poll_rejects_malformed_json() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json\n"))
        .mount(&server)
        .await;

    let result = client(&server, None)?
        .poll("ctl", &PollSince::Id("x".to_string()))
        .await;
    assert!(matches!(result, Err(AppError::Validation(_))));
    Ok(())
}

#[tokio::test]
async fn test_ntfy_notifier_publishes_to_its_topic() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/phone"))
        .and(header("Title", "T"))
        .and(body_string("B"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let notifier = NtfyNotifier::new(client(&server, None)?, "phone".to_string());
    notifier
        .notify("T", "B")
        .await
        .context("notify should succeed")?;
    Ok(())
}
