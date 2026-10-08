//! Integration tests for API responses and failures the client rejects.
//!
//! These tests use wiremock to simulate the gym API.
#![allow(clippy::float_cmp)]

use anyhow::{Context, Result};
use hardy_core::{api::GymApiClient, config::NetworkConfig};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn test_fetch_occupancy_server_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 500 error");
    let err = result.err().context("expected an error")?;
    assert!(
        err.to_string().contains("500"),
        "Error should mention status code"
    );
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_not_found() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 404 error");
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_invalid_json() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on invalid JSON");
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_missing_fields() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test"
    }"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on missing fields");
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_timeout() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"gym":1,"name":"Test","workload":"0%","numval":"0"}"#)
                .set_delay(std::time::Duration::from_secs(2)),
        )
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 1,
        connect_timeout_secs: 1,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should timeout");
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_very_large_percentage_rejected() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test Gym",
        "workload": "9999%",
        "numval": "9999.99"
    }"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let response = client.fetch_occupancy().await?;

    assert!(
        response.occupancy_percentage().is_err(),
        "percentages above 100 must be rejected"
    );
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_negative_percentage_rejected() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test Gym",
        "workload": "-10%",
        "numval": "-10.5"
    }"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let response = client.fetch_occupancy().await?;

    assert!(
        response.occupancy_percentage().is_err(),
        "negative percentages must be rejected"
    );
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_whitespace_numval_fails() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test",
        "workload": "50%",
        "numval": "  50.5  "
    }"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let response = client.fetch_occupancy().await?;

    let result = response.occupancy_percentage();
    assert!(
        result.is_err(),
        "Whitespace in numval should cause parse error"
    );
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_rate_limited() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 429 rate limit");
    let err = result.err().context("expected an error")?;
    assert!(
        err.to_string().contains("429"),
        "Error should mention 429 status"
    );
    Ok(())
}
