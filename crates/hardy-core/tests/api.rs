//! Integration tests for API responses the client accepts.
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
async fn test_fetch_occupancy_success() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test Gym",
        "workload": "45%",
        "numval": "45.5"
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

    let client =
        GymApiClient::new(mock_server.uri(), &config).context("Client creation should succeed")?;

    let result = client.fetch_occupancy().await;
    assert!(result.is_ok(), "Fetch should succeed");

    let response = result?;
    assert_eq!(response.gym, 1);
    assert_eq!(response.name, "Test Gym");
    assert_eq!(response.workload, "45%");

    let percentage = response.occupancy_percentage()?;
    assert_eq!(percentage, 45.5);
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_integer_value() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 2,
        "name": "Another Gym",
        "workload": "100%",
        "numval": "100"
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

    assert_eq!(response.occupancy_percentage()?, 100.0);
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_zero() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Empty Gym",
        "workload": "0%",
        "numval": "0"
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

    assert_eq!(response.occupancy_percentage()?, 0.0);
    Ok(())
}

#[tokio::test]
async fn test_api_client_clone_and_concurrent_use() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{"gym":1,"name":"Test","workload":"50%","numval":"50"}"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(3)
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config)?;

    let client1 = client.clone();
    let client2 = client.clone();

    let (r1, r2, r3) = tokio::join!(
        client.fetch_occupancy(),
        client1.fetch_occupancy(),
        client2.fetch_occupancy()
    );

    assert!(r1.is_ok());
    assert!(r2.is_ok());
    assert!(r3.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_unicode_in_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Fitnessclub München 🏋️",
        "workload": "50%",
        "numval": "50"
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

    assert_eq!(response.name, "Fitnessclub München 🏋️");
    assert_eq!(response.occupancy_percentage()?, 50.0);
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_decimal_as_integer() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test",
        "workload": "45.5%",
        "numval": "45"
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

    assert_eq!(response.occupancy_percentage()?, 45.0);
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_extra_fields() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test",
        "workload": "50%",
        "numval": "50",
        "extra_field": "ignored",
        "another_one": 12345
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

    assert_eq!(response.occupancy_percentage()?, 50.0);
    Ok(())
}

#[tokio::test]
async fn test_fetch_occupancy_very_small_decimal() -> Result<()> {
    let mock_server = MockServer::start().await;

    let body = r#"{
        "gym": 1,
        "name": "Test",
        "workload": "0.001%",
        "numval": "0.001"
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

    assert!((response.occupancy_percentage()? - 0.001).abs() < 0.0001);
    Ok(())
}
