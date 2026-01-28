//! Integration tests for API client.
//!
//! These tests use wiremock to simulate the gym API responses
//! and verify correct parsing and error handling.

use hardy_monitor::{api::GymApiClient, config::NetworkConfig};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

/// Test successful API response parsing.
#[tokio::test]
async fn test_fetch_occupancy_success() {
    let mock_server = MockServer::start().await;

    // Setup mock response matching the real API structure
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
        GymApiClient::new(mock_server.uri(), &config).expect("Client creation should succeed");

    let result = client.fetch_occupancy().await;
    assert!(result.is_ok(), "Fetch should succeed");

    let response = result.unwrap();
    assert_eq!(response.gym, 1);
    assert_eq!(response.name, "Test Gym");
    assert_eq!(response.workload, "45%");

    let percentage = response.occupancy_percentage().unwrap();
    assert_eq!(percentage, 45.5);
}

/// Test API response with integer percentage.
#[tokio::test]
async fn test_fetch_occupancy_integer_value() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    assert_eq!(response.occupancy_percentage().unwrap(), 100.0);
}

/// Test API response with zero occupancy.
#[tokio::test]
async fn test_fetch_occupancy_zero() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    assert_eq!(response.occupancy_percentage().unwrap(), 0.0);
}

/// Test handling of HTTP 500 errors.
#[tokio::test]
async fn test_fetch_occupancy_server_error() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 500 error");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("500"),
        "Error should mention status code"
    );
}

/// Test handling of HTTP 404 errors.
#[tokio::test]
async fn test_fetch_occupancy_not_found() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 404 error");
}

/// Test handling of malformed JSON response.
#[tokio::test]
async fn test_fetch_occupancy_invalid_json() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on invalid JSON");
}

/// Test handling of incomplete JSON response.
#[tokio::test]
async fn test_fetch_occupancy_missing_fields() {
    let mock_server = MockServer::start().await;

    // Missing 'numval' field
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on missing fields");
}

/// Test client timeout behavior.
#[tokio::test]
async fn test_fetch_occupancy_timeout() {
    let mock_server = MockServer::start().await;

    // Respond with a 2 second delay
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"gym":1,"name":"Test","workload":"0%","numval":"0"}"#)
                .set_delay(std::time::Duration::from_secs(2)),
        )
        .mount(&mock_server)
        .await;

    // Set timeout to 1 second
    let config = NetworkConfig {
        request_timeout_secs: 1,
        connect_timeout_secs: 1,
    };

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should timeout");
}

/// Test client can be cloned and used concurrently.
#[tokio::test]
async fn test_api_client_clone_and_concurrent_use() {
    let mock_server = MockServer::start().await;

    let body = r#"{"gym":1,"name":"Test","workload":"50%","numval":"50"}"#;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(3) // Expect 3 requests
        .mount(&mock_server)
        .await;

    let config = NetworkConfig {
        request_timeout_secs: 10,
        connect_timeout_secs: 5,
    };

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();

    // Clone and use concurrently
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
}

// ==================== Edge Case Tests ====================

/// Test API response with very large percentage value.
#[tokio::test]
async fn test_fetch_occupancy_very_large_percentage() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    // Should parse the value even if it's over 100%
    assert_eq!(response.occupancy_percentage().unwrap(), 9999.99);
}

/// Test API response with negative percentage value.
#[tokio::test]
async fn test_fetch_occupancy_negative_percentage() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    // Should parse negative value
    assert_eq!(response.occupancy_percentage().unwrap(), -10.5);
}

/// Test API response with Unicode characters in name.
#[tokio::test]
async fn test_fetch_occupancy_unicode_in_name() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    assert_eq!(response.name, "Fitnessclub München 🏋️");
    assert_eq!(response.occupancy_percentage().unwrap(), 50.0);
}

/// Test API response with decimal percentage as integer.
#[tokio::test]
async fn test_fetch_occupancy_decimal_as_integer() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    // numval is "45", so should parse as 45.0
    assert_eq!(response.occupancy_percentage().unwrap(), 45.0);
}

/// Test API response with extra fields (should be ignored).
#[tokio::test]
async fn test_fetch_occupancy_extra_fields() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    assert_eq!(response.occupancy_percentage().unwrap(), 50.0);
}

/// Test API response with whitespace in numval returns error.
#[tokio::test]
async fn test_fetch_occupancy_whitespace_numval_fails() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    // Whitespace in numval causes parse failure
    let result = response.occupancy_percentage();
    assert!(
        result.is_err(),
        "Whitespace in numval should cause parse error"
    );
}

/// Test API response with HTTP 429 Too Many Requests.
#[tokio::test]
async fn test_fetch_occupancy_rate_limited() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let result = client.fetch_occupancy().await;

    assert!(result.is_err(), "Should fail on 429 rate limit");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("429"),
        "Error should mention 429 status"
    );
}

/// Test API response with very small decimal.
#[tokio::test]
async fn test_fetch_occupancy_very_small_decimal() {
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

    let client = GymApiClient::new(mock_server.uri(), &config).unwrap();
    let response = client.fetch_occupancy().await.unwrap();

    assert!((response.occupancy_percentage().unwrap() - 0.001).abs() < 0.0001);
}
