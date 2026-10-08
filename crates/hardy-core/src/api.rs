use std::time::Duration;

use serde::Deserialize;

use crate::{
    config::NetworkConfig,
    error::{AppError, NetworkErrorKind},
};

/// Occupancy as reported by the gym API.
#[derive(Debug, Deserialize)]
pub struct GymResponse {
    pub gym: i32,
    pub name: String,
    pub workload: String,
    #[serde(rename = "numval")]
    pub num_val: String,
}

impl GymResponse {
    /// The reported occupancy in percent. Rejects values that cannot be a
    /// percentage (non-finite or outside `0..=100`) so they never reach the
    /// database.
    pub fn occupancy_percentage(&self) -> Result<f64, AppError> {
        let value = self.num_val.parse::<f64>().map_err(|e| {
            AppError::Validation(format!(
                "Failed to parse occupancy percentage from numval: {e}"
            ))
        })?;
        if (0.0..=100.0).contains(&value) {
            Ok(value)
        } else {
            Err(AppError::Validation(format!(
                "occupancy percentage out of range [0, 100]: {value}"
            )))
        }
    }
}

/// HTTP client for the gym's occupancy endpoint.
#[derive(Clone, Debug)]
pub struct GymApiClient {
    client: reqwest::Client,
    url: String,
}

impl GymApiClient {
    /// Builds a client for `url` with the configured timeouts.
    #[tracing::instrument(skip_all)]
    pub fn new(url: String, network_config: &NetworkConfig) -> Result<Self, AppError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(network_config.request_timeout_secs))
            .connect_timeout(Duration::from_secs(network_config.connect_timeout_secs))
            .build()
            .map_err(|e| AppError::Network {
                message: format!("Failed to create HTTP client: {e}"),
                kind: NetworkErrorKind::Unknown,
            })?;

        Ok(Self { client, url })
    }

    /// Fetches and validates the current occupancy.
    #[tracing::instrument(skip_all, fields(url = %self.url, http.status_code = tracing::field::Empty))]
    pub async fn fetch_occupancy(&self) -> Result<GymResponse, AppError> {
        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(AppError::from_reqwest)?;

        let status = response.status();
        tracing::Span::current().record("http.status_code", status.as_u16());
        if !status.is_success() {
            return Err(AppError::api_error(
                status.as_u16(),
                format!("API returned error status: {status}"),
            ));
        }

        let data = response
            .json::<GymResponse>()
            .await
            .map_err(AppError::from_reqwest)?;

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    fn make_response(num_val: &str) -> GymResponse {
        GymResponse {
            gym: 1,
            name: "Test Gym".to_string(),
            workload: "50%".to_string(),
            num_val: num_val.to_string(),
        }
    }

    #[test]
    fn test_occupancy_percentage_valid_integer() -> Result<()> {
        let response = make_response("75");
        let val = response.occupancy_percentage()?;
        assert_relative_eq!(val, 75.0);
        Ok(())
    }

    #[test]
    fn test_occupancy_percentage_valid_decimal() -> Result<()> {
        let response = make_response("42.5");
        let val = response.occupancy_percentage()?;
        assert_relative_eq!(val, 42.5);
        Ok(())
    }

    #[test]
    fn test_occupancy_percentage_zero() -> Result<()> {
        let response = make_response("0");
        let val = response.occupancy_percentage()?;
        assert_relative_eq!(val, 0.0);
        Ok(())
    }

    #[test]
    fn test_occupancy_percentage_hundred() -> Result<()> {
        let response = make_response("100");
        let val = response.occupancy_percentage()?;
        assert_relative_eq!(val, 100.0);
        Ok(())
    }

    #[test]
    fn test_occupancy_percentage_over_hundred_rejected() {
        let response = make_response("120.5");
        assert!(response.occupancy_percentage().is_err());
    }

    #[test]
    fn test_occupancy_percentage_non_finite_rejected() {
        for num_val in ["NaN", "inf", "-inf"] {
            let response = make_response(num_val);
            assert!(
                response.occupancy_percentage().is_err(),
                "{num_val} must be rejected"
            );
        }
    }

    #[test]
    fn test_occupancy_percentage_invalid_string() {
        let response = make_response("not-a-number");
        let result = response.occupancy_percentage();
        assert!(result.is_err());
    }

    #[test]
    fn test_occupancy_percentage_empty_string() {
        let response = make_response("");
        let result = response.occupancy_percentage();
        assert!(result.is_err());
    }

    #[test]
    fn test_occupancy_percentage_whitespace() {
        let response = make_response("  ");
        let result = response.occupancy_percentage();
        assert!(result.is_err());
    }

    #[test]
    fn test_occupancy_percentage_negative_rejected() {
        let response = make_response("-5.0");
        assert!(response.occupancy_percentage().is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn occupancy_percentage_accepts_exactly_the_valid_range(value in -1000.0f64..1000.0) {
            let result = make_response(&value.to_string()).occupancy_percentage();
            prop_assert_eq!(result.is_ok(), (0.0..=100.0).contains(&value));
        }
    }

    #[test]
    fn test_occupancy_percentage_scientific_notation() -> Result<()> {
        let response = make_response("1e2");
        let val = response.occupancy_percentage()?;
        assert_relative_eq!(val, 100.0);
        Ok(())
    }

    #[test]
    fn test_api_client_creation() {
        let config = NetworkConfig {
            request_timeout_secs: 30,
            connect_timeout_secs: 10,
        };
        let result = GymApiClient::new("https://example.com/api".to_string(), &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_api_client_with_custom_timeouts() {
        let config = NetworkConfig {
            request_timeout_secs: 60,
            connect_timeout_secs: 20,
        };
        let result = GymApiClient::new("https://test.example.com".to_string(), &config);
        assert!(result.is_ok());
    }
}
