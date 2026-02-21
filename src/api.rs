use std::time::Duration;

use serde::Deserialize;

use crate::{
    config::NetworkConfig,
    error::{AppError, NetworkErrorKind},
};

#[derive(Debug, Deserialize)]
pub struct GymResponse {
    pub gym: i32,
    pub name: String,
    pub workload: String,
    #[serde(rename = "numval")]
    pub num_val: String,
}

impl GymResponse {
    pub fn occupancy_percentage(&self) -> Result<f64, AppError> {
        self.num_val.parse::<f64>().map_err(|e| {
            AppError::Validation(format!(
                "Failed to parse occupancy percentage from numval: {e}"
            ))
        })
    }
}

#[derive(Clone, Debug)]
pub struct GymApiClient {
    client: reqwest::Client,
    url: String,
}

impl GymApiClient {
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
    fn test_occupancy_percentage_valid_integer() {
        let response = make_response("75");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 75.0);
    }

    #[test]
    fn test_occupancy_percentage_valid_decimal() {
        let response = make_response("42.5");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42.5);
    }

    #[test]
    fn test_occupancy_percentage_zero() {
        let response = make_response("0");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_occupancy_percentage_hundred() {
        let response = make_response("100");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100.0);
    }

    #[test]
    fn test_occupancy_percentage_over_hundred() {
        let response = make_response("120.5");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 120.5);
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
    fn test_occupancy_percentage_negative() {
        let response = make_response("-5.0");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), -5.0);
    }

    #[test]
    fn test_occupancy_percentage_scientific_notation() {
        let response = make_response("1e2");
        let result = response.occupancy_percentage();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100.0);
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
