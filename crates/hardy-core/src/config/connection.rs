//! Database, gym API and network settings.

use serde::Deserialize;

use super::SecretString;

/// PostgreSQL connection and pool limits (`DATABASE_URL` comes from the
/// environment).
#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    /// Connection URL, including credentials.
    pub url: SecretString,
    /// Upper bound on pooled connections; keep small for hosted free tiers.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    /// How long to wait for a free pooled connection before failing.
    #[serde(default = "default_acquire_timeout_secs")]
    pub acquire_timeout_secs: u64,
}

impl DatabaseConfig {
    /// Config for `url` with default pool settings.
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            url: SecretString::new(url),
            max_connections: default_max_connections(),
            acquire_timeout_secs: default_acquire_timeout_secs(),
        }
    }
}

pub(super) fn default_max_connections() -> u32 {
    5
}

pub(super) fn default_acquire_timeout_secs() -> u64 {
    10
}

/// Where the gym's occupancy API is.
#[derive(Debug, Deserialize, Clone)]
pub struct GymConfig {
    pub api_url: String,
}

/// HTTP timeouts for the gym API and ntfy.
#[derive(Debug, Deserialize, Clone)]
pub struct NetworkConfig {
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: 30,
            connect_timeout_secs: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_network_config_defaults() {
        let config = NetworkConfig::default();
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.connect_timeout_secs, 10);
    }

    #[test]
    fn test_database_config_pool_defaults() -> Result<()> {
        let config: DatabaseConfig = toml::from_str(r#"url = "postgres://x/y""#)?;
        assert_eq!(config.max_connections, 5);
        assert_eq!(config.acquire_timeout_secs, 10);
        Ok(())
    }

    #[test]
    fn test_database_config_debug_hides_url() {
        let config = DatabaseConfig::with_url("postgres://hardy:s3cret@db.example/hardy");
        let shown = format!("{config:?}");
        assert!(!shown.contains("s3cret"), "{shown}");
        assert!(!shown.contains("db.example"), "{shown}");
    }
}
