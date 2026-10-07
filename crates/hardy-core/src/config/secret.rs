//! Secrets read from configuration.

use serde::Deserialize;

/// A secret read from configuration; never shown by `Debug`.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct SecretString(String);

impl SecretString {
    /// Wraps `value`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The secret value, for the one place that needs it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_string_exposes_value_but_debug_redacts() {
        let secret = SecretString::new("tk_123");
        assert_eq!(secret.expose(), "tk_123");
        assert_eq!(format!("{secret:?}"), "<redacted>");
    }
}
