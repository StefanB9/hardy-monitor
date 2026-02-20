//! Structured Error Types for Hardy Monitor
//!
//! This module provides type-safe error handling with proper error chains
//! and recovery information.

use thiserror::Error;

/// Primary application error type with structured variants
#[derive(Error, Debug, Clone)]
pub enum AppError {
    #[error("Network error: {message}")]
    Network {
        message: String,
        kind: NetworkErrorKind,
    },

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("API error: {status_code} - {message}")]
    Api { status_code: u16, message: String },

    #[error("Unexpected error: {0}")]
    Unknown(String),

    #[cfg(feature = "gui")]
    #[error("ML training error: {0}")]
    MlTraining(String),
}

/// Network-specific error kinds for better error handling
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NetworkErrorKind {
    #[error("Connection timeout")]
    Timeout,
    #[error("Connection refused")]
    ConnectionRefused,
    #[error("DNS resolution failed")]
    DnsFailure,
    #[error("TLS/SSL error")]
    TlsError,
    #[error("Unknown network error")]
    Unknown,
}

/// Database-specific error types
#[derive(Error, Debug, Clone)]
pub enum DatabaseError {
    #[error("Query failed ({query_context}): {message}")]
    QueryFailed {
        query_context: String,
        message: String,
    },

    #[error("Connection pool exhausted")]
    PoolExhausted,

    #[error("Record not found")]
    NotFound,

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
}

impl AppError {
    /// Returns true if this error is likely transient and retrying may succeed
    pub fn is_retryable(&self) -> bool {
        match self {
            AppError::Network { kind, .. } => matches!(
                kind,
                NetworkErrorKind::Timeout | NetworkErrorKind::ConnectionRefused
            ),
            AppError::Database(
                DatabaseError::PoolExhausted | DatabaseError::ConnectionFailed(_),
            ) => true,
            _ => false,
        }
    }

    /// Create a database error from sqlx error with context
    pub fn from_sqlx(err: &sqlx::Error, context: &str) -> Self {
        let db_error = match err {
            sqlx::Error::PoolTimedOut => DatabaseError::PoolExhausted,
            sqlx::Error::RowNotFound => DatabaseError::NotFound,
            sqlx::Error::Database(db_err) => {
                if let Some(constraint) = db_err.constraint() {
                    DatabaseError::ConstraintViolation(constraint.to_string())
                } else {
                    DatabaseError::QueryFailed {
                        query_context: context.to_string(),
                        message: db_err.message().to_string(),
                    }
                }
            }
            sqlx::Error::Io(_) | sqlx::Error::Tls(_) => {
                DatabaseError::ConnectionFailed(err.to_string())
            }
            _ => DatabaseError::QueryFailed {
                query_context: context.to_string(),
                message: err.to_string(),
            },
        };
        AppError::Database(db_error)
    }

    /// Create a database error from anyhow error with context
    #[allow(clippy::needless_pass_by_value)] 
    pub fn from_anyhow_db(err: anyhow::Error, context: &str) -> Self {
        AppError::Database(DatabaseError::QueryFailed {
            query_context: context.to_string(),
            message: err.to_string(),
        })
    }

    /// Create a network error from reqwest error
    #[allow(clippy::needless_pass_by_value)] 
    pub fn from_reqwest(err: reqwest::Error) -> Self {
        let kind = if err.is_timeout() {
            NetworkErrorKind::Timeout
        } else if err.is_connect() {
            NetworkErrorKind::ConnectionRefused
        } else {
            NetworkErrorKind::Unknown
        };

        AppError::Network {
            message: err.to_string(),
            kind,
        }
    }

    /// Create an API error with status code
    pub fn api_error(status_code: u16, message: impl Into<String>) -> Self {
        AppError::Api {
            status_code,
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        AppError::Validation(message.into())
    }

    /// Create an IO error
    pub fn io(message: impl Into<String>) -> Self {
        AppError::Io(message.into())
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            AppError::Network { .. } => "network",
            AppError::Database(_) => "database",
            AppError::Validation(_) => "validation",
            AppError::Io(_) => "io",
            AppError::Config(_) => "config",
            AppError::Api { .. } => "api",
            AppError::Unknown(_) => "unknown",
            #[cfg(feature = "gui")]
            AppError::MlTraining(_) => "ml_training",
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] 
#[allow(clippy::panic)] 
mod tests {
    use super::*;

    #[test]
    fn test_retryable_network_timeout() {
        let err = AppError::Network {
            message: "timed out".to_string(),
            kind: NetworkErrorKind::Timeout,
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_retryable_pool_exhausted() {
        let err = AppError::Database(DatabaseError::PoolExhausted);
        assert!(err.is_retryable());
    }

    #[test]
    fn test_not_retryable_validation() {
        let err = AppError::Validation("invalid date".to_string());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_error_category() {
        let err = AppError::Network {
            message: "test".to_string(),
            kind: NetworkErrorKind::Timeout,
        };
        assert_eq!(err.category(), "network");

        let err = AppError::Database(DatabaseError::NotFound);
        assert_eq!(err.category(), "database");
    }

    #[test]
    fn test_validation_helper() {
        let err = AppError::validation("bad input");
        match err {
            AppError::Validation(msg) => assert_eq!(msg, "bad input"),
            _ => panic!("Wrong error type"),
        }
    }
}
