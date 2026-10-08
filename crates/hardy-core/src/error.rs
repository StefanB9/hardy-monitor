//! Application error types.

use thiserror::Error;

/// Recoverable, matchable application errors.
#[derive(Error, Debug, Clone)]
pub enum AppError {
    /// The gym API could not be reached.
    #[error("Network error: {message}")]
    Network {
        message: String,
        kind: NetworkErrorKind,
    },

    /// A database operation failed.
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    /// Input or data failed validation.
    #[error("Validation error: {0}")]
    Validation(String),

    /// The configuration is invalid.
    #[error("Configuration error: {0}")]
    Config(String),

    /// The gym API answered with an error status.
    #[error("API error: {status_code} - {message}")]
    Api { status_code: u16, message: String },

    /// Anything not covered by the other variants.
    #[error("Unexpected error: {0}")]
    Unknown(String),

    /// Model training failed.
    #[error("ML training error: {0}")]
    MlTraining(String),

    /// The database was migrated by a newer build than this one.
    #[error("Database schema {db} is newer than this build ({app}): update this program")]
    SchemaTooNew { db: i64, app: i64 },
}

/// Why a network request failed.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NetworkErrorKind {
    /// The request timed out.
    #[error("Connection timeout")]
    Timeout,
    /// The connection could not be established.
    #[error("Connection refused")]
    ConnectionRefused,
    /// Any other request failure.
    #[error("Unknown network error")]
    Unknown,
}

/// Why a database operation failed.
#[derive(Error, Debug, Clone)]
pub enum DatabaseError {
    /// A query failed for a non-transient reason.
    #[error("Query failed ({query_context}): {message}")]
    QueryFailed {
        query_context: String,
        message: String,
    },

    /// No pooled connection became free in time.
    #[error("Connection pool exhausted")]
    PoolExhausted,

    /// A query expected a row and found none.
    #[error("Record not found")]
    NotFound,

    /// A row violated the named constraint.
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// The connection to the database failed.
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
}

impl AppError {
    /// Whether retrying the operation may succeed (timeouts, refused
    /// connections, pool exhaustion).
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

    /// Classifies an `sqlx` error from the operation named by `context`.
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

    /// Classifies a context-wrapped database error. Searches the error chain
    /// for the underlying `sqlx::Error` so transient failures (I/O, pool
    /// timeouts) stay retryable; anything else becomes a non-retryable
    /// `QueryFailed`.
    pub fn from_anyhow_sqlx(err: &anyhow::Error, context: &str) -> Self {
        if let Some(app_err @ AppError::SchemaTooNew { .. }) = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<AppError>())
        {
            return app_err.clone();
        }
        err.chain()
            .find_map(|cause| cause.downcast_ref::<sqlx::Error>())
            .map_or_else(
                || {
                    AppError::Database(DatabaseError::QueryFailed {
                        query_context: context.to_string(),
                        message: format!("{err:#}"),
                    })
                },
                |sqlx_err| Self::from_sqlx(sqlx_err, context),
            )
    }

    /// Wraps any database-layer failure as a non-retryable `QueryFailed`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_anyhow_db(err: anyhow::Error, context: &str) -> Self {
        AppError::Database(DatabaseError::QueryFailed {
            query_context: context.to_string(),
            message: err.to_string(),
        })
    }

    /// Classifies a `reqwest` error by timeout or connect failure.
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

    /// An [`AppError::Api`] for the given HTTP status.
    pub fn api_error(status_code: u16, message: impl Into<String>) -> Self {
        AppError::Api {
            status_code,
            message: message.into(),
        }
    }

    /// An [`AppError::Validation`] with the given message.
    pub fn validation(message: impl Into<String>) -> Self {
        AppError::Validation(message.into())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_schema_too_new_display_and_not_retryable() {
        let err = AppError::SchemaTooNew { db: 7, app: 5 };
        assert_eq!(
            err.to_string(),
            "Database schema 7 is newer than this build (5): update this program"
        );
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_from_anyhow_sqlx_keeps_schema_too_new() {
        let err = anyhow::Error::new(AppError::SchemaTooNew { db: 7, app: 5 }).context("connect");
        assert!(matches!(
            AppError::from_anyhow_sqlx(&err, "connect_database"),
            AppError::SchemaTooNew { db: 7, app: 5 }
        ));
    }

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
    fn test_from_anyhow_sqlx_keeps_pool_timeout_retryable() {
        let err = anyhow::Error::new(sqlx::Error::PoolTimedOut).context("Failed to insert");
        let app = AppError::from_anyhow_sqlx(&err, "insert");
        assert!(matches!(
            app,
            AppError::Database(DatabaseError::PoolExhausted)
        ));
        assert!(app.is_retryable());
    }

    #[test]
    fn test_from_anyhow_sqlx_io_error_is_retryable() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let err = anyhow::Error::new(sqlx::Error::Io(io)).context("Failed to connect");
        assert!(AppError::from_anyhow_sqlx(&err, "connect").is_retryable());
    }

    #[test]
    fn test_from_anyhow_sqlx_non_sqlx_error_is_not_retryable() {
        let err = anyhow::anyhow!("something else");
        let app = AppError::from_anyhow_sqlx(&err, "insert");
        assert!(matches!(
            app,
            AppError::Database(DatabaseError::QueryFailed { .. })
        ));
        assert!(!app.is_retryable());
    }

    #[test]
    fn test_validation_helper() -> Result<()> {
        let err = AppError::validation("bad input");
        let AppError::Validation(msg) = err else {
            anyhow::bail!("Wrong error type");
        };

        assert_eq!(msg, "bad input");

        Ok(())
    }
}
