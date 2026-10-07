use thiserror::Error;

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

    #[error("ML training error: {0}")]
    MlTraining(String),

    #[error("Database schema {db} is newer than this build ({app}): update this program")]
    SchemaTooNew { db: i64, app: i64 },
}

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

    #[allow(clippy::needless_pass_by_value)]
    pub fn from_anyhow_db(err: anyhow::Error, context: &str) -> Self {
        AppError::Database(DatabaseError::QueryFailed {
            query_context: context.to_string(),
            message: err.to_string(),
        })
    }

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

    pub fn api_error(status_code: u16, message: impl Into<String>) -> Self {
        AppError::Api {
            status_code,
            message: message.into(),
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        AppError::Validation(message.into())
    }

    pub fn io(message: impl Into<String>) -> Self {
        AppError::Io(message.into())
    }

    pub fn category(&self) -> &'static str {
        match self {
            AppError::Network { .. } => "network",
            AppError::Database(_) => "database",
            AppError::Validation(_) => "validation",
            AppError::Io(_) => "io",
            AppError::Config(_) => "config",
            AppError::Api { .. } => "api",
            AppError::Unknown(_) => "unknown",
            AppError::MlTraining(_) => "ml_training",
            AppError::SchemaTooNew { .. } => "schema",
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_schema_too_new_display_category_and_not_retryable() {
        let err = AppError::SchemaTooNew { db: 7, app: 5 };
        assert_eq!(
            err.to_string(),
            "Database schema 7 is newer than this build (5): update this program"
        );
        assert_eq!(err.category(), "schema");
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
    fn test_validation_helper() -> Result<()> {
        let err = AppError::validation("bad input");
        let AppError::Validation(msg) = err else {
            anyhow::bail!("Wrong error type");
        };

        assert_eq!(msg, "bad input");

        Ok(())
    }
}
