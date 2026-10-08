//! Connection settings applied to every pooled connection.

mod common;

use anyhow::{Context, Result};
use common::RawTestDatabase;
use hardy_core::{config::DatabaseConfig, db::Database};

/// PostgreSQL's SQLSTATE for a statement cancelled by `statement_timeout`.
const QUERY_CANCELED: &str = "57014";

#[tokio::test]
async fn test_database_statement_timeout_cancels_slow_queries() -> Result<()> {
    let raw = RawTestDatabase::new().await?;
    let config = DatabaseConfig {
        statement_timeout_secs: 1,
        ..DatabaseConfig::with_url(raw.url.clone())
    };
    let pool = Database::pool_options(&config)
        .max_connections(1)
        .connect_with(Database::connect_options(&config)?)
        .await
        .context("connect with the configured options")?;

    let slow = sqlx::query!("SELECT pg_sleep(2)").fetch_one(&pool).await;
    let fast = sqlx::query!("SELECT pg_sleep(0)").fetch_one(&pool).await;
    pool.close().await;
    raw.cleanup().await;

    let code = slow
        .err()
        .context("a 2 s query must not outlive a 1 s timeout")?
        .as_database_error()
        .and_then(|e| e.code().map(std::borrow::Cow::into_owned));
    assert_eq!(code.as_deref(), Some(QUERY_CANCELED));
    fast.context("queries within the timeout still run")?;
    Ok(())
}
