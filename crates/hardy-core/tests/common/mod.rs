//! Shared test infrastructure.
//!
//! # Test Database Isolation
//!
//! [`TestDatabase`] creates a fresh, uniquely-named `PostgreSQL` database for
//! each test, runs all migrations, and drops the database on
//! [`TestDatabase::cleanup`].
//!
//! This prevents tests from reading or corrupting production data and ensures
//! every test starts from a known-empty state.
//!
//! ## Prerequisites
//!
//! - `DATABASE_URL` (or a `.env` file) must point to a `PostgreSQL` instance
//!   where the connecting user has `CREATEDB` privilege.
//! - `PostgreSQL` 13+ is required for `DROP DATABASE ... WITH (FORCE)`.
//!
//! ## Leftover databases
//!
//! If a test fails before [`TestDatabase::cleanup`] is called, the database
//! is left behind. Remove orphans with:
//!
//! ```sql
//! SELECT 'DROP DATABASE "' || datname || '";'
//! FROM   pg_database
//! WHERE  datname LIKE 'hardy_test_%';
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use hardy_core::db::Database;
use sqlx::{AssertSqlSafe, PgPool};

#[allow(dead_code)]
pub struct TestDatabase {
    db_name: String,
    admin_pool: PgPool,
    pub db: Database,
}

#[allow(dead_code)]
impl TestDatabase {
    pub async fn new() -> Result<Self> {
        let (db_name, admin_pool, test_url) = create_empty_database().await?;

        let db = Database::new(&test_url).await.with_context(|| {
            format!("failed to connect to test database '{db_name}' or run migrations")
        })?;

        Ok(Self {
            db_name,
            admin_pool,
            db,
        })
    }

    pub async fn cleanup(self) {
        self.db.close().await;
        drop_database(&self.admin_pool, &self.db_name).await;
    }
}

/// An isolated database with **no** migrations applied, for testing the
/// migrations themselves.
#[allow(dead_code)]
pub struct RawTestDatabase {
    db_name: String,
    admin_pool: PgPool,
    pub pool: PgPool,
    pub url: String,
}

#[allow(dead_code)]
impl RawTestDatabase {
    pub async fn new() -> Result<Self> {
        let (db_name, admin_pool, test_url) = create_empty_database().await?;
        let pool = PgPool::connect(&test_url)
            .await
            .with_context(|| format!("failed to connect to test database '{db_name}'"))?;
        Ok(Self {
            db_name,
            admin_pool,
            pool,
            url: test_url,
        })
    }

    pub async fn cleanup(self) {
        self.pool.close().await;
        drop_database(&self.admin_pool, &self.db_name).await;
    }
}

/// Creates a uniquely named empty database; returns its name, an admin pool
/// and its connection URL.
async fn create_empty_database() -> Result<(String, PgPool, String)> {
    dotenvy::dotenv().ok();

    let database_url = std::env::var("DATABASE_URL")
        .context("DATABASE_URL must be set to run database integration tests")?;

    let admin_url = replace_db_name(&database_url, "postgres")?;
    let db_name = unique_db_name()?;

    let admin_pool = PgPool::connect(&admin_url).await.context(
        "failed to connect to `PostgreSQL` admin database for test setup — ensure DATABASE_URL is \
         reachable and the user has CREATEDB privilege",
    )?;

    let mut conn = admin_pool
        .acquire()
        .await
        .context("failed to acquire admin connection for test setup")?;
    sqlx::raw_sql(AssertSqlSafe(format!(r#"CREATE DATABASE "{db_name}""#)))
        .execute(&mut *conn)
        .await
        .with_context(|| format!("failed to create test database '{db_name}'"))?;
    drop(conn);

    let test_url = replace_db_name(&database_url, &db_name)?;
    Ok((db_name, admin_pool, test_url))
}

async fn drop_database(admin_pool: &PgPool, db_name: &str) {
    let drop_result = match admin_pool.acquire().await {
        Ok(mut conn) => sqlx::raw_sql(AssertSqlSafe(format!(
            r#"DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)"#
        )))
        .execute(&mut *conn)
        .await
        .err(),
        Err(e) => Some(e),
    };
    if let Some(e) = drop_result {
        tracing::warn!(error = %e, %db_name, "failed to drop test database");
    }
}

fn replace_db_name(url: &str, new_db: &str) -> Result<String> {
    let (base, params) = url.split_once('?').unwrap_or((url, ""));

    // The URL carries the password, so it is not repeated in the error.
    let last_slash = base.rfind('/').context(
        "DATABASE_URL does not look like a valid `PostgreSQL` URL (expected \
         'postgres://host/dbname')",
    )?;

    let prefix = &base[..last_slash];

    Ok(if params.is_empty() {
        format!("{prefix}/{new_db}")
    } else {
        format!("{prefix}/{new_db}?{params}")
    })
}

fn unique_db_name() -> Result<String> {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the UNIX epoch")?
        .as_nanos();
    Ok(format!("hardy_test_{pid}_{nanos}"))
}
