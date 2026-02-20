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
//! If a test panics before [`TestDatabase::cleanup`] is called, the database
//! is left behind. Remove orphans with:
//!
//! ```sql
//! SELECT 'DROP DATABASE "' || datname || '";'
//! FROM   pg_database
//! WHERE  datname LIKE 'hardy_test_%';
//! ```

#![allow(clippy::panic)]

use std::time::{SystemTime, UNIX_EPOCH};

use hardy_monitor::db::Database;
use sqlx::{AssertSqlSafe, PgPool};

/// An isolated `PostgreSQL` database for a single test.
///
/// Created fresh (with all migrations applied) and dropped after use via
/// [`cleanup`](TestDatabase::cleanup).
///
/// # Usage
///
/// ```rust,ignore
/// #[tokio::test]
/// async fn my_test() {
///     let tdb = TestDatabase::new().await;
///     // use tdb.db ...
///     tdb.cleanup().await;
/// }
/// ```
pub struct TestDatabase {
    db_name: String,
    admin_pool: PgPool,
    /// Connected, migrated database handle ready for use in the test.
    pub db: Database,
}

impl TestDatabase {
    /// Create a fresh test database and run all migrations.
    ///
    /// Reads `DATABASE_URL` from the environment (or a `.env` file). Derives
    /// an admin connection URL by replacing the database name segment with
    /// `postgres`, then creates a uniquely-named database and runs all `SQLx`
    /// migrations before returning.
    pub async fn new() -> Self {
        dotenvy::dotenv().ok();

        let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
            panic!("DATABASE_URL must be set to run database integration tests")
        });

        let admin_url = replace_db_name(&database_url, "postgres");
        let db_name = unique_db_name();

        let admin_pool = PgPool::connect(&admin_url).await.unwrap_or_else(|e| {
            panic!(
                "failed to connect to `PostgreSQL` admin database for test setup — ensure \
                 DATABASE_URL is reachable and the user has CREATEDB privilege: {e}"
            )
        });

        let mut conn = admin_pool
            .acquire()
            .await
            .unwrap_or_else(|e| panic!("failed to acquire admin connection for test setup: {e}"));
        sqlx::raw_sql(AssertSqlSafe(format!(r#"CREATE DATABASE "{db_name}""#)))
            .execute(&mut *conn)
            .await
            .unwrap_or_else(|e| panic!("failed to create test database '{db_name}': {e}"));

        let test_url = replace_db_name(&database_url, &db_name);

        let db = Database::new(&test_url).await.unwrap_or_else(|e| {
            panic!("failed to connect to test database '{db_name}' or run migrations: {e}")
        });

        Self {
            db_name,
            admin_pool,
            db,
        }
    }

    /// Drop the test database.
    ///
    /// **Must be called at the end of every test.** If the test panics before
    /// this point, the database is left behind and must be cleaned up manually
    /// (see module-level docs).
    ///
    /// Requires `PostgreSQL` 13+ (`DROP DATABASE ... WITH (FORCE)`).
    #[allow(clippy::print_stderr)]
    pub async fn cleanup(self) {
        self.db.close().await;

        let drop_result = match self.admin_pool.acquire().await {
            Ok(mut conn) => sqlx::raw_sql(AssertSqlSafe(format!(
                r#"DROP DATABASE IF EXISTS "{}" WITH (FORCE)"#,
                self.db_name
            )))
            .execute(&mut *conn)
            .await
            .err(),
            Err(e) => Some(e),
        };
        if let Some(e) = drop_result {
            eprintln!(
                "warning: failed to drop test database '{}': {e}",
                self.db_name
            );
        }
    }
}

/// Replace the database name segment of a ``PostgreSQL`` connection URL.
///
/// Handles `postgres://` and `postgresql://` schemes, optional ports, and
/// preserved query parameters (e.g. `?sslmode=require`).
fn replace_db_name(url: &str, new_db: &str) -> String {
    let (base, params) = url.split_once('?').unwrap_or((url, ""));

    let last_slash = base.rfind('/').unwrap_or_else(|| {
        panic!(
            "DATABASE_URL does not look like a valid `PostgreSQL` URL (expected \
             'postgres://host/dbname', got '{url}')"
        )
    });

    let prefix = &base[..last_slash];

    if params.is_empty() {
        format!("{prefix}/{new_db}")
    } else {
        format!("{prefix}/{new_db}?{params}")
    }
}

/// Generate a database name that is unique to this process and call instant.
///
/// Combines the process ID with a nanosecond-precision timestamp. Within a
/// single nextest test binary (one binary per `tests/*.rs` file) all tests
/// share the same process, so the nanosecond counter advances between calls
/// and ensures uniqueness even across fully-parallel workers.
fn unique_db_name() -> String {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| panic!("system clock is before the UNIX epoch"))
        .as_nanos();
    format!("hardy_test_{pid}_{nanos}")
}
