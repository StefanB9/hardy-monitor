//! Schema version checks: who migrates and what happens on a mismatch.

mod common;

use anyhow::{Context, Result, bail};
use hardy_core::{
    AppError,
    config::DatabaseConfig,
    db::{Database, Migrations, SchemaStatus, app_schema_version},
};

const FUTURE_VERSION: i64 = 99_991_231_000_000;

async fn table_exists(pool: &sqlx::PgPool, name: &str) -> Result<bool> {
    let exists: Option<String> = sqlx::query_scalar("SELECT to_regclass($1)::text")
        .bind(name)
        .fetch_one(pool)
        .await
        .context("to_regclass")?;
    Ok(exists.is_some())
}

/// Pretends a newer build migrated the database.
async fn add_future_migration(pool: &sqlx::PgPool) -> Result<()> {
    sqlx::query(
        "INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time) \
         VALUES ($1, 'from the future', TRUE, '\\x00'::bytea, 0)",
    )
    .bind(FUTURE_VERSION)
    .execute(pool)
    .await
    .context("insert future migration")?;
    Ok(())
}

#[tokio::test]
async fn test_schema_verify_does_not_migrate() -> Result<()> {
    let raw = common::RawTestDatabase::new().await;
    let db = Database::connect(&DatabaseConfig::with_url(&raw.url), Migrations::Verify).await?;

    let status = db.schema_status().await?;
    let migrated = table_exists(&raw.pool, "occupancy_logs").await?;
    db.close().await;
    raw.cleanup().await;

    assert_eq!(
        status,
        SchemaStatus::DbOlder {
            db: None,
            app: app_schema_version()
        }
    );
    assert!(!migrated);
    Ok(())
}

#[tokio::test]
async fn test_schema_apply_migrates_to_current() -> Result<()> {
    let raw = common::RawTestDatabase::new().await;
    let db = Database::connect(&DatabaseConfig::with_url(&raw.url), Migrations::Apply).await?;

    let status = db.schema_status().await?;
    db.close().await;
    raw.cleanup().await;

    assert_eq!(status, SchemaStatus::Current);
    Ok(())
}

#[tokio::test]
async fn test_schema_newer_database_is_reported() -> Result<()> {
    let raw = common::RawTestDatabase::new().await;
    let config = DatabaseConfig::with_url(&raw.url);
    Database::connect(&config, Migrations::Apply)
        .await?
        .close()
        .await;
    add_future_migration(&raw.pool).await?;

    let verified = Database::connect(&config, Migrations::Verify).await?;
    let status = verified.schema_status().await?;
    verified.close().await;
    let applied = Database::connect(&config, Migrations::Apply).await;
    raw.cleanup().await;

    assert_eq!(
        status,
        SchemaStatus::DbNewer {
            db: FUTURE_VERSION,
            app: app_schema_version()
        }
    );
    match applied {
        Ok(_) => bail!("an older build must not run against a newer schema"),
        Err(e) => match e.downcast_ref::<AppError>() {
            Some(AppError::SchemaTooNew { db, app }) => {
                assert_eq!((*db, *app), (FUTURE_VERSION, app_schema_version()));
            }
            _ => bail!("expected SchemaTooNew, got: {e:#}"),
        },
    }
    Ok(())
}
