//! Schema version of the database compared with this build's migrations.

use anyhow::{Context, Result};

use super::Database;

/// Migrations compiled into this build.
pub(super) static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("../../migrations");

/// Whether migrations run when connecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Migrations {
    /// Apply pending migrations (the daemon owns the schema).
    Apply,
    /// Only connect; check with [`Database::schema_status`].
    Verify,
}

/// The database schema relative to this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaStatus {
    Current,
    /// Migrations of this build are missing in the database (`db` is `None`
    /// when none were ever applied).
    DbOlder {
        db: Option<i64>,
        app: i64,
    },
    /// The database was migrated by a newer build.
    DbNewer {
        db: i64,
        app: i64,
    },
}

impl SchemaStatus {
    /// Compares the newest applied migration with the newest one built in.
    pub fn compare(db: Option<i64>, app: i64) -> Self {
        match db {
            Some(db) if db == app => SchemaStatus::Current,
            Some(db) if db > app => SchemaStatus::DbNewer { db, app },
            db => SchemaStatus::DbOlder { db, app },
        }
    }
}

/// Version of the newest migration compiled into this build.
pub fn app_schema_version() -> i64 {
    MIGRATOR.iter().map(|m| m.version).max().unwrap_or(0)
}

impl Database {
    /// Compares the database's applied migrations with this build's.
    #[tracing::instrument(skip_all)]
    pub async fn schema_status(&self) -> Result<SchemaStatus> {
        Ok(SchemaStatus::compare(
            applied_version(&self.pool).await?,
            app_schema_version(),
        ))
    }
}

/// Newest successfully applied migration; `None` on a database never
/// migrated.
pub(super) async fn applied_version(pool: &sqlx::PgPool) -> Result<Option<i64>> {
    let tracked = sqlx::query_scalar!(
        r#"SELECT to_regclass('public._sqlx_migrations') IS NOT NULL AS "tracked!""#
    )
    .fetch_one(pool)
    .await
    .context("failed to look up the migrations table")?;
    if !tracked {
        return Ok(None);
    }
    sqlx::query_scalar!("SELECT max(version) FROM _sqlx_migrations WHERE success")
        .fetch_one(pool)
        .await
        .context("failed to read the applied schema version")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_status_compare() {
        assert_eq!(SchemaStatus::compare(Some(5), 5), SchemaStatus::Current);
        assert_eq!(
            SchemaStatus::compare(Some(4), 5),
            SchemaStatus::DbOlder {
                db: Some(4),
                app: 5
            }
        );
        assert_eq!(
            SchemaStatus::compare(None, 5),
            SchemaStatus::DbOlder { db: None, app: 5 }
        );
        assert_eq!(
            SchemaStatus::compare(Some(6), 5),
            SchemaStatus::DbNewer { db: 6, app: 5 }
        );
    }

    #[test]
    fn test_app_schema_version_is_newest_migration() {
        let newest = MIGRATOR.iter().map(|m| m.version).max();
        assert_eq!(Some(app_schema_version()), newest);
        assert!(app_schema_version() > 0);
    }
}
