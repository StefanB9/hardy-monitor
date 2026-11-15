//! Every model's live forecast, logged each full hour while the gym is
//! open, and how accurate those forecasts turned out.

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, NaiveDate, Utc};
use chrono_tz::Tz;

use super::Database;
use crate::accuracy::ErrorSums;

/// One model's logged forecast: element `i` is for 15 min × (i + 1) after
/// the time it was made; `None` where the gym is closed (and in `low` /
/// `high` where the model had no interval yet).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelForecast {
    /// Model key; `selected` is the default forecast.
    pub model: String,
    pub forecast: Vec<Option<f64>>,
    pub low: Vec<Option<f64>>,
    pub high: Vec<Option<f64>>,
}

/// Error sums of the logged forecasts per model, by step (1 = 15 min
/// ahead) and by gym-local target day.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelErrors {
    pub by_step: Vec<(String, u32, ErrorSums)>,
    pub by_day: Vec<(String, NaiveDate, ErrorSums)>,
}

/// One model's logged forecast next to what happened: element `i` of
/// both is for 15 min × (i + 1) after `made_at`; `actual` is the measured
/// or smoothed reading, `None` where there is none.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedOutcome {
    pub made_at: DateTime<Utc>,
    pub forecast: Vec<Option<f64>>,
    pub actual: Vec<Option<f64>>,
}

/// One grouped row of [`Database::model_forecast_errors`]: a model and
/// either a step or a day, with the error sums.
struct SumsRow {
    model: String,
    step: Option<i32>,
    day: Option<NaiveDate>,
    count: i64,
    abs: f64,
    squared: f64,
    signed: f64,
    actual: f64,
    over: i64,
    within_5: i64,
    within_10: i64,
    baseline_count: i64,
    abs_on_baseline: f64,
    baseline_abs: f64,
    last_week_count: i64,
    abs_on_last_week: f64,
    last_week_abs: f64,
    banded: i64,
    covered: i64,
    width: f64,
    interval_score: f64,
}

impl SumsRow {
    fn split(self) -> (String, Option<i32>, Option<NaiveDate>, ErrorSums) {
        let sums = ErrorSums {
            count: self.count,
            abs: self.abs,
            squared: self.squared,
            signed: self.signed,
            actual: self.actual,
            over: self.over,
            within_5: self.within_5,
            within_10: self.within_10,
            baseline_count: self.baseline_count,
            abs_on_baseline: self.abs_on_baseline,
            baseline_abs: self.baseline_abs,
            last_week_count: self.last_week_count,
            abs_on_last_week: self.abs_on_last_week,
            last_week_abs: self.last_week_abs,
            banded: self.banded,
            covered: self.covered,
            width: self.width,
            interval_score: self.interval_score,
        };
        (self.model, self.step, self.day, sums)
    }
}

/// The values stored as `real`: forecasts need no more precision.
fn to_real(values: &[Option<f64>]) -> impl Iterator<Item = Option<f32>> + '_ {
    #[allow(clippy::cast_possible_truncation)]
    values.iter().map(|v| v.map(|v| v as f32))
}

impl Database {
    /// Logs the forecasts made at `made_at`; ones already logged for the
    /// same time and model are kept. Every array must have the same length.
    /// Returns how many were added.
    #[tracing::instrument(skip(self, rows), fields(db.operation = "save_model_forecasts", count = rows.len()))]
    pub async fn save_model_forecasts(
        &self,
        made_at: DateTime<Utc>,
        rows: &[ModelForecast],
    ) -> Result<u64> {
        let Some(first) = rows.first() else {
            return Ok(0);
        };
        let steps = first.forecast.len();
        ensure!(
            rows.iter().all(|r| r.forecast.len() == steps
                && r.low.len() == steps
                && r.high.len() == steps),
            "logged forecasts must all have {steps} steps"
        );
        let width = i32::try_from(steps).context("too many steps")?;
        let total = steps * rows.len();
        let (mut forecast, mut low, mut high) = (
            Vec::with_capacity(total),
            Vec::with_capacity(total),
            Vec::with_capacity(total),
        );
        for row in rows {
            forecast.extend(to_real(&row.forecast));
            low.extend(to_real(&row.low));
            high.extend(to_real(&row.high));
        }
        // Arrays of arrays cannot be unnested row by row, so each row's
        // arrays are sliced out of one flat array per column.
        let result = sqlx::query!(
            r#"
            INSERT INTO model_forecast_log (made_at, model, forecast, low, high)
            SELECT $1, m.model,
                   ($3::real[])[(m.i::int4 - 1) * $6 + 1 : m.i::int4 * $6],
                   ($4::real[])[(m.i::int4 - 1) * $6 + 1 : m.i::int4 * $6],
                   ($5::real[])[(m.i::int4 - 1) * $6 + 1 : m.i::int4 * $6]
            FROM UNNEST($2::text[]) WITH ORDINALITY AS m(model, i)
            ON CONFLICT (made_at, model) DO NOTHING
            "#,
            made_at,
            &rows.iter().map(|r| r.model.clone()).collect::<Vec<_>>(),
            &forecast as &[Option<f32>],
            &low as &[Option<f32>],
            &high as &[Option<f32>],
            width
        )
        .execute(&self.pool)
        .await
        .context("failed to log the model forecasts")?;
        Ok(result.rows_affected())
    }

    /// Error sums of the logged forecasts for targets in `[since, until)`,
    /// scored against measured or smoothed readings, per model by step and
    /// by gym-local day.
    #[tracing::instrument(skip(self), fields(db.operation = "model_forecast_errors"))]
    pub async fn model_forecast_errors(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
        tz: Tz,
    ) -> Result<ModelErrors> {
        // References are looked up once per target, not per model.
        let rows = sqlx::query_as!(
            SumsRow,
            r#"
            WITH f AS (
                SELECT l.model, s.step::int4 AS step,
                       l.made_at + s.step * INTERVAL '15 minutes' AS target,
                       s.value::float8 AS forecast, s.low::float8 AS low, s.high::float8 AS high
                FROM model_forecast_log l
                CROSS JOIN LATERAL UNNEST(l.forecast, l.low, l.high)
                    WITH ORDINALITY AS s(value, low, high, step)
                WHERE l.made_at >= $1::timestamptz - INTERVAL '1 day' AND l.made_at < $2
                  AND s.value IS NOT NULL
            ),
            refs AS (
                SELECT t.target, o.percentage AS actual,
                       (SELECT AVG(p.percentage) FROM occupancy_logs p
                        WHERE p.source IN ('measured', 'smoothed')
                          AND p.timestamp IN (
                              SELECT ((t.target AT TIME ZONE $3) - w * INTERVAL '7 days')
                                     AT TIME ZONE $3
                              FROM generate_series(1, 4) AS w)) AS baseline,
                       (SELECT p.percentage FROM occupancy_logs p
                        WHERE p.source IN ('measured', 'smoothed')
                          AND p.timestamp = ((t.target AT TIME ZONE $3) - INTERVAL '7 days')
                                            AT TIME ZONE $3) AS last_week
                FROM (SELECT DISTINCT target FROM f WHERE target >= $1 AND target < $2) t
                JOIN occupancy_logs o
                  ON o.timestamp = t.target AND o.source IN ('measured', 'smoothed')
            ),
            scored AS (
                SELECT f.model, f.step, (f.target AT TIME ZONE $3)::date AS day,
                       f.forecast - r.actual AS e, r.actual, r.baseline, r.last_week,
                       f.low, f.high
                FROM f JOIN refs r ON r.target = f.target
            )
            SELECT model AS "model!", step, day,
                   COUNT(*) AS "count!",
                   COALESCE(SUM(ABS(e)), 0) AS "abs!",
                   COALESCE(SUM(e * e), 0) AS "squared!",
                   COALESCE(SUM(e), 0) AS "signed!",
                   COALESCE(SUM(actual), 0) AS "actual!",
                   COUNT(*) FILTER (WHERE e > 0) AS "over!",
                   COUNT(*) FILTER (WHERE ABS(e) <= 5) AS "within_5!",
                   COUNT(*) FILTER (WHERE ABS(e) <= 10) AS "within_10!",
                   COUNT(baseline) AS "baseline_count!",
                   COALESCE(SUM(ABS(e)) FILTER (WHERE baseline IS NOT NULL), 0)
                       AS "abs_on_baseline!",
                   COALESCE(SUM(ABS(baseline - actual)), 0) AS "baseline_abs!",
                   COUNT(last_week) AS "last_week_count!",
                   COALESCE(SUM(ABS(e)) FILTER (WHERE last_week IS NOT NULL), 0)
                       AS "abs_on_last_week!",
                   COALESCE(SUM(ABS(last_week - actual)), 0) AS "last_week_abs!",
                   COUNT(*) FILTER (WHERE low IS NOT NULL AND high IS NOT NULL) AS "banded!",
                   COUNT(*) FILTER (WHERE actual BETWEEN low AND high) AS "covered!",
                   COALESCE(SUM(high - low), 0) AS "width!",
                   COALESCE(SUM((high - low)
                       + 10 * (GREATEST(low - actual, 0) + GREATEST(actual - high, 0))), 0)
                       AS "interval_score!"
            FROM scored
            GROUP BY GROUPING SETS ((model, step), (model, day))
            ORDER BY model, step, day
            "#,
            since,
            until,
            tz.name()
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to score the model forecasts")?;

        let mut errors = ModelErrors::default();
        for row in rows {
            let (model, step, day, sums) = row.split();
            match (step, day) {
                (Some(step), _) => errors.by_step.push((
                    model,
                    u32::try_from(step).context("negative step")?,
                    sums,
                )),
                (None, Some(day)) => errors.by_day.push((model, day, sums)),
                (None, None) => {}
            }
        }
        Ok(errors)
    }

    /// `model`'s forecasts logged in `[since, until)` with the readings at
    /// their targets, oldest first.
    #[tracing::instrument(skip(self), fields(db.operation = "logged_outcomes"))]
    pub async fn logged_outcomes(
        &self,
        model: &str,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Result<Vec<LoggedOutcome>> {
        let rows = sqlx::query!(
            r#"
            SELECT l.made_at,
                   l.forecast::float8[] AS "forecast!: Vec<Option<f64>>",
                   ARRAY(
                       SELECT o.percentage
                       FROM generate_series(1, cardinality(l.forecast)) AS g(i)
                       LEFT JOIN occupancy_logs o
                         ON o.timestamp = l.made_at + g.i * INTERVAL '15 minutes'
                        AND o.source IN ('measured', 'smoothed')
                       ORDER BY g.i
                   ) AS "actual!: Vec<Option<f64>>"
            FROM model_forecast_log l
            WHERE l.model = $1 AND l.made_at >= $2 AND l.made_at < $3
            ORDER BY l.made_at
            "#,
            model,
            since,
            until
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load the logged forecasts with their readings")?;
        Ok(rows
            .into_iter()
            .map(|r| LoggedOutcome {
                made_at: r.made_at,
                forecast: r.forecast,
                actual: r.actual,
            })
            .collect())
    }
}
