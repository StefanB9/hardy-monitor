//! The forecast leaderboard on the Model & Data page: every model with its
//! benchmark error per horizon and whether it is in use, best first.

mod order;

use chrono::{DateTime, Utc};
use hardy_core::db::FitStatus;
use hardy_ml::{
    benchmark::Bucket,
    suite::{Family, ModelId},
};

pub use self::order::{Sort, SortKey};
use crate::forecasting::SuiteStatus;

/// Where a model stands.
#[derive(Debug, Clone, PartialEq)]
pub enum Standing {
    /// Fitted on `days` days of data and benchmarked on enough forecasts.
    Ready {
        days: i64,
    },
    /// Fitted, but too few benchmark forecasts yet to be chosen.
    Learning {
        days: i64,
        forecasts: u32,
    },
    /// Not enough history yet.
    Waiting(String),
    Failed(String),
    /// Not run yet.
    Pending,
}

/// Short name of a family for chips and row tags.
pub fn family_label(family: Family) -> &'static str {
    match family {
        Family::Baseline => "Baselines",
        Family::Profile => "Profile",
        Family::Statistical => "Statistical",
        Family::MachineLearning => "ML",
        Family::NeuralNetwork => "Neural",
        Family::Ensemble => "Ensembles",
    }
}

/// One model's line.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub id: ModelId,
    pub name: &'static str,
    pub family: Family,
    /// Mean absolute error per bucket, in [`Bucket::ALL`] order.
    pub mae: [Option<f64>; 4],
    /// Share of actual values inside the 80 % band, over all horizons.
    pub coverage: Option<f64>,
    /// Buckets this model is shown for by default.
    pub selected: Vec<Bucket>,
    pub standing: Standing,
}

/// The whole table.
#[derive(Debug, Clone, PartialEq)]
pub struct Leaderboard {
    pub rows: Vec<Row>,
    /// Backtested days, e.g. "12 May – 8 Jun · 28 days · 480 forecasts".
    pub period: Option<String>,
    pub finished_at: Option<DateTime<Utc>>,
}

/// Builds the table from what the daemon reported.
pub fn build(status: &SuiteStatus) -> Leaderboard {
    let run = status.benchmark.as_ref();
    let rows: Vec<Row> = ModelId::ALL
        .iter()
        .map(|id| {
            let key = id.key();
            let score = |bucket: Bucket| {
                run?.scores
                    .iter()
                    .find(|s| s.model == key && s.bucket == bucket.code())
            };
            let benchmarked = run.and_then(|r| r.models.iter().find(|m| m.model == key));
            let record = status.records.iter().find(|r| r.model == key);
            let standing = match record {
                None => Standing::Pending,
                Some(r) => match r.status {
                    FitStatus::Failed => Standing::Failed(r.message.clone().unwrap_or_default()),
                    FitStatus::Waiting => Standing::Waiting(r.message.clone().unwrap_or_default()),
                    FitStatus::Fitted => {
                        let days = r
                            .data_from
                            .zip(r.data_until)
                            .map_or(0, |(from, until)| (until - from).num_days());
                        match benchmarked {
                            Some(b) if b.eligible => Standing::Ready { days },
                            b => Standing::Learning {
                                days,
                                forecasts: b.map_or(0, |b| b.origins),
                            },
                        }
                    }
                },
            };
            Row {
                id: *id,
                name: id.name(),
                family: id.family(),
                mae: Bucket::ALL.map(|b| score(b).map(|s| s.mae)),
                coverage: score(Bucket::Overall).and_then(|s| s.coverage),
                selected: status
                    .selection
                    .iter()
                    .filter(|(_, model)| model == key)
                    .filter_map(|(code, _)| Bucket::from_code(*code))
                    .collect(),
                standing,
            }
        })
        .collect();
    Leaderboard {
        rows,
        period: run.map(|r| {
            let span = match (r.first_day, r.last_day) {
                (Some(first), Some(last)) => {
                    format!("{} – {} · ", first.format("%-d %b"), last.format("%-d %b"))
                }
                _ => String::new(),
            };
            format!("{span}{} days · {} forecasts", r.days, r.origins)
        }),
        finished_at: run.map(|r| r.finished_at),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{NaiveDate, TimeDelta, TimeZone};
    use hardy_core::db::{BenchmarkModel, BenchmarkRun, BenchmarkScore, ModelRecord};

    use super::*;

    fn at(day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, day, 22, 0, 0)
            .single()
            .unwrap_or_default()
    }

    fn record(model: &str, status: FitStatus, message: Option<&str>) -> ModelRecord {
        ModelRecord {
            model: model.to_string(),
            status,
            message: message.map(ToString::to_string),
            fitted_at: (status == FitStatus::Fitted).then(|| at(10)),
            data_from: (status == FitStatus::Fitted).then(|| at(10) - TimeDelta::days(9)),
            data_until: (status == FitStatus::Fitted).then(|| at(10)),
            updated_at: at(10),
        }
    }

    fn score(model: &str, bucket: Bucket, mae: f64) -> BenchmarkScore {
        BenchmarkScore {
            model: model.to_string(),
            bucket: bucket.code(),
            mae,
            rmse: mae,
            mase: None,
            coverage: Some(0.75),
            pinball: None,
            errors: 10,
            interval: None,
        }
    }

    fn status() -> SuiteStatus {
        let benchmarked = |model: &str, origins: u32| BenchmarkModel {
            model: model.to_string(),
            origins,
            fitted_days: 5,
            eligible: origins >= 24,
            failure: None,
        };
        SuiteStatus {
            records: vec![
                record("naive", FitStatus::Fitted, None),
                record("daily_seasonal_naive", FitStatus::Fitted, None),
                record("ridge", FitStatus::Fitted, None),
                record(
                    "weekly_seasonal_naive",
                    FitStatus::Waiting,
                    Some("needs 7 days of history, has 5"),
                ),
                record(
                    "random_forest",
                    FitStatus::Failed,
                    Some("model crashed: boom"),
                ),
            ],
            benchmark: Some(BenchmarkRun {
                id: 1,
                started_at: at(10),
                finished_at: at(10) + TimeDelta::minutes(3),
                first_day: NaiveDate::from_ymd_opt(2024, 6, 5),
                last_day: NaiveDate::from_ymd_opt(2024, 6, 9),
                days: 5,
                origins: 90,
                models: vec![
                    benchmarked("naive", 90),
                    benchmarked("daily_seasonal_naive", 72),
                    benchmarked("ridge", 10),
                ],
                scores: vec![
                    score("naive", Bucket::Overall, 9.0),
                    score("naive", Bucket::NextHour, 2.0),
                    score("daily_seasonal_naive", Bucket::Overall, 4.0),
                    score("ridge", Bucket::Overall, 1.0),
                ],
            }),
            selection: vec![
                (Bucket::NextHour.code(), "naive".to_string()),
                (
                    Bucket::ThreeToDay.code(),
                    "daily_seasonal_naive".to_string(),
                ),
            ],
            state: hardy_core::db::MlState::default(),
        }
    }

    #[test]
    fn test_rows_carry_scores_selection_and_standing() -> Result<()> {
        let board = build(&status());
        let view = board.view(Sort::default(), None);
        let names: Vec<_> = view.iter().take(3).map(|r| r.name).collect();
        assert_eq!(
            names,
            ["Ridge regression", "Daily seasonal naive", "Persistence"]
        );
        // Ridge has the lowest error but too few forecasts to count yet.
        assert_eq!(
            view[0].standing,
            Standing::Learning {
                days: 9,
                forecasts: 10
            }
        );
        let naive = board
            .rows
            .iter()
            .find(|r| r.id == ModelId::Naive)
            .context("naive")?;
        assert_eq!(naive.family, Family::Baseline);
        assert_eq!(naive.mae, [Some(9.0), Some(2.0), None, None]);
        assert_eq!(naive.coverage, Some(0.75));
        assert_eq!(naive.selected, vec![Bucket::NextHour]);
        assert_eq!(naive.standing, Standing::Ready { days: 9 });
        assert_eq!(
            board.period.as_deref(),
            Some("5 Jun – 9 Jun · 5 days · 90 forecasts")
        );
        assert!(board.finished_at.is_some());
        Ok(())
    }

    #[test]
    fn test_waiting_failed_and_pending_models_are_listed() -> Result<()> {
        let board = build(&status());
        assert_eq!(board.rows.len(), ModelId::ALL.len());
        let find = |name: &str| board.rows.iter().find(|r| r.name == name);
        assert_eq!(
            find("Weekly seasonal naive").context("weekly")?.standing,
            Standing::Waiting("needs 7 days of history, has 5".to_string())
        );
        assert_eq!(
            find("Random forest").context("forest")?.standing,
            Standing::Failed("model crashed: boom".to_string())
        );
        assert_eq!(
            find("Recent moving average")
                .context("moving average")?
                .standing,
            Standing::Pending
        );
        Ok(())
    }

    #[test]
    fn test_family_labels_are_short_and_distinct() {
        let labels: std::collections::HashSet<_> =
            Family::ALL.iter().map(|f| family_label(*f)).collect();
        assert_eq!(labels.len(), Family::ALL.len());
        assert!(labels.iter().all(|l| !l.is_empty() && l.len() <= 11));
    }

    #[test]
    fn test_empty_status_lists_every_model_as_pending() {
        let board = build(&SuiteStatus::default());
        assert!(board.rows.iter().all(|r| r.standing == Standing::Pending));
        assert_eq!(board.period, None);
    }
}
