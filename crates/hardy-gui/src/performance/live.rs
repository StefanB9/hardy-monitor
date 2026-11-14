//! Every model's live accuracy from its logged forecasts: per horizon
//! bucket for the leaderboard, and next to its backtest error.

use std::{collections::HashMap, hash::BuildHasher};

use hardy_core::{accuracy::ErrorSums, db::ModelErrors};
use hardy_ml::{benchmark::Bucket, suite::ModelId};

use crate::leaderboard::Leaderboard;

/// One model's live accuracy over the period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveScore {
    /// Mean absolute error per bucket, in [`Bucket::ALL`] order; the last
    /// bucket only reaches 6 h ahead live.
    pub mae: [Option<f64>; 4],
    /// Share of readings inside the band, over all lead times.
    pub coverage: Option<f64>,
    /// Scored forecasts.
    pub count: i64,
}

/// A model's backtest and live error an hour ahead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterPoint {
    pub id: ModelId,
    pub backtest: f64,
    pub live: f64,
    /// Selected for the next hour.
    pub selected: bool,
}

/// Every model's live score, by model key.
pub fn live_scores(errors: &ModelErrors) -> HashMap<String, LiveScore> {
    let mut buckets: HashMap<&str, [ErrorSums; 4]> = HashMap::new();
    for (model, step, sums) in &errors.by_step {
        let entry = buckets.entry(model.as_str()).or_default();
        let bucket = Bucket::of_step(usize::try_from(*step).unwrap_or(usize::MAX));
        if let Some(i) = Bucket::ALL.iter().position(|b| *b == bucket) {
            entry[i].merge(sums);
        }
        entry[0].merge(sums);
    }
    buckets
        .into_iter()
        .map(|(model, sums)| {
            let measures = sums.map(|s| s.measures());
            (
                model.to_string(),
                LiveScore {
                    mae: measures.map(|m| m.map(|m| m.mae)),
                    coverage: measures[0].and_then(|m| m.coverage),
                    count: sums[0].count,
                },
            )
        })
        .collect()
}

/// The leaderboard with live numbers in place of the backtest's.
pub fn live_board<S: BuildHasher>(
    board: &Leaderboard,
    live: &HashMap<String, LiveScore, S>,
) -> Leaderboard {
    let mut rows = board.rows.clone();
    for row in &mut rows {
        let score = live.get(row.id.key());
        row.mae = score.map_or([None; 4], |s| s.mae);
        row.coverage = score.and_then(|s| s.coverage);
    }
    Leaderboard {
        rows,
        period: board.period.clone(),
        finished_at: board.finished_at,
    }
}

/// Models with both a backtest and a live error for the next hour.
pub fn backtest_vs_live<S: BuildHasher>(
    board: &Leaderboard,
    live: &HashMap<String, LiveScore, S>,
) -> Vec<ScatterPoint> {
    board
        .rows
        .iter()
        .filter_map(|row| {
            Some(ScatterPoint {
                id: row.id,
                backtest: row.mae[1]?,
                live: live.get(row.id.key())?.mae[1]?,
                selected: row.selected.contains(&Bucket::NextHour),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use hardy_core::accuracy::{ErrorSums, Pair};

    use super::*;
    use crate::leaderboard::{Row, Standing};

    fn sums(errors: &[f64], banded: bool) -> ErrorSums {
        let mut sums = ErrorSums::default();
        for e in errors {
            sums.add(&Pair {
                actual: 30.0,
                forecast: 30.0 + e,
                band: banded.then_some((25.0, 35.0)),
                baseline: None,
                last_week: None,
            });
        }
        sums
    }

    fn errors() -> ModelErrors {
        ModelErrors {
            by_step: vec![
                ("naive".to_string(), 1, sums(&[2.0], true)),
                ("naive".to_string(), 4, sums(&[4.0], true)),
                ("naive".to_string(), 8, sums(&[6.0], true)),
                // 20 is 5 h ahead: the furthest bucket.
                ("naive".to_string(), 20, sums(&[8.0, 12.0], false)),
                ("ridge".to_string(), 2, sums(&[1.0], false)),
            ],
            by_day: Vec::new(),
        }
    }

    fn row(id: ModelId, next_hour: Option<f64>) -> Row {
        Row {
            id,
            name: id.name(),
            family: id.family(),
            mae: [Some(9.0), next_hour, None, None],
            coverage: Some(0.9),
            selected: vec![Bucket::NextHour],
            standing: Standing::Ready { days: 30 },
            fitted_at: None,
        }
    }

    #[test]
    fn test_live_scores_merge_steps_into_buckets() -> Result<()> {
        let scores = live_scores(&errors());
        let naive = scores.get("naive").context("naive")?;
        assert_eq!(naive.count, 5);
        let mae = naive.mae.map(Option::unwrap_or_default);
        assert_relative_eq!(mae[1], 3.0);
        assert_relative_eq!(mae[2], 6.0);
        assert_relative_eq!(mae[3], 10.0);
        assert_relative_eq!(mae[0], 32.0 / 5.0);
        // Three of five had a band, each with the reading inside.
        assert_relative_eq!(naive.coverage.context("coverage")?, 1.0);
        let ridge = scores.get("ridge").context("ridge")?;
        assert_eq!(
            (ridge.mae[2], ridge.mae[3], ridge.coverage),
            (None, None, None)
        );
        Ok(())
    }

    #[test]
    fn test_live_board_swaps_the_numbers_and_keeps_the_rest() {
        let board = Leaderboard {
            rows: vec![
                row(ModelId::Naive, Some(5.0)),
                row(ModelId::Theta, Some(4.0)),
            ],
            period: Some("backtest".to_string()),
            finished_at: None,
        };
        let live = live_board(&board, &live_scores(&errors()));
        assert_eq!(live.rows[0].mae[1], Some(3.0));
        assert_eq!(live.rows[0].selected, vec![Bucket::NextHour]);
        assert_eq!(live.rows[0].standing, Standing::Ready { days: 30 });
        // No live forecasts logged for this model.
        assert_eq!(live.rows[1].mae, [None; 4]);
        assert_eq!(live.rows[1].coverage, None);
    }

    #[test]
    fn test_backtest_vs_live_pairs_the_next_hour() {
        let board = Leaderboard {
            rows: vec![
                row(ModelId::Naive, Some(5.0)),
                row(ModelId::Theta, Some(4.0)),
                row(ModelId::Ridge, None),
            ],
            period: None,
            finished_at: None,
        };
        let points = backtest_vs_live(&board, &live_scores(&errors()));
        assert_eq!(
            points,
            vec![ScatterPoint {
                id: ModelId::Naive,
                backtest: 5.0,
                live: 3.0,
                selected: true,
            }]
        );
    }
}
