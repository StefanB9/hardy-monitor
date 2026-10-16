//! The default forecast as it would have been on each backtest day: per
//! step the model then selected for its horizon, with the interval from
//! that model's errors before the day.

use chrono::{DateTime, Utc};

use super::{DayResult, Interval, Selection};
use crate::grid::SLOT;

/// What the default forecast falls back to, and what it is compared with.
#[derive(Debug, Clone)]
pub struct Defaults<L> {
    /// Used in this order when no selected model has a forecast (or nothing
    /// is selected yet).
    pub fallbacks: Vec<L>,
    /// The model recorded alongside the default forecast for comparison.
    pub comparison: Option<L>,
}

impl<L> Default for Defaults<L> {
    fn default() -> Self {
        Self {
            fallbacks: Vec::new(),
            comparison: None,
        }
    }
}

/// The default forecast of one past slot, as it would have been made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PastForecast<L> {
    /// The full hour it was made at.
    pub made_at: DateTime<Utc>,
    /// Slots after `made_at` (1 = the first).
    pub step: usize,
    pub target: DateTime<Utc>,
    /// The model the value came from.
    pub model: L,
    pub value: f64,
    pub bounds: Option<(f64, f64)>,
    /// The comparison model's forecast of the same slot.
    pub comparison: Option<f64>,
}

/// Appends the default forecast of every scored step of `day` to `out`:
/// the value of the model selected for the step's bucket, else of another
/// selected model, else of the first fallback that has one. `labels` and
/// `intervals` are indexed like `day.models`.
pub(super) fn day_defaults<L: Copy + PartialEq>(
    day: &DayResult,
    labels: &[L],
    selection: &Selection<L>,
    intervals: &[[Option<Interval>; 4]],
    defaults: &Defaults<L>,
    out: &mut Vec<PastForecast<L>>,
) {
    let index_of = |label: L| labels.iter().position(|&l| l == label);
    let value = |model: usize, s: usize| {
        day.models
            .get(model)
            .and_then(|m| m.values.get(s).copied().flatten())
    };
    let comparison = defaults.comparison.and_then(index_of);
    for (s, step) in day.steps.iter().enumerate() {
        let Some(&made_at) = day.origin_times.get(step.origin) else {
            continue;
        };
        let preferred = selection
            .iter()
            .find(|(b, _)| *b == step.bucket)
            .map(|(_, m)| *m);
        let chosen = preferred
            .into_iter()
            .chain(selection.iter().map(|(_, m)| *m))
            .chain(defaults.fallbacks.iter().copied())
            .find_map(|label| {
                let i = index_of(label)?;
                Some((label, i, value(i, s)?))
            });
        let Some((model, i, v)) = chosen else {
            continue;
        };
        let interval = intervals
            .get(i)
            .and_then(|by_bucket| by_bucket[step.bucket.index()]);
        out.push(PastForecast {
            made_at,
            step: step.horizon,
            target: made_at + SLOT * i32::try_from(step.horizon).unwrap_or(i32::MAX),
            model,
            value: v,
            bounds: interval.map(|iv| iv.around(v)),
            comparison: comparison.and_then(|c| value(c, s)),
        });
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeDelta;

    use super::*;
    use crate::{
        benchmark::{Bucket, ModelDay, Step},
        test_support::local,
    };

    /// Two origins an hour apart; steps 1 (next hour) and 6 (1–3 h) of
    /// each; model `m` forecasts `10 * m + step`, except model 1 has no
    /// forecast for step 6 of the second origin.
    fn day() -> DayResult {
        let steps: Vec<Step> = [(0, 1), (0, 6), (1, 1), (1, 6)]
            .into_iter()
            .map(|(origin, step)| Step {
                origin,
                horizon: step,
                bucket: Bucket::of_step(step),
                actual: 50.0,
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let models = (0..3)
            .map(|m| ModelDay {
                fitted: true,
                origins: 2,
                values: steps
                    .iter()
                    .enumerate()
                    .map(|(s, step)| {
                        (m != 1 || s != 3).then_some(10.0 * f64::from(m) + step.horizon as f64)
                    })
                    .collect(),
                ..ModelDay::default()
            })
            .collect();
        DayResult {
            origins: 2,
            origin_times: vec![local(5, 8, 0), local(5, 9, 0)],
            steps,
            models,
        }
    }

    #[test]
    fn test_day_defaults_use_the_selected_model_then_fallbacks() -> Result<()> {
        let interval = Interval {
            low: -2.0,
            high: 3.0,
        };
        let mut intervals = [[None; 4]; 3];
        intervals[1][Bucket::NextHour.index()] = Some(interval);
        let defaults = Defaults {
            fallbacks: vec![2],
            comparison: Some(0),
        };
        let mut out = Vec::new();
        day_defaults(
            &day(),
            &[0, 1, 2],
            &vec![(Bucket::NextHour, 1)],
            &intervals,
            &defaults,
            &mut out,
        );
        let models: Vec<usize> = out.iter().map(|p| p.model).collect();
        // Model 1 serves its own bucket and the unselected one; its gap
        // falls back to model 2.
        assert_eq!(models, vec![1, 1, 1, 2]);
        let first = out.first().context("no defaults")?;
        assert_eq!(first.made_at, local(5, 8, 0));
        assert_eq!(first.step, 1);
        assert_eq!(first.target, local(5, 8, 15));
        assert!((first.value - 11.0).abs() < 1e-12);
        assert_eq!(first.bounds, Some((9.0, 14.0)));
        assert_eq!(first.comparison, Some(1.0));
        let last = out.last().context("no defaults")?;
        assert_eq!(last.target, local(5, 9, 0) + TimeDelta::minutes(90));
        assert!((last.value - 26.0).abs() < 1e-12);
        // No interval known for model 2 there.
        assert_eq!(last.bounds, None);
        Ok(())
    }

    #[test]
    fn test_day_defaults_without_selection_or_fallback_record_nothing() {
        let mut out = Vec::new();
        day_defaults(
            &day(),
            &[0, 1, 2],
            &Vec::new(),
            &[[None; 4]; 3],
            &Defaults::default(),
            &mut out,
        );
        assert_eq!(out, Vec::new());
    }
}
