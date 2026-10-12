//! Ensembles: forecasts combined from other models' forecasts, with weights
//! per horizon bucket fitted on the members' backtest forecasts.

use serde::{Deserialize, Serialize};

use super::Bucket;
use crate::suite::linalg::simplex_least_squares;

/// Members the mean and median ensembles combine (those with the lowest
/// MAE).
pub const TOP_MEMBERS: usize = 5;
/// Share of a bucket's scored steps a model must have forecast to be a
/// member there.
pub const MIN_COVERAGE: f64 = 0.8;

/// How an ensemble combines its members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Combination {
    /// Mean of the [`TOP_MEMBERS`] best members.
    Mean,
    /// Median of the [`TOP_MEMBERS`] best members.
    Median,
    /// All members, weighted by the inverse of their MAE.
    InverseMae,
    /// The best member alone.
    BestMember,
    /// Least-squares weights, `≥ 0` and summing to 1.
    Stacked,
}

/// Scored backtest steps of one horizon bucket: the actual value and each
/// member's forecast (if it made one).
#[derive(Debug, Clone)]
pub struct Rows {
    members: usize,
    actual: Vec<f64>,
    /// Row-major: one entry per member per row.
    values: Vec<Option<f64>>,
}

impl Rows {
    /// No rows yet, for `members` members.
    pub fn new(members: usize) -> Self {
        Self {
            members,
            actual: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Adds a step: its actual value and each member's forecast, in member
    /// order (`values.len()` must equal the member count, else ignored).
    pub fn push(&mut self, actual: f64, values: &[Option<f64>]) {
        if values.len() == self.members {
            self.actual.push(actual);
            self.values.extend_from_slice(values);
        }
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.actual.len()
    }

    /// Whether there are no steps.
    pub fn is_empty(&self) -> bool {
        self.actual.is_empty()
    }

    fn row(&self, i: usize) -> &[Option<f64>] {
        &self.values[i * self.members..(i + 1) * self.members]
    }
}

/// An ensemble's members and their weights per horizon bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ensemble<L> {
    combination: Combination,
    /// Per bucket of [`Bucket::HORIZONS`]: members and weights.
    buckets: [Vec<(L, f64)>; 3],
}

impl<L: Copy> Ensemble<L> {
    /// Fits the weights of `combination` on `rows` (one per bucket of
    /// [`Bucket::HORIZONS`], with values in the order of `members`). `None`
    /// if no bucket has a member.
    pub fn fit(combination: Combination, members: &[L], rows: &[Rows; 3]) -> Option<Self> {
        let buckets: [Vec<(L, f64)>; 3] =
            std::array::from_fn(|i| fit_bucket(combination, members, &rows[i]));
        buckets.iter().any(|b| !b.is_empty()).then_some(Self {
            combination,
            buckets,
        })
    }

    /// The ensemble's forecast in `bucket`, from each member's forecast
    /// (`value`); members without one are skipped. `None` if no member has
    /// a forecast.
    pub fn combine(&self, bucket: Bucket, value: impl Fn(L) -> Option<f64>) -> Option<f64> {
        let members = self.members(bucket);
        if self.combination == Combination::Median {
            let mut values: Vec<f64> = members.iter().filter_map(|&(m, _)| value(m)).collect();
            return median(&mut values);
        }
        let (sum, total) = members
            .iter()
            .filter_map(|&(m, w)| value(m).map(|v| (v * w, w)))
            .fold((0.0, 0.0), |(s, t), (v, w)| (s + v, t + w));
        (total > 0.0).then(|| sum / total)
    }

    /// Members and weights in `bucket` (none for [`Bucket::Overall`]).
    pub fn members(&self, bucket: Bucket) -> &[(L, f64)] {
        horizon_index(bucket).map_or(&[], |i| &self.buckets[i])
    }

    /// How the members are combined.
    pub fn combination(&self) -> Combination {
        self.combination
    }

    /// The same ensemble with every member relabelled by `f`; `None` if `f`
    /// fails for any member.
    pub fn try_map<M>(&self, f: impl Fn(L) -> Option<M>) -> Option<Ensemble<M>> {
        let mut buckets: [Vec<(M, f64)>; 3] = std::array::from_fn(|_| Vec::new());
        for (bucket, members) in buckets.iter_mut().zip(&self.buckets) {
            *bucket = members
                .iter()
                .map(|&(l, w)| f(l).map(|m| (m, w)))
                .collect::<Option<_>>()?;
        }
        Some(Ensemble {
            combination: self.combination,
            buckets,
        })
    }
}

fn horizon_index(bucket: Bucket) -> Option<usize> {
    Bucket::HORIZONS.iter().position(|&b| b == bucket)
}

/// Members and weights of one bucket.
fn fit_bucket<L: Copy>(combination: Combination, members: &[L], rows: &Rows) -> Vec<(L, f64)> {
    let ranked = ranked_members(rows);
    let top = |k: usize| -> Vec<(L, f64)> {
        ranked
            .iter()
            .take(k)
            .map(|&(i, _)| (members[i], 1.0))
            .collect()
    };
    match combination {
        Combination::Mean | Combination::Median => top(TOP_MEMBERS),
        Combination::BestMember => top(1),
        Combination::InverseMae => {
            let inverse: Vec<f64> = ranked.iter().map(|&(_, mae)| 1.0 / mae.max(1e-9)).collect();
            let total: f64 = inverse.iter().sum();
            ranked
                .iter()
                .zip(inverse)
                .map(|(&(i, _), w)| (members[i], w / total))
                .collect()
        }
        Combination::Stacked => {
            let chosen: Vec<usize> = ranked.iter().map(|&(i, _)| i).collect();
            let kept: Vec<(L, f64)> = stacked_weights(rows, &chosen)
                .into_iter()
                .zip(&chosen)
                .filter(|(w, _)| *w > STACKED_MIN_WEIGHT)
                .map(|(w, &i)| (members[i], w))
                .collect();
            normalised(kept)
        }
    }
}

/// Weights below this are dropped from a stacked ensemble (the solver
/// leaves tiny remainders on members it has all but excluded).
const STACKED_MIN_WEIGHT: f64 = 1e-6;

/// `weights` rescaled to sum to 1.
fn normalised<L>(mut weights: Vec<(L, f64)>) -> Vec<(L, f64)> {
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    if total > 0.0 {
        for (_, w) in &mut weights {
            *w /= total;
        }
    }
    weights
}

/// Members (by index) that forecast at least [`MIN_COVERAGE`] of the steps,
/// with their MAE, best first; ties keep member order.
fn ranked_members(rows: &Rows) -> Vec<(usize, f64)> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut sums = vec![(0.0, 0_usize); rows.members];
    for i in 0..rows.len() {
        let actual = rows.actual[i];
        for (sum, value) in sums.iter_mut().zip(rows.row(i)) {
            if let Some(v) = value {
                sum.0 += (v - actual).abs();
                sum.1 += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let needed = MIN_COVERAGE * rows.len() as f64;
    let mut ranked: Vec<(usize, f64)> = sums
        .into_iter()
        .enumerate()
        .filter(|(_, (_, n))| {
            #[allow(clippy::cast_precision_loss)]
            let n = *n as f64;
            n > 0.0 && n >= needed
        })
        .map(|(i, (sum, n))| {
            #[allow(clippy::cast_precision_loss)]
            let mae = sum / n as f64;
            (i, mae)
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked
}

/// Convex least-squares weights of the `chosen` members on the steps where
/// all of them forecast; empty without such steps.
fn stacked_weights(rows: &Rows, chosen: &[usize]) -> Vec<f64> {
    let n = chosen.len();
    let mut gram = vec![0.0; n * n];
    let mut cross = vec![0.0; n];
    let mut x = vec![0.0; n];
    let mut complete = 0_usize;
    for i in 0..rows.len() {
        let row = rows.row(i);
        let filled = chosen
            .iter()
            .zip(x.iter_mut())
            .all(|(&m, slot)| row[m].map(|v| *slot = v).is_some());
        if !filled {
            continue;
        }
        complete += 1;
        let actual = rows.actual[i];
        for a in 0..n {
            cross[a] += x[a] * actual;
            for b in 0..n {
                gram[a * n + b] += x[a] * x[b];
            }
        }
    }
    if complete == 0 {
        return Vec::new();
    }
    simplex_least_squares(&gram, &cross)
}

/// Median of `values` (mean of the middle two for an even count).
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        f64::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    })
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    const ALL: [Combination; 5] = [
        Combination::Mean,
        Combination::Median,
        Combination::InverseMae,
        Combination::BestMember,
        Combination::Stacked,
    ];

    /// Rows where member `i` forecasts `actual + bias[i]` (`None` where
    /// `bias[i]` is NaN), the same in every bucket.
    fn biased_rows(biases: &[f64], steps: usize) -> [Rows; 3] {
        std::array::from_fn(|_| {
            let mut rows = Rows::new(biases.len());
            for s in 0..steps {
                #[allow(clippy::cast_precision_loss)]
                let actual = 20.0 + (s % 17) as f64 * 3.0;
                let values: Vec<Option<f64>> = biases
                    .iter()
                    .map(|&b| (!b.is_nan()).then_some(actual + b))
                    .collect();
                rows.push(actual, &values);
            }
            rows
        })
    }

    fn weight(ensemble: &Ensemble<usize>, bucket: Bucket, member: usize) -> f64 {
        ensemble
            .members(bucket)
            .iter()
            .find(|(m, _)| *m == member)
            .map_or(0.0, |(_, w)| *w)
    }

    #[test]
    fn test_mean_ensemble_averages_the_five_best_members() -> Result<()> {
        let biases = [9.0, 1.0, -2.0, 3.0, 20.0, -4.0, 5.0];
        let members: Vec<usize> = (0..biases.len()).collect();
        let e = Ensemble::fit(Combination::Mean, &members, &biased_rows(&biases, 50))
            .context("fitted")?;
        let chosen: Vec<usize> = e
            .members(Bucket::NextHour)
            .iter()
            .map(|(m, _)| *m)
            .collect();
        assert_eq!(chosen.len(), TOP_MEMBERS);
        for worst in [0, 4] {
            assert!(!chosen.contains(&worst), "{worst} is among the worst");
        }
        let value = e
            .combine(Bucket::NextHour, |m| {
                Some(f64::from(u8::try_from(m).unwrap_or(0)))
            })
            .context("combined")?;
        // Mean of members 1, 2, 3, 5 and 6.
        assert_relative_eq!(value, 17.0 / 5.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_median_ensemble_ignores_one_wild_member() -> Result<()> {
        let members: Vec<usize> = (0..5).collect();
        let e = Ensemble::fit(Combination::Median, &members, &biased_rows(&[0.0; 5], 30))
            .context("fitted")?;
        let values = [10.0, 11.0, 12.0, 13.0, 95.0];
        let value = e
            .combine(Bucket::ThreeToDay, |m| values.get(m).copied())
            .context("combined")?;
        assert_relative_eq!(value, 12.0);
        // An even number of forecasts averages the middle two.
        let value = e
            .combine(Bucket::ThreeToDay, |m| (m < 4).then(|| values[m]))
            .context("combined")?;
        assert_relative_eq!(value, 11.5);
        Ok(())
    }

    #[test]
    fn test_inverse_mae_weights_are_proportional_to_accuracy() -> Result<()> {
        let e = Ensemble::fit(
            Combination::InverseMae,
            &[0, 1],
            &biased_rows(&[1.0, -3.0], 40),
        )
        .context("fitted")?;
        assert_relative_eq!(
            weight(&e, Bucket::OneToThreeHours, 0),
            0.75,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            weight(&e, Bucket::OneToThreeHours, 1),
            0.25,
            epsilon = 1e-12
        );
        Ok(())
    }

    #[test]
    fn test_best_member_differs_per_bucket() -> Result<()> {
        let mut rows = biased_rows(&[1.0, 4.0], 30);
        rows[2] = biased_rows(&[6.0, -2.0], 30)[2].clone();
        let e = Ensemble::fit(Combination::BestMember, &[0, 1], &rows).context("fitted")?;
        assert_eq!(e.members(Bucket::NextHour), &[(0, 1.0)]);
        assert_eq!(e.members(Bucket::ThreeToDay), &[(1, 1.0)]);
        assert_eq!(e.members(Bucket::Overall), &[]);
        Ok(())
    }

    #[test]
    fn test_stacking_recovers_the_mix_that_cancels_biases() -> Result<()> {
        // 0.25·(a + 6) + 0.75·(a − 2) = a: the only convex mix without bias.
        let e = Ensemble::fit(
            Combination::Stacked,
            &[0, 1],
            &biased_rows(&[6.0, -2.0], 60),
        )
        .context("fitted")?;
        assert_relative_eq!(weight(&e, Bucket::NextHour, 0), 0.25, epsilon = 1e-6);
        assert_relative_eq!(weight(&e, Bucket::NextHour, 1), 0.75, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_members_with_too_few_forecasts_are_left_out() -> Result<()> {
        let mut rows = biased_rows(&[2.0, 0.5], 50);
        // Member 1 (the better one) forecast only 30 of 50 steps.
        for bucket in &mut rows {
            let mut thin = Rows::new(2);
            for i in 0..bucket.len() {
                let row = bucket.row(i);
                thin.push(bucket.actual[i], &[row[0], row[1].filter(|_| i % 5 < 3)]);
            }
            *bucket = thin;
        }
        for combination in ALL {
            let e = Ensemble::fit(combination, &[0, 1], &rows).context("fitted")?;
            assert!(
                e.members(Bucket::NextHour).iter().all(|(m, _)| *m == 0),
                "{combination:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_combine_renormalises_over_available_members() -> Result<()> {
        let e = Ensemble::fit(
            Combination::InverseMae,
            &[0, 1],
            &biased_rows(&[1.0, -3.0], 40),
        )
        .context("fitted")?;
        let value = e
            .combine(Bucket::NextHour, |m| (m == 1).then_some(40.0))
            .context("combined")?;
        assert_relative_eq!(value, 40.0);
        assert_eq!(e.combine(Bucket::NextHour, |_| None), None);
        assert_eq!(e.combine(Bucket::Overall, |_| Some(1.0)), None);
        Ok(())
    }

    #[test]
    fn test_no_rows_means_no_ensemble() {
        let empty: [Rows; 3] = std::array::from_fn(|_| Rows::new(2));
        for combination in ALL {
            assert_eq!(Ensemble::fit(combination, &[0, 1], &empty), None);
        }
    }

    #[test]
    fn test_try_map_relabels_members() -> Result<()> {
        let e = Ensemble::fit(
            Combination::BestMember,
            &[0, 1],
            &biased_rows(&[1.0, 4.0], 30),
        )
        .context("fitted")?;
        let named = e
            .try_map(|m| ["a", "b"].get(m).copied())
            .context("mapped")?;
        assert_eq!(named.members(Bucket::NextHour), &[("a", 1.0)]);
        assert_eq!(named.combination(), Combination::BestMember);
        assert!(e.try_map(|m| (m == 5).then_some(m)).is_none());
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Every ensemble forecast lies between its members' forecasts.
        #[test]
        fn test_ensembles_stay_within_their_members(
            biases in prop::collection::vec(-15.0_f64..15.0, 2..8),
            forecasts in prop::collection::vec(prop::option::of(0.0_f64..100.0), 8),
            which in 0_usize..5,
        ) {
            let members: Vec<usize> = (0..biases.len()).collect();
            let rows = biased_rows(&biases, 40);
            let e = Ensemble::fit(ALL[which], &members, &rows);
            prop_assert!(e.is_some());
            if let Some(e) = e {
                for bucket in Bucket::HORIZONS {
                    let available: Vec<f64> = e
                        .members(bucket)
                        .iter()
                        .filter_map(|(m, _)| forecasts[*m])
                        .collect();
                    let value = e.combine(bucket, |m| forecasts[m]);
                    prop_assert_eq!(value.is_some(), !available.is_empty());
                    if let Some(value) = value {
                        let low = available.iter().copied().fold(f64::INFINITY, f64::min);
                        let high = available.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        prop_assert!(value >= low - 1e-9 && value <= high + 1e-9);
                    }
                    let total: f64 = e.members(bucket).iter().map(|(_, w)| w).sum();
                    prop_assert!(e.members(bucket).iter().all(|(_, w)| *w > 0.0));
                    if matches!(ALL[which], Combination::InverseMae | Combination::Stacked) {
                        prop_assert!((total - 1.0).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn test_rows_ignore_misshaped_steps() {
        let mut rows = Rows::new(2);
        rows.push(1.0, &[Some(1.0)]);
        assert!(rows.is_empty());
        rows.push(1.0, &[Some(1.0), None]);
        assert_eq!(rows.len(), 1);
    }
}
