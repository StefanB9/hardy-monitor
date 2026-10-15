//! Ordering the leaderboard: the column it is sorted by, the family it is
//! filtered to, and the order models are offered in the chart's lists.

use std::cmp::Ordering;

use hardy_ml::{
    benchmark::Bucket,
    suite::{Family, ModelId},
};

use super::{Leaderboard, Row};

/// A column the table can be sorted by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Name,
    /// Mean absolute error in a bucket.
    Error(Bucket),
    /// Share of actual values inside the band.
    Coverage,
}

/// How the table is sorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sort {
    pub key: SortKey,
    pub descending: bool,
}

impl Default for Sort {
    /// Overall error, best first.
    fn default() -> Self {
        Self {
            key: SortKey::Error(Bucket::Overall),
            descending: false,
        }
    }
}

impl Sort {
    /// The sort after clicking `key`: the same column reverses, another
    /// starts ascending.
    #[must_use]
    pub fn clicked(self, key: SortKey) -> Self {
        Self {
            key,
            descending: self.key == key && !self.descending,
        }
    }

    /// `row`'s value in the sorted column, if it has one (names sort
    /// separately).
    fn value(self, row: &Row) -> Option<f64> {
        match self.key {
            SortKey::Name => None,
            SortKey::Error(bucket) => Bucket::ALL
                .iter()
                .position(|&b| b == bucket)
                .and_then(|i| row.mae[i]),
            SortKey::Coverage => row.coverage,
        }
    }

    fn compare(self, a: &Row, b: &Row) -> Ordering {
        let ordering = if self.key == SortKey::Name {
            a.name.cmp(b.name)
        } else {
            match (self.value(a), self.value(b)) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                // Missing values last, whatever the direction.
                (Some(_), None) => return Ordering::Less,
                (None, Some(_)) => return Ordering::Greater,
                (None, None) => return Ordering::Equal,
            }
        };
        if self.descending {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

impl Leaderboard {
    /// The rows of `family` (all with `None`) in `sort` order; rows without
    /// a value for the sorted column come last either way.
    pub fn view(&self, sort: Sort, family: Option<Family>) -> Vec<&Row> {
        let mut rows: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| family.is_none_or(|f| r.family == f))
            .collect();
        rows.sort_by(|a, b| sort.compare(a, b));
        rows
    }

    /// Every model in the order the chart offers them: family by family,
    /// best overall error first within a family, unscored ones last.
    pub fn model_order(&self) -> Vec<ModelId> {
        Family::ALL
            .iter()
            .flat_map(|&family| self.view(Sort::default(), Some(family)))
            .map(|r| r.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::leaderboard::Standing;

    fn row(id: ModelId, overall: Option<f64>, coverage: Option<f64>) -> Row {
        Row {
            id,
            name: id.name(),
            family: id.family(),
            mae: [overall, overall.map(|m| m / 2.0), None, None],
            coverage,
            selected: Vec::new(),
            standing: Standing::Pending,
            fitted_at: None,
        }
    }

    fn board(rows: Vec<Row>) -> Leaderboard {
        Leaderboard {
            rows,
            period: None,
            finished_at: None,
        }
    }

    fn ids(rows: &[&Row]) -> Vec<ModelId> {
        rows.iter().map(|r| r.id).collect()
    }

    #[test]
    fn test_sort_clicked_reverses_the_same_column_only() {
        let sort = Sort::default();
        let reversed = sort.clicked(SortKey::Error(Bucket::Overall));
        assert!(reversed.descending);
        assert!(!reversed.clicked(SortKey::Error(Bucket::Overall)).descending);
        let by_name = reversed.clicked(SortKey::Name);
        assert_eq!(by_name.key, SortKey::Name);
        assert!(!by_name.descending);
    }

    #[test]
    fn test_view_sorts_with_missing_values_last_both_ways() {
        let table = board(vec![
            row(ModelId::Naive, Some(3.0), Some(0.7)),
            row(ModelId::Ridge, None, None),
            row(ModelId::Mlp, Some(1.0), Some(0.9)),
            row(ModelId::StackedEnsemble, Some(2.0), None),
        ]);
        let ascending = table.view(Sort::default(), None);
        assert_eq!(
            ids(&ascending),
            [
                ModelId::Mlp,
                ModelId::StackedEnsemble,
                ModelId::Naive,
                ModelId::Ridge
            ]
        );
        let descending = table.view(
            Sort::default().clicked(SortKey::Error(Bucket::Overall)),
            None,
        );
        assert_eq!(
            ids(&descending),
            [
                ModelId::Naive,
                ModelId::StackedEnsemble,
                ModelId::Mlp,
                ModelId::Ridge
            ]
        );
        let coverage = Sort {
            key: SortKey::Coverage,
            descending: true,
        };
        assert_eq!(
            ids(&table.view(coverage, None))[..2],
            [ModelId::Mlp, ModelId::Naive]
        );
        let names = table.view(
            Sort {
                key: SortKey::Name,
                descending: false,
            },
            None,
        );
        let sorted: Vec<&str> = names.iter().map(|r| r.name).collect();
        let mut expected = sorted.clone();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_view_filters_to_one_family() {
        let table = board(vec![
            row(ModelId::Naive, Some(3.0), None),
            row(ModelId::Mlp, Some(1.0), None),
            row(ModelId::PatchTst, Some(2.0), None),
        ]);
        let neural = table.view(Sort::default(), Some(Family::NeuralNetwork));
        assert_eq!(ids(&neural), [ModelId::Mlp, ModelId::PatchTst]);
        assert_eq!(
            table.view(Sort::default(), Some(Family::Ensemble)),
            Vec::<&Row>::new()
        );
    }

    #[test]
    fn test_model_order_is_by_family_then_error() {
        let table = board(vec![
            row(ModelId::PatchTst, Some(2.0), None),
            row(ModelId::Naive, None, None),
            row(ModelId::Mlp, Some(1.0), None),
            row(ModelId::WeeklySeasonalNaive, Some(5.0), None),
            row(ModelId::Ridge, Some(9.0), None),
        ]);
        assert_eq!(
            table.model_order(),
            [
                ModelId::WeeklySeasonalNaive,
                ModelId::Naive,
                ModelId::Ridge,
                ModelId::Mlp,
                ModelId::PatchTst,
            ]
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// The view holds exactly the family's rows, sorted by the column,
        /// with every missing value after every present one.
        #[test]
        fn test_view_is_a_sorted_permutation(
            errors in proptest::collection::vec(proptest::option::of(0.0_f64..20.0), 1..40),
            descending: bool,
            family in 0_usize..7,
        ) {
            let rows: Vec<Row> = errors
                .iter()
                .zip(ModelId::ALL)
                .map(|(e, id)| row(*id, *e, None))
                .collect();
            let table = board(rows);
            let family = Family::ALL.get(family).copied();
            let sort = Sort { key: SortKey::Error(Bucket::Overall), descending };
            let view = table.view(sort, family);
            let expected = table
                .rows
                .iter()
                .filter(|r| family.is_none_or(|f| r.family == f))
                .count();
            prop_assert_eq!(view.len(), expected);
            prop_assert!(view.iter().all(|r| family.is_none_or(|f| r.family == f)));
            let values: Vec<Option<f64>> = view.iter().map(|r| r.mae[0]).collect();
            let first_missing = values.iter().position(Option::is_none).unwrap_or(values.len());
            prop_assert!(values[first_missing..].iter().all(Option::is_none));
            for pair in values[..first_missing].windows(2) {
                if let [Some(a), Some(b)] = pair {
                    let in_order = if descending { a >= b } else { a <= b };
                    prop_assert!(in_order);
                }
            }
        }
    }
}
