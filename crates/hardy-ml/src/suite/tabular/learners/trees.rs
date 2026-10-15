//! Tree ensembles: a random forest (smartcore) and gradient-boosted trees
//! (forust-ml).

use forust_ml::{GradientBooster, Matrix, objective::ObjectiveType};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::basic::matrix::DenseMatrix,
};

use super::{Learner, check};
use crate::{
    error::MlError,
    suite::tabular::features::{FEATURES, Row, Rows},
};

/// Trees in the forest.
const FOREST_TREES: usize = 100;
/// Deepest tree of the forest.
const FOREST_DEPTH: u16 = 12;
/// Smallest leaf of the forest.
const FOREST_LEAF: usize = 10;
/// Fixed seed, so a fit is reproducible.
const SEED: u64 = 42;
/// Boosting rounds.
const BOOST_ROUNDS: usize = 300;
/// Boosting learning rate.
const BOOST_RATE: f32 = 0.1;
/// Deepest boosted tree.
const BOOST_DEPTH: usize = 6;

type Forest = RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

/// Random forest regression (smartcore).
#[derive(Debug, Serialize, Deserialize)]
pub struct RandomForest {
    forest: Forest,
}

impl Learner for RandomForest {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let parameters = RandomForestRegressorParameters::default()
            .with_n_trees(FOREST_TREES)
            .with_max_depth(FOREST_DEPTH)
            .with_min_samples_leaf(FOREST_LEAF)
            .with_min_samples_split(FOREST_LEAF * 2)
            .with_seed(SEED);
        let forest = RandomForestRegressor::fit(&matrix(&rows.x)?, &rows.y, parameters)
            .map_err(|e| MlError::Fit(format!("random forest: {e}")))?;
        Ok(Self { forest })
    }

    fn predict(&self, row: &Row) -> f64 {
        matrix(std::slice::from_ref(row))
            .ok()
            .and_then(|x| self.forest.predict(&x).ok())
            .and_then(|values| values.first().copied())
            .unwrap_or(0.0)
    }
}

/// `rows` as a smartcore matrix.
fn matrix(rows: &[Row]) -> Result<DenseMatrix<f64>, MlError> {
    let nested: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    DenseMatrix::from_2d_vec(&nested).map_err(|e| MlError::Fit(format!("feature matrix: {e}")))
}

/// Gradient-boosted trees (forust-ml), stored as forust's own JSON: its
/// serde derive does not survive a postcard round trip.
pub struct BoostedTrees {
    booster: GradientBooster,
}

impl std::fmt::Debug for BoostedTrees {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoostedTrees").finish_non_exhaustive()
    }
}

impl Serialize for BoostedTrees {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let json = self
            .booster
            .json_dump()
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&json)
    }
}

impl<'de> Deserialize<'de> for BoostedTrees {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let json = String::deserialize(deserializer)?;
        GradientBooster::from_json(&json)
            .map(|booster| Self { booster })
            .map_err(serde::de::Error::custom)
    }
}

impl Learner for BoostedTrees {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let mut booster = GradientBooster::default()
            .set_objective_type(ObjectiveType::SquaredLoss)
            .set_iterations(BOOST_ROUNDS)
            .set_learning_rate(BOOST_RATE)
            .set_max_depth(BOOST_DEPTH)
            .set_seed(SEED)
            .set_parallel(false);
        let columns = column_major(&rows.x);
        let data = Matrix::new(&columns, rows.x.len(), FEATURES);
        let weights = vec![1.0; rows.y.len()];
        booster
            .fit(&data, &rows.y, &weights, None)
            .map_err(|e| MlError::Fit(format!("boosted trees: {e}")))?;
        Ok(Self { booster })
    }

    fn predict(&self, row: &Row) -> f64 {
        let data = Matrix::new(row, 1, FEATURES);
        self.booster
            .predict(&data, false)
            .first()
            .copied()
            .unwrap_or(0.0)
    }
}

/// `rows` column by column, as forust's matrix expects.
fn column_major(rows: &[Row]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * FEATURES);
    for j in 0..FEATURES {
        out.extend(rows.iter().map(|r| r[j]));
    }
    out
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::suite::{
        decode, encode,
        tabular::learners::test_rows::{noise, rows},
    };

    /// A step and an interaction: hard for a line, easy for trees.
    fn truth(row: &Row) -> f64 {
        (if row[0] > 0.0 { 5.0 } else { -5.0 })
            + if row[1] > 0.0 && row[2] > 0.0 {
                3.0
            } else {
                0.0
            }
    }

    fn mae(model: &impl Learner, data: &Rows) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n = data.x.len() as f64;
        data.x
            .iter()
            .zip(&data.y)
            .map(|(r, y)| (model.predict(r) - y).abs())
            .sum::<f64>()
            / n
    }

    #[test]
    fn test_forest_learns_steps_and_round_trips() -> Result<()> {
        let mut e = noise(5);
        let data = rows(2_000, |r| truth(r) + 0.2 * e());
        let test = rows(300, truth);
        let model = RandomForest::fit(&data)?;
        assert!(mae(&model, &test) < 1.0, "MAE {}", mae(&model, &test));
        let restored: RandomForest = decode(&encode(&model)?)?;
        assert_eq!(model.predict(&test.x[0]), restored.predict(&test.x[0]));
        Ok(())
    }

    #[test]
    fn test_boosted_trees_learn_steps_and_round_trip() -> Result<()> {
        let mut e = noise(6);
        let data = rows(2_000, |r| truth(r) + 0.2 * e());
        let test = rows(300, truth);
        let model = BoostedTrees::fit(&data)?;
        assert!(mae(&model, &test) < 1.0, "MAE {}", mae(&model, &test));
        let restored: BoostedTrees = decode(&encode(&model)?)?;
        for row in test.x.iter().take(10) {
            assert!((model.predict(row) - restored.predict(row)).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn test_tree_learners_refuse_empty_rows() {
        assert!(RandomForest::fit(&Rows::default()).is_err());
        assert!(BoostedTrees::fit(&Rows::default()).is_err());
    }
}
