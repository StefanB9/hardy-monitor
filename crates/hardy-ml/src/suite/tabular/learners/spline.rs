//! Additive model with P-splines (Eilers & Marx 1996): every continuous
//! feature gets a cubic B-spline basis with a second-difference penalty,
//! two-valued features a linear term; one smoothing penalty for all, chosen
//! by generalised cross-validation.

use faer::{Mat, Side, prelude::Solve};
use serde::{Deserialize, Serialize};

use super::{Learner, check};
use crate::{
    error::MlError,
    suite::tabular::features::{FEATURES, Row, Rows},
};

/// Basis functions per continuous feature.
const BASES: usize = 8;
/// Penalties tried, relative to the number of rows.
const PENALTIES: [f64; 6] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0];
/// Tiny ridge on every column: the centred bases of one feature sum to
/// zero, so the unpenalised system is singular without it.
const JITTER: f64 = 1e-6;

/// The cardinal cubic B-spline on [0, 4).
fn cardinal(u: f64) -> f64 {
    if !(0.0..4.0).contains(&u) {
        0.0
    } else if u < 1.0 {
        u.powi(3) / 6.0
    } else if u < 2.0 {
        (-3.0 * u.powi(3) + 12.0 * u * u - 12.0 * u + 4.0) / 6.0
    } else if u < 3.0 {
        (3.0 * u.powi(3) - 24.0 * u * u + 60.0 * u - 44.0) / 6.0
    } else {
        (4.0 - u).powi(3) / 6.0
    }
}

/// How one feature enters the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Term {
    /// Two values at most: a linear term on the raw value.
    Linear,
    /// B-spline bases with uniform knots spanning `[low, high]`.
    Spline { low: f64, high: f64 },
}

impl Term {
    fn width(&self) -> usize {
        match self {
            Term::Linear => 1,
            Term::Spline { .. } => BASES,
        }
    }

    /// Appends this term's columns for `value` to `out`.
    fn expand(&self, value: f64, out: &mut Vec<f64>) {
        match *self {
            Term::Linear => out.push(value),
            Term::Spline { low, high } => {
                #[allow(clippy::cast_precision_loss)]
                let h = ((high - low) / (BASES - 3) as f64).max(1e-9);
                let x = value.clamp(low, high);
                #[allow(clippy::cast_precision_loss)]
                out.extend((0..BASES).map(|k| cardinal((x - (low + (k as f64 - 3.0) * h)) / h)));
            }
        }
    }
}

/// Second-difference penalty `DᵀD` on every spline block (row-major,
/// `width × width`).
fn difference_penalty(terms: &[Term], width: usize) -> Vec<f64> {
    let mut penalty = vec![0.0; width * width];
    let mut start = 0;
    for term in terms {
        if matches!(term, Term::Spline { .. }) {
            for k in 0..BASES - 2 {
                let row = [(k, 1.0), (k + 1, -2.0), (k + 2, 1.0)];
                for &(i, a) in &row {
                    for &(j, b) in &row {
                        penalty[(start + i) * width + start + j] += a * b;
                    }
                }
            }
        }
        start += term.width();
    }
    penalty
}

/// Solves `a · β = cross` and returns β with the trace of `a⁻¹ · gram` (the
/// effective degrees of freedom), from one Cholesky factorisation.
fn solve_with_trace(
    a: &[f64],
    cross: &[f64],
    gram: &[f64],
    width: usize,
) -> Option<(Vec<f64>, f64)> {
    let matrix = Mat::from_fn(width, width, |i, j| a[i * width + j]);
    let factor = matrix.llt(Side::Lower).ok()?;
    let rhs = Mat::from_fn(width, width + 1, |i, j| {
        if j == 0 {
            cross[i]
        } else {
            gram[i * width + j - 1]
        }
    });
    let solution = factor.solve(&rhs);
    let beta: Vec<f64> = (0..width).map(|i| solution[(i, 0)]).collect();
    let trace: f64 = (0..width).map(|j| solution[(j, j + 1)]).sum();
    (beta.iter().all(|b| b.is_finite()) && trace.is_finite()).then_some((beta, trace))
}

/// See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineGam {
    terms: Vec<Term>,
    /// Training mean of every column, subtracted before the coefficients.
    centres: Vec<f64>,
    intercept: f64,
    coefficients: Vec<f64>,
}

impl SplineGam {
    fn columns(&self, row: &Row) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.centres.len());
        for (term, value) in self.terms.iter().zip(row) {
            term.expand(*value, &mut out);
        }
        out
    }
}

impl Learner for SplineGam {
    const MIN_DAYS: i64 = 2;

    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let terms: Vec<Term> = (0..FEATURES)
            .map(|j| {
                let (low, high) = rows
                    .x
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), r| {
                        (l.min(r[j]), h.max(r[j]))
                    });
                // Exact: the values compared are copies of the same ones.
                #[allow(clippy::float_cmp)]
                let two_valued = rows.x.iter().all(|r| r[j] == low || r[j] == high);
                if two_valued {
                    Term::Linear
                } else {
                    Term::Spline { low, high }
                }
            })
            .collect();
        let width: usize = terms.iter().map(Term::width).sum();
        let mut model = Self {
            terms,
            centres: vec![0.0; width],
            intercept: 0.0,
            coefficients: vec![0.0; width],
        };
        let design: Vec<Vec<f64>> = rows.x.iter().map(|r| model.columns(r)).collect();
        #[allow(clippy::cast_precision_loss)]
        let n = design.len() as f64;
        for column in 0..width {
            model.centres[column] = design.iter().map(|r| r[column]).sum::<f64>() / n;
        }
        model.intercept = rows.y.iter().sum::<f64>() / n;
        let mut gram = vec![0.0; width * width];
        let mut cross = vec![0.0; width];
        let mut yy = 0.0;
        for (row, y) in design.iter().zip(&rows.y) {
            let x: Vec<f64> = row.iter().zip(&model.centres).map(|(v, c)| v - c).collect();
            let y = y - model.intercept;
            yy += y * y;
            for a in 0..width {
                cross[a] += x[a] * y;
                for b in a..width {
                    gram[a * width + b] += x[a] * x[b];
                }
            }
        }
        for a in 0..width {
            for b in 0..a {
                gram[a * width + b] = gram[b * width + a];
            }
        }
        let penalty = difference_penalty(&model.terms, width);
        let mut best: Option<(f64, Vec<f64>)> = None;
        for relative in PENALTIES {
            let a: Vec<f64> = gram
                .iter()
                .zip(&penalty)
                .enumerate()
                .map(|(i, (g, p))| {
                    g + relative * n * p
                        + if i % (width + 1) == 0 {
                            JITTER * n
                        } else {
                            0.0
                        }
                })
                .collect();
            let Some((beta, trace)) = solve_with_trace(&a, &cross, &gram, width) else {
                continue;
            };
            let g_beta: Vec<f64> = (0..width)
                .map(|i| (0..width).map(|j| gram[i * width + j] * beta[j]).sum())
                .collect();
            let fit: f64 = beta.iter().zip(&cross).map(|(b, c)| b * c).sum();
            let spread: f64 = beta.iter().zip(&g_beta).map(|(b, g)| b * g).sum();
            let rss = (yy - 2.0 * fit + spread).max(0.0);
            let gcv = n * rss / (n - trace).max(1.0).powi(2);
            if best.as_ref().is_none_or(|(g, _)| gcv < *g) {
                best = Some((gcv, beta));
            }
        }
        let (_, coefficients) =
            best.ok_or_else(|| MlError::Fit("spline system is singular".to_string()))?;
        model.coefficients = coefficients;
        Ok(model)
    }

    fn predict(&self, row: &Row) -> f64 {
        self.intercept
            + self
                .columns(row)
                .iter()
                .zip(&self.centres)
                .zip(&self.coefficients)
                .map(|((x, c), b)| (x - c) * b)
                .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;
    use crate::suite::tabular::learners::test_rows::rows;

    #[test]
    fn test_cardinal_spline_reference_values() {
        assert_relative_eq!(cardinal(0.0), 0.0);
        assert_relative_eq!(cardinal(1.0), 1.0 / 6.0);
        assert_relative_eq!(cardinal(2.0), 2.0 / 3.0);
        assert_relative_eq!(cardinal(3.0), 1.0 / 6.0);
        assert_relative_eq!(cardinal(4.0), 0.0);
        assert_relative_eq!(cardinal(-1.0), 0.0);
    }

    #[test]
    fn test_spline_bases_sum_to_one_inside_the_range() {
        let term = Term::Spline {
            low: -2.0,
            high: 3.0,
        };
        for x in [-2.0, -1.3, 0.0, 1.7, 3.0] {
            let mut out = Vec::new();
            term.expand(x, &mut out);
            assert_eq!(out.len(), BASES);
            assert_relative_eq!(out.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        }
        // Outside the range the value is clamped.
        let (mut inside, mut outside) = (Vec::new(), Vec::new());
        term.expand(3.0, &mut inside);
        term.expand(9.0, &mut outside);
        assert_eq!(inside, outside);
    }

    #[test]
    fn test_gam_recovers_a_smooth_nonlinear_relation() -> Result<()> {
        let truth = |r: &Row| (2.0 * r[0]).sin() + 0.1 * r[2] * r[2];
        let mut data = rows(3_000, truth);
        // A two-valued feature with a step effect.
        for (i, (row, y)) in data.x.iter_mut().zip(data.y.iter_mut()).enumerate() {
            row[5] = f64::from(u8::from(i % 3 == 0));
            *y += 2.0 * row[5];
        }
        let model = SplineGam::fit(&data)?;
        for (row, y) in data.x.iter().zip(&data.y).take(30) {
            assert_relative_eq!(model.predict(row), *y, epsilon = 0.15);
        }
        Ok(())
    }

    #[test]
    fn test_gam_refuses_empty_rows() {
        assert!(SplineGam::fit(&Rows::default()).is_err());
    }
}
