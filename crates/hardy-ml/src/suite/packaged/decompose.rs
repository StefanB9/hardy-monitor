//! Seasonal decomposition with stlrs: STL for one period, MSTL for several.
//! The seasonal part is removed before a model fits the rest and carried
//! forward by repeating each component's last period.

use crate::error::MlError;

/// The seasonal components of a decomposed series.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Seasonal {
    /// One component per period, each as long as the series.
    components: Vec<Vec<f64>>,
    periods: Vec<usize>,
}

impl Seasonal {
    /// The seasonal part `h` steps after the series ends (h ≥ 1).
    pub(super) fn ahead(&self, h: usize) -> f64 {
        self.components
            .iter()
            .zip(&self.periods)
            .map(|(component, &period)| {
                let n = component.len();
                component
                    .get(n - period + (h.max(1) - 1) % period)
                    .copied()
                    .unwrap_or(0.0)
            })
            .sum()
    }
}

/// Splits `values` into its seasonal part (periods `periods`) and the rest
/// (seasonally adjusted).
pub(super) fn decompose(
    values: &[f64],
    periods: &[usize],
) -> Result<(Seasonal, Vec<f64>), MlError> {
    let longest = periods
        .iter()
        .copied()
        .max()
        .ok_or_else(|| MlError::Fit("a decomposition needs a period".to_string()))?;
    if values.len() < 2 * longest {
        return Err(MlError::InsufficientData {
            found: values.len(),
            required: 2 * longest,
        });
    }
    let failed = |e: stlrs::Error| MlError::Fit(format!("decomposition failed: {e}"));
    let components = if let [period] = periods {
        vec![
            stlrs::Stl::fit(values, *period)
                .map_err(failed)?
                .seasonal()
                .to_vec(),
        ]
    } else {
        stlrs::Mstl::fit(values, periods)
            .map_err(failed)?
            .seasonal()
            .to_vec()
    };
    let adjusted = values
        .iter()
        .enumerate()
        .map(|(i, v)| v - components.iter().map(|c| c[i]).sum::<f64>())
        .collect();
    Ok((
        Seasonal {
            components,
            periods: periods.to_vec(),
        },
        adjusted,
    ))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;

    /// Level 10 plus a period-4 pattern, for `cycles` cycles.
    fn periodic(cycles: usize) -> Vec<f64> {
        let pattern = [1.0, 3.0, -2.0, -2.0];
        (0..cycles * 4).map(|i| 10.0 + pattern[i % 4]).collect()
    }

    #[test]
    fn test_stl_separates_a_pure_season() -> Result<()> {
        let values = periodic(12);
        let (seasonal, adjusted) = decompose(&values, &[4])?;
        assert_eq!(adjusted.len(), values.len());
        for value in &adjusted {
            assert_relative_eq!(*value, 10.0, epsilon = 0.2);
        }
        // The next cycle repeats the pattern.
        assert_relative_eq!(seasonal.ahead(1), 1.0, epsilon = 0.2);
        assert_relative_eq!(seasonal.ahead(2), 3.0, epsilon = 0.2);
        assert_relative_eq!(seasonal.ahead(6), 3.0, epsilon = 0.2);
        Ok(())
    }

    #[test]
    fn test_mstl_sums_its_components() -> Result<()> {
        // Periods 4 and 8: an extra +2 every other cycle half.
        let values: Vec<f64> = periodic(16)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + if i % 8 < 4 { 2.0 } else { -2.0 })
            .collect();
        let (seasonal, adjusted) = decompose(&values, &[4, 8])?;
        for (i, (value, rest)) in values.iter().zip(&adjusted).enumerate().skip(16) {
            let restored = rest + seasonal.components.iter().map(|c| c[i]).sum::<f64>();
            assert_relative_eq!(restored, *value, epsilon = 1e-6);
        }
        // Step 1 after 64 values: index 64, i % 4 = 0, i % 8 = 0.
        assert_relative_eq!(seasonal.ahead(1), 3.0, epsilon = 0.5);
        Ok(())
    }

    #[test]
    fn test_too_short_a_series_is_refused() {
        assert!(decompose(&[1.0, 2.0, 3.0], &[4]).is_err());
        assert!(decompose(&periodic(4), &[]).is_err());
    }
}
