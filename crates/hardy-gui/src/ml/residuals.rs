use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 10th percentile for lower bound of 80% prediction interval.
pub const LOWER_PERCENTILE: f64 = 10.0;
/// 90th percentile for upper bound of 80% prediction interval.
pub const UPPER_PERCENTILE: f64 = 90.0;
/// Minimum residuals needed to use slot-specific quantiles.
pub const MIN_RESIDUALS_PER_SLOT: usize = 10;
/// Horizon scaling factor — interval widens by this fraction per additional
/// hour.
pub const HORIZON_SCALING_FACTOR: f64 = 0.10;
/// Maximum interval width used for confidence score normalization.
pub const MAX_INTERVAL_WIDTH: f64 = 100.0;

/// Quantile pair for a single (weekday, hour) slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotQuantiles {
    /// Lower quantile residual (10th percentile, typically negative).
    pub q_low: f64,
    /// Upper quantile residual (90th percentile, typically positive).
    pub q_high: f64,
    /// Number of residuals used to compute these quantiles.
    pub count: usize,
}

/// Residual-based quantile lookup table for confidence intervals.
///
/// Built from cross-validation residuals grouped by (weekday, hour) slot.
/// At prediction time, provides empirically calibrated prediction intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualQuantiles {
    /// Per-slot quantiles keyed by (weekday 0–6, hour 0–23).
    slot_quantiles: HashMap<(u32, u32), SlotQuantiles>,
    /// Global quantiles computed from all residuals (fallback).
    global_quantiles: SlotQuantiles,
    /// Minimum residuals required to use a slot's own quantiles.
    min_residuals_per_slot: usize,
}

/// Compute a percentile from a sorted slice using linear interpolation.
///
/// `percentile` is in [0, 100]. Returns the interpolated value at that
/// percentile.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_quantile(sorted: &[f64], percentile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=100.0).contains(&percentile));

    if sorted.len() == 1 {
        return sorted[0];
    }

    let n = sorted.len();
    let rank = percentile / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - rank.floor();

    if lower == upper || upper >= n {
        sorted[lower.min(n - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Build `SlotQuantiles` from a slice of residuals.
fn build_quantiles(residuals: &mut [f64]) -> SlotQuantiles {
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_low = compute_quantile(residuals, LOWER_PERCENTILE);
    let q_high = compute_quantile(residuals, UPPER_PERCENTILE);
    SlotQuantiles {
        q_low,
        q_high,
        count: residuals.len(),
    }
}

impl ResidualQuantiles {
    /// Build quantile lookup from (weekday, hour, residual) triples.
    ///
    /// Returns `None` if `residuals` is empty.
    pub fn from_residuals(residuals: &[(u32, u32, f64)]) -> Option<Self> {
        if residuals.is_empty() {
            return None;
        }

        // Group residuals by (weekday, hour)
        let mut slot_map: HashMap<(u32, u32), Vec<f64>> = HashMap::new();
        let mut all_residuals = Vec::with_capacity(residuals.len());

        for &(weekday, hour, residual) in residuals {
            slot_map.entry((weekday, hour)).or_default().push(residual);
            all_residuals.push(residual);
        }

        // Build global quantiles from all residuals
        let global_quantiles = build_quantiles(&mut all_residuals);

        // Build per-slot quantiles
        let slot_quantiles: HashMap<(u32, u32), SlotQuantiles> = slot_map
            .into_iter()
            .map(|(key, mut vals)| (key, build_quantiles(&mut vals)))
            .collect();

        Some(Self {
            slot_quantiles,
            global_quantiles,
            min_residuals_per_slot: MIN_RESIDUALS_PER_SLOT,
        })
    }

    /// Get the quantiles for a specific (weekday, hour) slot.
    ///
    /// Falls back to global quantiles if the slot has fewer than
    /// `min_residuals_per_slot` residuals.
    pub fn get_quantiles(&self, weekday: u32, hour: u32) -> &SlotQuantiles {
        self.slot_quantiles
            .get(&(weekday, hour))
            .filter(|q| q.count >= self.min_residuals_per_slot)
            .unwrap_or(&self.global_quantiles)
    }

    /// Compute a confidence interval for a prediction.
    ///
    /// Returns `(confidence_low, confidence_high, confidence_score)` with
    /// horizon scaling applied.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_confidence_interval(
        &self,
        predicted: f64,
        weekday: u32,
        hour: u32,
        hours_ahead: i64,
    ) -> (f64, f64, f64) {
        let quantiles = self.get_quantiles(weekday, hour);
        let horizon_factor = 1.0 + (hours_ahead.max(1) - 1) as f64 * HORIZON_SCALING_FACTOR;

        let low = (predicted + quantiles.q_low * horizon_factor).clamp(0.0, 100.0);
        let high = (predicted + quantiles.q_high * horizon_factor).clamp(0.0, 100.0);

        // Swap if inverted after clamping
        let (final_low, final_high) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };

        let width = final_high - final_low;
        let score = (1.0 - width / MAX_INTERVAL_WIDTH).clamp(0.1, 0.95);

        (final_low, final_high, score)
    }

    /// Reconstruct from pre-computed quantiles (e.g. loaded from disk).
    pub fn from_persisted(
        slot_quantiles: HashMap<(u32, u32), SlotQuantiles>,
        global_quantiles: SlotQuantiles,
    ) -> Self {
        Self {
            slot_quantiles,
            global_quantiles,
            min_residuals_per_slot: MIN_RESIDUALS_PER_SLOT,
        }
    }

    /// Access the per-slot quantiles map (for persistence serialization).
    pub fn slot_quantiles_map(&self) -> &HashMap<(u32, u32), SlotQuantiles> {
        &self.slot_quantiles
    }

    /// Access the global quantiles (for persistence serialization).
    pub fn global_quantiles(&self) -> &SlotQuantiles {
        &self.global_quantiles
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    // ── compute_quantile tests ────────────────────────────────────────

    #[test]
    fn test_compute_quantile_simple() {
        // 10 evenly spaced values: 0, 1, 2, ..., 9
        let sorted: Vec<f64> = (0..10).map(f64::from).collect();
        let q10 = compute_quantile(&sorted, 10.0);
        let q90 = compute_quantile(&sorted, 90.0);

        // 10th percentile of [0..9]: rank = 0.1 * 9 = 0.9
        // interpolate: 0 * 0.1 + 1 * 0.9 = 0.9
        assert_relative_eq!(q10, 0.9, epsilon = 1e-10);
        // 90th percentile: rank = 0.9 * 9 = 8.1
        // interpolate: 8 * 0.9 + 9 * 0.1 = 8.1
        assert_relative_eq!(q90, 8.1, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_quantile_single_value() {
        let sorted = vec![42.0];
        assert_relative_eq!(compute_quantile(&sorted, 10.0), 42.0);
        assert_relative_eq!(compute_quantile(&sorted, 50.0), 42.0);
        assert_relative_eq!(compute_quantile(&sorted, 90.0), 42.0);
    }

    #[test]
    fn test_compute_quantile_two_values() {
        let sorted = vec![10.0, 20.0];
        // 10th percentile: rank = 0.1 * 1 = 0.1
        // interpolate: 10 * 0.9 + 20 * 0.1 = 11.0
        assert_relative_eq!(compute_quantile(&sorted, 10.0), 11.0, epsilon = 1e-10);
        // 90th: rank = 0.9 * 1 = 0.9
        // interpolate: 10 * 0.1 + 20 * 0.9 = 19.0
        assert_relative_eq!(compute_quantile(&sorted, 90.0), 19.0, epsilon = 1e-10);
    }

    // ── from_residuals tests ──────────────────────────────────────────

    #[test]
    fn test_from_residuals_empty() {
        let result = ResidualQuantiles::from_residuals(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_from_residuals_single_slot() {
        // 20 residuals for slot (1, 10), all between -5 and 5
        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (1, 10, -5.0 + f64::from(i) * 0.5))
            .collect();

        let quantiles = ResidualQuantiles::from_residuals(&residuals);
        assert!(quantiles.is_some());

        let q = quantiles.unwrap_or_else(|| unreachable!());
        assert_eq!(q.slot_quantiles.len(), 1);
        assert!(q.slot_quantiles.contains_key(&(1, 10)));

        let slot = q.get_quantiles(1, 10);
        assert_eq!(slot.count, 20);
        assert!(slot.q_low < slot.q_high);
    }

    #[test]
    fn test_from_residuals_multiple_slots() {
        let mut residuals = Vec::new();
        // Slot (0, 8): tighter residuals
        for i in 0..15 {
            residuals.push((0, 8, -2.0 + f64::from(i) * 0.25));
        }
        // Slot (5, 18): wider residuals
        for i in 0..15 {
            residuals.push((5, 18, -10.0 + f64::from(i) * 1.5));
        }

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        assert_eq!(q.slot_quantiles.len(), 2);

        let slot_a = q.get_quantiles(0, 8);
        let slot_b = q.get_quantiles(5, 18);

        // Slot B should have wider spread than slot A
        let width_a = slot_a.q_high - slot_a.q_low;
        let width_b = slot_b.q_high - slot_b.q_low;
        assert!(
            width_b > width_a,
            "Expected slot B wider ({width_b}) than slot A ({width_a})"
        );
    }

    #[test]
    fn test_get_quantiles_falls_back_to_global() {
        // 5 residuals in slot (2, 14) — below threshold of 10
        // 15 residuals in slot (3, 16) — above threshold
        let mut residuals = Vec::new();
        for i in 0..5 {
            residuals.push((2, 14, f64::from(i)));
        }
        for i in 0..15 {
            residuals.push((3, 16, f64::from(i)));
        }

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        // Slot (2, 14) has only 5 residuals → should fall back to global
        let quantiles_214 = q.get_quantiles(2, 14);
        let global = q.global_quantiles();
        assert_eq!(quantiles_214.count, global.count);
        assert_relative_eq!(quantiles_214.q_low, global.q_low);
        assert_relative_eq!(quantiles_214.q_high, global.q_high);
    }

    #[test]
    fn test_get_quantiles_uses_slot_when_sufficient() {
        let mut residuals = Vec::new();
        // 12 residuals in slot (0, 10) — above threshold
        for i in 0..12 {
            residuals.push((0, 10, -3.0 + f64::from(i) * 0.5));
        }
        // 12 residuals in another slot to give global different values
        for i in 0..12 {
            residuals.push((6, 20, -10.0 + f64::from(i)));
        }

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        let slot = q.get_quantiles(0, 10);
        let global = q.global_quantiles();

        // Slot (0, 10) has 12 residuals — should use slot-specific, not global
        assert_eq!(slot.count, 12);
        assert_ne!(slot.count, global.count);
    }

    // ── compute_confidence_interval tests ─────────────────────────────

    #[test]
    fn test_confidence_interval_basic() {
        let residuals: Vec<(u32, u32, f64)> =
            (0..20).map(|i| (1, 10, -10.0 + f64::from(i))).collect();

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        let (low, high, score) = q.compute_confidence_interval(50.0, 1, 10, 1);

        assert!(low < 50.0, "Low ({low}) should be below predicted (50)");
        assert!(high > 50.0, "High ({high}) should be above predicted (50)");
        assert!(low <= high, "Low ({low}) should be <= high ({high})");
        assert!(
            (0.1..=0.95).contains(&score),
            "Score ({score}) should be in [0.1, 0.95]"
        );
    }

    #[test]
    fn test_confidence_interval_horizon_scaling() {
        let residuals: Vec<(u32, u32, f64)> =
            (0..20).map(|i| (0, 8, -10.0 + f64::from(i))).collect();

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        let (low1, high1, _) = q.compute_confidence_interval(50.0, 0, 8, 1);
        let (low3, high3, _) = q.compute_confidence_interval(50.0, 0, 8, 3);

        let width1 = high1 - low1;
        let width3 = high3 - low3;

        assert!(
            width3 >= width1,
            "3-hour interval ({width3:.2}) should be >= 1-hour ({width1:.2})"
        );
    }

    #[test]
    fn test_confidence_interval_clamps() {
        // Residuals that would push interval outside [0, 100]
        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 8, -60.0 + f64::from(i) * 6.0))
            .collect();

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        // Predicted near 0 — low would go negative
        let (low, high, _) = q.compute_confidence_interval(5.0, 0, 8, 1);
        assert!(low >= 0.0, "Low ({low}) should be >= 0");
        assert!(high <= 100.0, "High ({high}) should be <= 100");
        assert!(low <= high);

        // Predicted near 100 — high would exceed 100
        let (low, high, _) = q.compute_confidence_interval(95.0, 0, 8, 1);
        assert!(low >= 0.0, "Low ({low}) should be >= 0");
        assert!(high <= 100.0, "High ({high}) should be <= 100");
        assert!(low <= high);
    }

    #[test]
    fn test_confidence_score_narrow_interval() {
        // All residuals near zero → narrow interval → high confidence
        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 8, -0.5 + f64::from(i) * 0.05))
            .collect();

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        let (_, _, score) = q.compute_confidence_interval(50.0, 0, 8, 1);
        assert!(
            score > 0.8,
            "Narrow interval should yield high confidence, got {score}"
        );
    }

    #[test]
    fn test_confidence_score_wide_interval() {
        // Large residual spread → wide interval → low confidence
        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 8, -40.0 + f64::from(i) * 4.0))
            .collect();

        let q = ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!());

        let (_, _, score) = q.compute_confidence_interval(50.0, 0, 8, 1);
        assert!(
            score < 0.5,
            "Wide interval should yield low confidence, got {score}"
        );
    }

    // ── Property-based tests ──────────────────────────────────────────

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_quantiles_ordering(
            n in 2_usize..50,
            spread in 1.0_f64..50.0,
        ) {
            let residuals: Vec<(u32, u32, f64)> = (0..n)
                .map(|i| (0_u32, 10_u32, -spread + (2.0 * spread * i as f64 / n as f64)))
                .collect();

            if let Some(q) = ResidualQuantiles::from_residuals(&residuals) {
                let global = q.global_quantiles();
                prop_assert!(
                    global.q_low <= global.q_high,
                    "q_low ({}) should be <= q_high ({})",
                    global.q_low, global.q_high
                );
            }
        }

        #[test]
        fn prop_confidence_interval_ordered(
            predicted in 0.0_f64..=100.0,
            hours_ahead in 1_i64..=6,
            weekday in 0_u32..7,
            hour in 0_u32..24,
            n in 10_usize..30,
            spread in 1.0_f64..30.0,
        ) {
            let residuals: Vec<(u32, u32, f64)> = (0..n)
                .map(|i| (weekday, hour, -spread + (2.0 * spread * i as f64 / n as f64)))
                .collect();

            if let Some(q) = ResidualQuantiles::from_residuals(&residuals) {
                let (low, high, _) = q.compute_confidence_interval(
                    predicted, weekday, hour, hours_ahead,
                );
                prop_assert!(
                    low <= high,
                    "low ({low}) should be <= high ({high})"
                );
            }
        }

        #[test]
        fn prop_confidence_score_in_range(
            predicted in 0.0_f64..=100.0,
            hours_ahead in 1_i64..=6,
            n in 10_usize..30,
            spread in 1.0_f64..30.0,
        ) {
            let residuals: Vec<(u32, u32, f64)> = (0..n)
                .map(|i| (0_u32, 10_u32, -spread + (2.0 * spread * i as f64 / n as f64)))
                .collect();

            if let Some(q) = ResidualQuantiles::from_residuals(&residuals) {
                let (_, _, score) = q.compute_confidence_interval(
                    predicted, 0, 10, hours_ahead,
                );
                prop_assert!(
                    (0.1..=0.95).contains(&score),
                    "Score ({score}) should be in [0.1, 0.95]"
                );
            }
        }

        #[test]
        fn prop_horizon_widens_interval(
            predicted in 10.0_f64..=90.0,
            n in 10_usize..30,
            spread in 2.0_f64..20.0,
        ) {
            let residuals: Vec<(u32, u32, f64)> = (0..n)
                .map(|i| (0_u32, 10_u32, -spread + (2.0 * spread * i as f64 / n as f64)))
                .collect();

            if let Some(q) = ResidualQuantiles::from_residuals(&residuals) {
                let (low1, high1, _) = q.compute_confidence_interval(
                    predicted, 0, 10, 1,
                );
                let (low3, high3, _) = q.compute_confidence_interval(
                    predicted, 0, 10, 3,
                );
                let width1 = high1 - low1;
                let width3 = high3 - low3;
                prop_assert!(
                    width3 >= width1 - 1e-10,
                    "3-hour width ({width3:.4}) should be >= 1-hour width ({width1:.4})"
                );
            }
        }
    }
}
