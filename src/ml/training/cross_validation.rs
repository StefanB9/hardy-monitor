/// A single fold's train/validation index ranges (half-open: `start..end`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    pub train_start: usize,
    pub train_end: usize,
    pub val_start: usize,
    pub val_end: usize,
}

impl Fold {
    /// Number of training samples in this fold.
    pub fn train_len(&self) -> usize {
        self.train_end - self.train_start
    }

    /// Number of validation samples in this fold.
    pub fn val_len(&self) -> usize {
        self.val_end - self.val_start
    }
}

/// Expanding-window time-series cross-validation splitter.
///
/// Produces `k` folds where each fold's training set is all data before a
/// cutoff point, a gap of `gap_samples` is skipped, and the validation set is
/// the next chunk. Each successive fold has a larger training set (expanding
/// window).
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    k: usize,
    gap_samples: usize,
}

impl TimeSeriesSplit {
    /// Create a new splitter.
    ///
    /// Returns `None` if `k < 2` (need at least 2 folds for meaningful
    /// cross-validation).
    pub fn new(k: usize, gap_samples: usize) -> Option<Self> {
        if k < 2 {
            return None;
        }
        Some(Self { k, gap_samples })
    }

    /// Generate fold index ranges for `n_samples` data points.
    ///
    /// Returns `None` if there aren't enough samples to produce all `k` folds
    /// with at least 1 training sample and 1 validation sample each.
    pub fn split(&self, n_samples: usize) -> Option<Vec<Fold>> {
        // Each fold needs: at least 1 train sample + gap + at least 1 val sample.
        // The first fold needs the least training data.
        // val_size is the same for all folds.
        //
        // Layout: the data is divided into (k + 1) roughly equal segments.
        // Segment 0 is the minimum training set for fold 0.
        // Segments 1..=k are the validation sets for folds 0..k-1.
        //
        // Fold i:
        //   train = [0 .. segment_boundary(i+1) - gap)
        //   val   = [segment_boundary(i+1) .. segment_boundary(i+2))

        let segment_count = self.k + 1;

        // We need at least: 1 (min train) + gap + k (one val sample per fold)
        let min_needed = 1 + self.gap_samples + self.k;
        if n_samples < min_needed {
            return None;
        }

        let segment_size = n_samples / segment_count;
        if segment_size == 0 {
            return None;
        }

        let mut folds = Vec::with_capacity(self.k);

        for i in 0..self.k {
            let val_start = (i + 1) * segment_size;
            let val_end = if i + 1 == self.k {
                n_samples // Last fold gets all remaining samples
            } else {
                (i + 2) * segment_size
            };

            // Training ends gap_samples before validation starts
            let train_end = val_start.saturating_sub(self.gap_samples);

            // Need at least 1 training sample and 1 validation sample
            if train_end == 0 || val_start >= val_end {
                return None;
            }

            folds.push(Fold {
                train_start: 0,
                train_end,
                val_start,
                val_end,
            });
        }

        Some(folds)
    }

    /// Number of folds.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Gap size in samples.
    pub fn gap_samples(&self) -> usize {
        self.gap_samples
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_split_basic_4_folds() {
        let splitter = TimeSeriesSplit::new(4, 0);
        assert!(splitter.is_some());

        let folds = splitter.and_then(|s| s.split(1000));
        assert!(folds.is_some());

        let folds = folds.unwrap_or_default();
        assert_eq!(folds.len(), 4);

        // Each fold should have training data
        for fold in &folds {
            assert!(fold.train_len() > 0);
            assert!(fold.val_len() > 0);
        }
    }

    #[test]
    fn test_split_with_gap() {
        let splitter = TimeSeriesSplit::new(4, 50);
        assert!(splitter.is_some());

        let folds = splitter.and_then(|s| s.split(1000));
        assert!(folds.is_some());

        let folds = folds.unwrap_or_default();
        for fold in &folds {
            assert!(
                fold.val_start >= fold.train_end + 50,
                "Gap not respected: val_start={}, train_end={}",
                fold.val_start,
                fold.train_end
            );
        }
    }

    #[test]
    fn test_split_minimum_k() {
        let splitter = TimeSeriesSplit::new(2, 0);
        assert!(splitter.is_some());

        let folds = splitter.and_then(|s| s.split(100));
        assert!(folds.is_some());
        assert_eq!(folds.unwrap_or_default().len(), 2);
    }

    #[test]
    fn test_split_k_less_than_2() {
        assert!(TimeSeriesSplit::new(1, 0).is_none());
        assert!(TimeSeriesSplit::new(0, 0).is_none());
    }

    #[test]
    fn test_split_insufficient_samples() {
        let splitter = TimeSeriesSplit::new(4, 5);
        assert!(splitter.is_some());

        let folds = splitter.and_then(|s| s.split(5));
        assert!(folds.is_none());
    }

    #[test]
    fn test_fold_train_before_val() {
        let folds = TimeSeriesSplit::new(4, 0)
            .and_then(|s| s.split(1000))
            .unwrap_or_default();

        for fold in &folds {
            assert!(
                fold.train_end <= fold.val_start,
                "Train must end before val starts: train_end={}, val_start={}",
                fold.train_end,
                fold.val_start
            );
        }
    }

    #[test]
    fn test_fold_no_overlap() {
        let folds = TimeSeriesSplit::new(4, 0)
            .and_then(|s| s.split(1000))
            .unwrap_or_default();

        for fold in &folds {
            // No index can be in both train and val for the same fold
            assert!(
                fold.train_end <= fold.val_start,
                "Overlap detected: train_end={}, val_start={}",
                fold.train_end,
                fold.val_start
            );
        }
    }

    #[test]
    fn test_fold_expanding_train() {
        let folds = TimeSeriesSplit::new(4, 0)
            .and_then(|s| s.split(1000))
            .unwrap_or_default();

        for i in 1..folds.len() {
            assert!(
                folds[i].train_end >= folds[i - 1].train_end,
                "Training set should expand: fold {} has train_end={}, fold {} has train_end={}",
                i - 1,
                folds[i - 1].train_end,
                i,
                folds[i].train_end
            );
        }
    }

    #[test]
    fn test_fold_gap_respected() {
        let gap = 50;
        let folds = TimeSeriesSplit::new(4, gap)
            .and_then(|s| s.split(1000))
            .unwrap_or_default();

        for fold in &folds {
            let actual_gap = fold.val_start - fold.train_end;
            assert!(
                actual_gap >= gap,
                "Gap not respected: expected >= {gap}, got {actual_gap}"
            );
        }
    }

    #[test]
    fn test_fold_complete_val_coverage() {
        let folds = TimeSeriesSplit::new(4, 0)
            .and_then(|s| s.split(1000))
            .unwrap_or_default();

        // Validation ranges should be contiguous and cover from first val_start to
        // n_samples
        for i in 1..folds.len() {
            assert_eq!(
                folds[i].val_start,
                folds[i - 1].val_end,
                "Validation ranges not contiguous between folds {} and {}",
                i - 1,
                i
            );
        }

        // Last fold should reach n_samples
        assert_eq!(
            folds.last().map(|f| f.val_end),
            Some(1000),
            "Last fold should reach n_samples"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_no_train_val_overlap(
            n in 50_usize..2000,
            k in 2_usize..8,
            gap in 0_usize..20,
        ) {
            let Some(splitter) = TimeSeriesSplit::new(k, gap) else { return Ok(()); };
            let Some(folds) = splitter.split(n) else { return Ok(()); };

            for fold in &folds {
                prop_assert!(
                    fold.train_end <= fold.val_start,
                    "Overlap: train_end={} > val_start={}",
                    fold.train_end,
                    fold.val_start
                );
            }
        }

        #[test]
        fn prop_temporal_ordering(
            n in 50_usize..2000,
            k in 2_usize..8,
            gap in 0_usize..20,
        ) {
            let Some(splitter) = TimeSeriesSplit::new(k, gap) else { return Ok(()); };
            let Some(folds) = splitter.split(n) else { return Ok(()); };

            for fold in &folds {
                prop_assert!(fold.train_start < fold.train_end);
                prop_assert!(fold.val_start < fold.val_end);
                prop_assert!(fold.train_end <= fold.val_start);
            }
        }

        #[test]
        fn prop_expanding_window(
            n in 50_usize..2000,
            k in 2_usize..8,
            gap in 0_usize..20,
        ) {
            let Some(splitter) = TimeSeriesSplit::new(k, gap) else { return Ok(()); };
            let Some(folds) = splitter.split(n) else { return Ok(()); };

            for i in 1..folds.len() {
                prop_assert!(
                    folds[i].train_end >= folds[i - 1].train_end,
                    "Not expanding: fold {} train_end={} < fold {} train_end={}",
                    i - 1,
                    folds[i - 1].train_end,
                    i,
                    folds[i].train_end
                );
            }
        }

        #[test]
        fn prop_gap_respected(
            n in 50_usize..2000,
            k in 2_usize..8,
            gap in 0_usize..20,
        ) {
            let Some(splitter) = TimeSeriesSplit::new(k, gap) else { return Ok(()); };
            let Some(folds) = splitter.split(n) else { return Ok(()); };

            for fold in &folds {
                let actual_gap = fold.val_start - fold.train_end;
                prop_assert!(
                    actual_gap >= gap,
                    "Gap not respected: expected >= {}, got {}",
                    gap,
                    actual_gap
                );
            }
        }

        #[test]
        fn prop_all_val_indices_covered(
            n in 50_usize..2000,
            k in 2_usize..8,
        ) {
            let Some(splitter) = TimeSeriesSplit::new(k, 0) else { return Ok(()); };
            let Some(folds) = splitter.split(n) else { return Ok(()); };

            // Validation ranges should be contiguous (no holes)
            for i in 1..folds.len() {
                prop_assert_eq!(
                    folds[i].val_start,
                    folds[i - 1].val_end,
                    "Hole between fold {} val_end={} and fold {} val_start={}",
                    i - 1,
                    folds[i - 1].val_end,
                    i,
                    folds[i].val_start
                );
            }

            // Last fold reaches end
            if let Some(last) = folds.last() {
                prop_assert_eq!(last.val_end, n);
            }
        }
    }
}
