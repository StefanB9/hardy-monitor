//! Picking the model shown by default: the best one per horizon bucket.

use super::{Bucket, ModelReport, Settings};

/// The chosen model per horizon bucket.
pub type Selection<L> = Vec<(Bucket, L)>;

/// For each horizon bucket, the eligible model with the lowest MAE there.
/// A tie keeps the `previous` winner, so the choice does not flip between
/// equally good models; a bucket without any eligible model is left out.
pub fn select<L: Copy + PartialEq>(
    reports: &[ModelReport<L>],
    previous: &[(Bucket, L)],
    settings: &Settings,
) -> Selection<L> {
    Bucket::HORIZONS
        .into_iter()
        .filter_map(|bucket| {
            let candidates = reports
                .iter()
                .filter(|r| r.is_eligible(settings))
                .filter_map(|r| r.score(bucket).map(|s| (r.model, s.mae)));
            let best = candidates
                .clone()
                .map(|(_, mae)| mae)
                .min_by(f64::total_cmp)?;
            let kept = previous
                .iter()
                .find(|(b, _)| *b == bucket)
                .map(|(_, model)| *model)
                .filter(|model| {
                    candidates
                        .clone()
                        .any(|(m, mae)| m == *model && (mae - best).abs() <= 1e-9)
                });
            let winner = kept.or_else(|| {
                candidates
                    .clone()
                    .find(|(_, mae)| (mae - best).abs() <= 1e-9)
                    .map(|(m, _)| m)
            })?;
            Some((bucket, winner))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Accumulator, ModelReport};

    fn report(model: &'static str, origins: usize, maes: [f64; 3]) -> ModelReport<&'static str> {
        let mut scores = [None; 4];
        for (bucket, mae) in Bucket::HORIZONS.into_iter().zip(maes) {
            let mut acc = Accumulator::default();
            acc.add(mae, 0.0, None);
            scores[bucket.index()] = acc.score(None);
        }
        ModelReport {
            model,
            origins,
            fitted_days: 1,
            failure: None,
            scores,
            intervals: [None; 4],
        }
    }

    #[test]
    fn test_select_picks_the_best_per_bucket() {
        let reports = [
            report("a", 30, [1.0, 5.0, 9.0]),
            report("b", 30, [2.0, 4.0, 8.0]),
            report("c", 30, [3.0, 6.0, 7.0]),
        ];
        let selection = select(&reports, &[], &Settings::default());
        assert_eq!(
            selection,
            vec![
                (Bucket::NextHour, "a"),
                (Bucket::OneToThreeHours, "b"),
                (Bucket::ThreeToDay, "c"),
            ]
        );
    }

    #[test]
    fn test_select_skips_models_with_too_few_origins() {
        let reports = [report("lucky", 3, [0.0; 3]), report("steady", 30, [2.0; 3])];
        let selection = select(&reports, &[], &Settings::default());
        assert!(selection.iter().all(|(_, m)| *m == "steady"));
        assert_eq!(
            select(&[report("new", 1, [1.0; 3])], &[], &Settings::default()),
            Vec::new()
        );
    }

    #[test]
    fn test_select_keeps_the_previous_winner_on_a_tie() {
        let reports = [report("a", 30, [2.0; 3]), report("b", 30, [2.0; 3])];
        let previous = [(Bucket::NextHour, "b")];
        let selection = select(&reports, &previous, &Settings::default());
        assert_eq!(selection[0], (Bucket::NextHour, "b"));
        assert_eq!(selection[1], (Bucket::OneToThreeHours, "a"));
    }
}
