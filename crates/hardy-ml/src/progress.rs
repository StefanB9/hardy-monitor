//! Progress of the long computations (the nightly run, the history
//! backfill), for whoever wants to show it. Called from worker threads.

/// A long stage of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Fits shared by several backtest days (the neural networks).
    SharedFits,
    /// Backtest days, each fitted and forecast.
    Backtest,
    /// The final fits on the newest data.
    Fit,
}

impl Stage {
    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            Stage::SharedFits => "Weekly fits",
            Stage::Backtest => "Backtest days",
            Stage::Fit => "Final fits",
        }
    }
}

/// Receives progress; every method must be cheap.
pub trait Progress: Send + Sync {
    /// `stage` begins with `total` steps.
    fn start(&self, stage: Stage, total: usize);
    /// One step of `stage` is done.
    fn advance(&self, stage: Stage);
    /// `stage` is over.
    fn finish(&self, stage: Stage);
}

/// Reports nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silent;

impl Progress for Silent {
    fn start(&self, _: Stage, _: usize) {}

    fn advance(&self, _: Stage) {}

    fn finish(&self, _: Stage) {}
}

#[cfg(test)]
pub(crate) mod counting {
    use std::{collections::HashMap, sync::Mutex};

    use super::*;

    /// Records each stage's total, steps and whether it finished.
    #[derive(Debug, Default)]
    pub(crate) struct Counting(pub(crate) Mutex<HashMap<Stage, (usize, usize, bool)>>);

    impl Counting {
        pub(crate) fn get(&self, stage: Stage) -> Option<(usize, usize, bool)> {
            self.0.lock().ok().and_then(|m| m.get(&stage).copied())
        }
    }

    impl Progress for Counting {
        fn start(&self, stage: Stage, total: usize) {
            if let Ok(mut m) = self.0.lock() {
                m.insert(stage, (total, 0, false));
            }
        }

        fn advance(&self, stage: Stage) {
            if let Ok(mut m) = self.0.lock() {
                m.entry(stage).or_default().1 += 1;
            }
        }

        fn finish(&self, stage: Stage) {
            if let Ok(mut m) = self.0.lock() {
                m.entry(stage).or_default().2 = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_labels() {
        for stage in [Stage::SharedFits, Stage::Backtest, Stage::Fit] {
            assert_ne!(stage.label(), "");
        }
        Silent.start(Stage::Fit, 3);
        Silent.advance(Stage::Fit);
        Silent.finish(Stage::Fit);
    }
}
