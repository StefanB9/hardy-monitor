//! Terminal progress: one bar per running stage of a forecast run and a
//! status line between runs. indicatif redraws at most 20 times a second,
//! and draws nothing when stderr is not a terminal.

use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Mutex,
    time::Duration,
};

use hardy_ml::progress::{Progress, Stage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing_subscriber::fmt::MakeWriter;

/// How often the status spinner turns.
const SPIN: Duration = Duration::from_millis(250);

/// The bars and the status line, shared with the log writer.
pub(crate) struct Terminal {
    multi: MultiProgress,
    status: ProgressBar,
    bars: Mutex<HashMap<Stage, ProgressBar>>,
}

impl Terminal {
    /// Draws on `multi` (stderr, or hidden in tests).
    pub(crate) fn new(multi: MultiProgress) -> Self {
        let status = multi.add(ProgressBar::new_spinner());
        status.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        status.enable_steady_tick(SPIN);
        Self {
            multi,
            status,
            bars: Mutex::new(HashMap::new()),
        }
    }

    /// Replaces the status line.
    pub(crate) fn status(&self, message: String) {
        self.status.set_message(message);
    }

    /// A writer that prints log lines above the bars.
    pub(crate) fn log_writer(&self) -> AboveBars {
        AboveBars(self.multi.clone())
    }

    /// Clears the status line when the program ends.
    pub(crate) fn close(&self) {
        self.status.finish_and_clear();
    }

    #[cfg(test)]
    fn bar(&self, stage: Stage) -> Option<ProgressBar> {
        self.bars.lock().ok()?.get(&stage).cloned()
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {prefix:<14} [{bar:30.cyan/blue}] {pos:>4}/{len:4} {elapsed_precise} · ETA {eta}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

impl Progress for Terminal {
    fn start(&self, stage: Stage, total: usize) {
        let bar = self
            .multi
            .add(ProgressBar::new(u64::try_from(total).unwrap_or(u64::MAX)));
        bar.set_style(bar_style());
        bar.set_prefix(stage.label());
        if let Ok(mut bars) = self.bars.lock()
            && let Some(old) = bars.insert(stage, bar)
        {
            old.finish_and_clear();
        }
    }

    fn advance(&self, stage: Stage) {
        if let Ok(bars) = self.bars.lock()
            && let Some(bar) = bars.get(&stage)
        {
            bar.inc(1);
        }
    }

    fn finish(&self, stage: Stage) {
        if let Ok(bars) = self.bars.lock()
            && let Some(bar) = bars.get(&stage)
        {
            // Kept on screen (finished) until the run ends.
            bar.finish();
        }
    }
}

impl Terminal {
    /// Removes the finished bars of a run that is over.
    pub(crate) fn clear_run(&self) {
        if let Ok(mut bars) = self.bars.lock() {
            for (_, bar) in bars.drain() {
                bar.finish_and_clear();
                self.multi.remove(&bar);
            }
        }
    }
}

/// Writes to stderr with the bars hidden for the moment, so log lines do
/// not tear them.
#[derive(Clone)]
pub(crate) struct AboveBars(MultiProgress);

impl Write for AboveBars {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.suspend(|| io::stderr().write(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

impl<'a> MakeWriter<'a> for AboveBars {
    type Writer = AboveBars;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use indicatif::ProgressDrawTarget;

    use super::*;

    fn terminal() -> Terminal {
        Terminal::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()))
    }

    #[test]
    fn test_terminal_tracks_each_stage() -> Result<()> {
        let terminal = terminal();
        terminal.start(Stage::Backtest, 10);
        terminal.start(Stage::Fit, 2);
        for _ in 0..3 {
            terminal.advance(Stage::Backtest);
        }
        terminal.advance(Stage::Fit);
        let backtest = terminal.bar(Stage::Backtest).context("backtest bar")?;
        assert_eq!((backtest.position(), backtest.length()), (3, Some(10)));
        // A stage that is over shows as complete.
        terminal.finish(Stage::Backtest);
        assert!(backtest.is_finished());
        assert_eq!(backtest.position(), 10);
        let fit = terminal.bar(Stage::Fit).context("fit bar")?;
        assert_eq!(fit.position(), 1);
        assert!(!fit.is_finished());
        terminal.clear_run();
        assert!(terminal.bar(Stage::Fit).is_none());
        Ok(())
    }

    #[test]
    fn test_restarting_a_stage_replaces_its_bar() -> Result<()> {
        let terminal = terminal();
        terminal.start(Stage::Fit, 5);
        terminal.advance(Stage::Fit);
        terminal.start(Stage::Fit, 7);
        let fit = terminal.bar(Stage::Fit).context("fit bar")?;
        assert_eq!((fit.position(), fit.length()), (0, Some(7)));
        // Unknown stages are ignored.
        terminal.advance(Stage::SharedFits);
        terminal.finish(Stage::SharedFits);
        terminal.status("waiting".to_string());
        terminal.close();
        Ok(())
    }

    #[test]
    fn test_log_writer_passes_bytes_through() -> Result<()> {
        let terminal = terminal();
        let mut writer = terminal.log_writer().make_writer();
        assert_eq!(writer.write(b"")?, 0);
        writer.flush()?;
        Ok(())
    }
}
