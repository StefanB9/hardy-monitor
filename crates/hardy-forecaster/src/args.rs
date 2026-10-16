//! The forecaster's command line: the shared options plus `--backfill`.

use anyhow::Result;
use hardy_core::cli::Options;

const BACKFILL: &str = "--backfill";

/// What the command line asked for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Args {
    pub(crate) options: Options,
    /// Run the history backfill even though history is stored (fills gaps).
    pub(crate) backfill: bool,
}

/// Parses the arguments after the program name.
pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Args> {
    let mut backfill = false;
    let shared: Vec<String> = args
        .into_iter()
        .filter(|arg| {
            let is_backfill = arg == BACKFILL;
            backfill |= is_backfill;
            !is_backfill
        })
        .collect();
    let options = Options::parse("hardy-forecaster [--backfill]", shared)?;
    Ok(Args { options, backfill })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Args> {
        parse(args.iter().map(ToString::to_string))
    }

    #[test]
    fn test_parse_backfill_and_shared_options() -> Result<()> {
        assert_eq!(parse_strs(&[])?, Args::default());
        let both = parse_strs(&["--backfill", "--console"])?;
        assert!(both.backfill);
        assert!(both.options.console);
        assert!(!parse_strs(&["--console"])?.backfill);
        Ok(())
    }

    #[test]
    fn test_parse_rejects_unknown_arguments_with_usage() {
        let error = parse_strs(&["--refit"])
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(error.contains("--refit"), "{error}");
        assert!(
            error.contains("usage: hardy-forecaster [--backfill] [--console]"),
            "{error}"
        );
    }
}
