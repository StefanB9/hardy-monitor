//! Command-line options shared by the daemon and the GUI.

use anyhow::{Result, bail};

/// What the command line asked for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Also log to the console (release builds log to a file only).
    pub console: bool,
}

impl Options {
    /// Parses the arguments after the program name; `program` names it in
    /// the usage line of an error.
    pub fn parse(program: &str, args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self::default();
        for arg in args {
            match arg.as_str() {
                "--console" => options.console = true,
                other => bail!("unknown argument {other:?}; usage: {program} [--console]"),
            }
        }
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options> {
        Options::parse("hardy-daemon", args.iter().map(ToString::to_string))
    }

    #[test]
    fn test_options_default_without_arguments() -> Result<()> {
        assert_eq!(parse(&[])?, Options { console: false });
        Ok(())
    }

    #[test]
    fn test_options_console_flag() -> Result<()> {
        assert_eq!(parse(&["--console"])?, Options { console: true });
        assert_eq!(
            parse(&["--console", "--console"])?,
            Options { console: true }
        );
        Ok(())
    }

    #[test]
    fn test_options_reject_unknown_arguments_with_usage() {
        let error = parse(&["--verbose"])
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(error.contains("--verbose"), "{error}");
        assert!(error.contains("usage: hardy-daemon [--console]"), "{error}");
    }
}
