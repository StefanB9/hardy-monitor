# Console logging in release builds

Release builds log only to `logs/hardy-monitor.log` (daily rotation). A
`--console` argument adds console output for watching a run live.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Programs | Daemon and GUI |
| Form | `--console` command-line argument, parsed with std only (no new dependency); an unknown argument is an error that names the usage |
| Destination | Console and file: the file keeps everything, the console mirrors it |

## Design

- `hardy_core::cli::Options::parse(args)` → `Options { console }`, unit-tested.
- Each binary's release `setup_logging(console)` adds an optional console
  `fmt` layer next to the file layer, under the same filter (`RUST_LOG` or
  the default). Debug builds already log to the console; the flag changes
  nothing there.
