# Daemon upkeep robustness

Nothing bounds how long the daemon's nightly upkeep may take: the pool's
acquire timeout limits waiting for a connection, not a running query. A
hanging database stalls the fetch loop (and shutdown, since repair is not
raced against it). A failing day in a multi-day catch-up also loses the
progress of the whole range. Separately, the database URL (with password)
is printed by `Debug` on `DatabaseConfig` / `AppConfig`.

Measured on 57 days of local data: repairing 30 days takes 0.9 s, so the
fetch for that minute is only delayed; the problem is the unbounded case.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| DB URL | `DatabaseConfig.url` becomes `SecretString` (moved to `config/secret.rs`, still exported as `config::SecretString`) |
| Statement timeout | `database.statement_timeout_secs`, default **60**, > 0, applied by PostgreSQL to every query; shared by daemon and GUI |
| Repair budget | Nightly repair gets **30 s**; running out counts as a failure, so the usual 30-minute retry delay applies |
| Shutdown | Daemon stops waiting for upkeep when shutdown is requested |
| Progress | Nightly catch-up repairs days **one at a time**, oldest first, saving `repaired_through` after each; a failing day stops the run. The GUI's manual repair keeps 4 days at a time |
| Doc fix | Remove the stray `\n` in the `repair_date_range` doc comment |

## Cancellation safety

Each repair step commits on its own and repairing a day again is harmless,
so dropping the repair future mid-day (budget or shutdown) leaves at worst a
partially repaired day that the next run repairs again.

## Tests (written first)

- `Debug` of `DatabaseConfig` / `AppConfig` contains no password.
- Config: `statement_timeout_secs` default 60; 0 rejected by validation.
- With a 1 s statement timeout, `SELECT pg_sleep(2)` fails (`TestDatabase`).
- Zero repair budget: the run reports a timeout, records a failure, and no
  repair is due again within the retry delay.
- Per-day loop with a fake day step failing on day 2 of 3: day 1 is recorded
  as repaired, the run returns the error.

## Verification

fmt, `clippy --workspace --all-targets`, `check --workspace --all-targets`,
`nextest --workspace`, `.sqlx` regenerated, short daemon run against the
local database.
