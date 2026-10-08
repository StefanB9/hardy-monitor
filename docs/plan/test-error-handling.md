# Integration tests without unwrap / expect / panic

CLAUDE.md forbids `.unwrap()`, `.expect()` and `panic!()` in tests, but 14
integration-test files allowed them file-wide (`#![allow(clippy::unwrap_used,
expect_used)]`, plus `clippy::panic` in `tests/common/mod.rs`): 128
`unwrap`, 100 `expect`, 8 `panic!`.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Tests | Every test returns `anyhow::Result<()>`; `unwrap` → `?` (or `.context()?` where the failure would be unclear), `expect("m")` → `.context("m")?` |
| Helpers | `TestDatabase::new()` / `RawTestDatabase::new()` return `anyhow::Result<Self>` |
| Allows | `unwrap_used` / `expect_used` / `panic` allows removed; `float_cmp` allows kept (exact comparisons are intended there) |
| Leftover databases | Unchanged: a failing test still skips `cleanup()`; the `common/mod.rs` docs explain removing orphans. Automatic cleanup may follow separately |
| Behaviour | None: no assertion is weakened or removed |

## Verification

Each commit keeps `clippy --workspace --all-targets` at zero warnings and
all tests passing; one PR.
