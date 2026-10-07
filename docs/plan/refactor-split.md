# Refactor: split oversized files, remove dead code

Behaviour-preserving restructuring to meet the module rules in CLAUDE.md
(split at ~300 lines, directory modules at ~500).

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Dead code | **Remove**: old `calculate_predictions*` / `find_best_time_today*`, `midnight_utc`, `TimePeriod`, GUI `CombinedNotifier`, `analytics.prediction_window_days`, `Database::get_history(days)` and unused re-exports |
| Unit tests | Stay **with their submodule** (`#[cfg(test)]` per file) |
| Integration tests | **Split by topic** (database, api, app_logic), sharing `common/` |
| Public API | Root re-exports **trimmed to widely used types**; everything else by module path; callers updated |

## Splits

- **core `analytics/`**: `comparison` (period comparison, trend), `stats`
  (stats, day analysis), `slots` (peak/quiet hours and windows), `insights`,
  `time` (weekday names, local midnight).
- **core `config/`**: `mod` (`AppConfig`, loading, validation) plus one file
  per section group (database/network, ml, notifications, schedule,
  ui: window/refresh/thresholds).
- **core `schedule/`**: `mod` (opening hours) and `holidays` (Bavarian
  holidays, Easter).
- **core `db/`**: readings and averages queries out of `mod.rs`; CSV export
  and repair helpers in their own files.
- **core `repair/`**: per-day steps (boundaries, gaps, smoothing) out of
  `mod.rs`.
- **ml**: `training` and `confidence` only where production code exceeds the
  limit after moving tests next to their code.
- **daemon**: fetch cycle out of `main.rs`.
- **gui**: `app/update.rs` by message group, `app/mod.rs` state types,
  `history_chart` drawing layers, `heatmap` grid vs widget.
- **integration tests**: `database.rs` → readings / averages / alerts /
  models; `api.rs`, `app_logic.rs` by topic.

## Rules

- No behaviour change: only moves, renames of private items, dead-code
  removal and import updates.
- Each step compiles, passes all tests and clippy with zero warnings, and is
  its own commit.
- Test counts drop only by the removed dead code's tests.
- GUI screenshot-checked headless at the end (temporary Linux features, not
  committed).
