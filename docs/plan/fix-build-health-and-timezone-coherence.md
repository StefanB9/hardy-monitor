# Fix: Build Health and Timezone-Coherent Data

## Problem

### Build health (broken on `main`)

1. `hardy-core` test target does not compile: the `cargo upgrade` to `toml` 1.x
   moved `toml::from_str` behind the `serde` feature, which the dev-dependency
   does not enable (`config.rs` tests).
2. `cargo fmt --all -- --check` fails (3 comments not wrapped per
   `wrap_comments`).
3. `rust-toolchain.toml` names a floating `nightly` with no components, so
   `clippy`/`rustfmt`/`rust-src` (required by `build-std`) are not guaranteed
   to exist.
4. `Cargo.lock` is git-ignored, but the `Dockerfile` copies it — a clean clone
   cannot build the image, and builds are not reproducible.

### Data coherence depends on the host's timezone

Every "local" computation uses `chrono::Local`, i.e. the timezone of whatever
machine runs the binary. The gym is in Bavaria, so the correct local time is
always `Europe/Berlin`, regardless of host.

| Location | Effect when host TZ ≠ Europe/Berlin (e.g. Docker = UTC) |
|---|---|
| `hardy-daemon` schedule check | Records 23:00–01:00 and skips 06:00–08:00 in summer |
| `repair` | Day boundaries, open/close boundary rows and gap fill written at wrong instants — **mutates the DB incorrectly** |
| `db::get_records_for_date` | Day window is the host's day |
| `ml` features (hour, weekday, holiday) | Model features differ per host |

Independent of host, the weekly aggregation is also wrong:
`get_averages_range` buckets with `EXTRACT(HOUR FROM timestamp)`, which uses the
Postgres session timezone (UTC on Neon). The GUI then shifts buckets by the
*current* UTC offset (`heatmap.rs`, `find_best_time_today`), so any range
spanning a DST change mixes hours that are one hour apart. ML
`historical_stats` are keyed by local (weekday, hour) but filled from these UTC
buckets, and `fallback_predict`/`calculate_predictions` look up UTC slots.

Stored timestamps themselves are `TIMESTAMPTZ` (absolute instants), so existing
rows are coherent; only interpretation and schedule-driven writes are affected.

## Decisions

- **Explicit gym timezone:** new `schedule.timezone` config key (IANA name,
  default `Europe/Berlin`, overridable via `HARDY__SCHEDULE__TIMEZONE`). It
  lives in `[schedule]` because opening hours and Bavarian holidays are defined
  in that zone.
- **New dependency `chrono-tz`** (`default-features = false`,
  `features = ["serde"]`) in `hardy-core`. Problem: need a fixed IANA zone with
  DST rules independent of host tzdata (Docker `debian-slim` has none).
  Alternatives: `Local` (host-dependent — the bug), `FixedOffset` (no DST),
  `jiff` (would replace chrono throughout). `chrono-tz` embeds the tz database
  at compile time and integrates with the existing chrono types.
- **`GymSchedule` owns the timezone.** `is_open` takes `DateTime<Utc>` and
  converts internally; `timezone()` exposes the `Tz` for other callers.
- **`Clock::now_local` is removed** — it is inherently host-dependent. Callers
  use `now_utc()` + the gym timezone.
- **Bucketing happens in SQL** with `timestamp AT TIME ZONE $tz`, so
  `HourlyAverage.weekday/hour` are gym-local. All UTC-offset shifting in the
  GUI and analytics is removed.
- **GUI displays gym-local time** (not viewer-local), so every view agrees with
  the schedule and the stored aggregates.
- Pin the toolchain to `nightly-2026-10-06` with components
  `clippy`, `rustfmt`, `rust-src`. Commit `Cargo.lock`.

## Steps

1. Build health: `toml` dev-dependency gets `serde`; `cargo fmt --all`;
   toolchain pin + components; un-ignore and commit `Cargo.lock`.
2. Config: `ScheduleConfig.timezone: Tz` with default and tests (default,
   parse, invalid name rejected).
3. Schedule: store `Tz`; `is_open(&DateTime<Utc>)`; tests at DST boundaries
   with UTC input.
4. Clock: drop `now_local`.
5. DB: `get_averages_range(start, end, tz)` and
   `get_records_for_date(date, tz)`; integration tests that bucket correctly
   across a DST change; regenerate `.sqlx`.
6. Analytics: drop offset shifting, use gym-local slots for predictions and
   best time; `midnight_local_as_utc` takes `Tz`.
7. Repair: use gym timezone.
8. Daemon: schedule check from `Utc::now()`.
9. GUI: pass `Tz` to views/widgets for formatting and heatmap; ML feature
   extraction and fallback use gym-local slots.
10. Verify the full suite under `TZ=UTC` and `TZ=America/New_York` to prove
    host independence.

## Follow-ups (out of scope)

- Persisted ML models trained before this change used mismatched
  historical-stat features and should be retrained.
- Rows recorded outside opening hours by a UTC-hosted daemon are removed by
  running Data Repair over the affected range after this fix.
- Unique timestamp constraint, percentage validation, repair provenance
  column, daemon retry/shutdown — tracked separately.
