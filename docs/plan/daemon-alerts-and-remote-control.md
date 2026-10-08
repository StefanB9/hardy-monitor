# Daemon Alerts with GUI and Phone Control

## Problem

Low-occupancy alerts are computed only inside the GUI (`app.rs`), so phone
pushes (ntfy) stop whenever the desktop app is closed. The GUI's threshold
slider and on/off toggle live in memory and are lost on restart. Alerts are
either "on" indefinitely or off, which in practice means 24/7 pushes, and the
first reading after opening (≈0 %) always fires.

## Goal

The daemon sends phone alerts whenever they are armed. Arming, threshold and
duration are controlled from the GUI **or** from the phone, and an armed
alert switches itself off again.

## Design

### Shared settings in the database

New reversible migration `alert_settings` — a single-row table both binaries
read and write:

| Column | Meaning |
|---|---|
| `enabled BOOLEAN` | armed or not |
| `threshold_percent DOUBLE PRECISION` | `CHECK 0..=100` |
| `active_until TIMESTAMPTZ NULL` | auto-off instant; `NULL` = no expiry |
| `updated_at TIMESTAMPTZ`, `updated_by TEXT` | `gui` / `phone` / `migration`, shown in the GUI |

Seeded with `enabled = false`, `threshold_percent = 30`. A `CHECK (id = 1)`
primary key keeps it to one row.

### Arming durations

Arming always picks a duration (`AlertDuration` enum):

- **Until closing** (default) — `active_until` = today's closing time in the
  gym timezone.
- **2 hours** — now + 2 h.
- **Always** — no expiry.

Expiry is evaluated at read time (`enabled && now < active_until`), so nothing
has to "turn off" a row.

### Alert engine (hardy-core, pure)

`alert::AlertEngine::observe(reading, now, &settings, &schedule) -> Option<Alert>`,
fully clock-driven (no I/O) and property-tested. An alert fires when all
hold:

1. Settings are armed and not expired.
2. The gym is open and past the **opening grace period**
   (`opening_grace_minutes`, default 60) — kills the daily "0 % at opening"
   push.
3. Inside an optional **alert window** from config (gym-local, e.g.
   weekdays 16:00–21:00). No windows configured = any opening hour.
4. Occupancy crossed from ≥ threshold to < threshold (edge-triggered), or
   alerts were **just armed** while it is already below (you get an
   immediate "it's quiet now").
5. Cooldown (`cooldown_secs`) since the last alert has elapsed.

### ntfy client (hardy-core)

`ntfy::NtfyClient` (reqwest): `publish(topic, title, body)` and
`poll(topic, since) -> Vec<NtfyMessage>` via
`GET {server}/{topic}/json?poll=1&since=<id|all>`. Optional access token sent
as `Authorization: Bearer …`, read from `HARDY__NOTIFICATIONS__NTFY_TOKEN`
(environment only; documented as never belonging in `config.toml`).
`NtfyClient` implements `Notifier`.

### Phone control

Optional `notifications.control_topic`. Each daemon tick polls it (since the
last seen message id) and parses commands (pure parser, proptested):

| Message | Effect |
|---|---|
| `on` | arm until closing at current threshold |
| `on 25` | arm until closing at 25 % |
| `on 25 2h` / `on always` / `on 25 always` | choose duration |
| `off` | disarm |
| `status` | reply with current state |

Every accepted command is confirmed on the alert topic ("Alerts on below 25 %
until 21:00"); unknown text gets a short usage reply. The first poll after
start uses `since=<now>` so old commands are never replayed.

### Daemon

Per tick, after storing the reading: poll control topic → apply commands to
`alert_settings` → load settings → `AlertEngine::observe` → publish. Alert and
control failures are logged and never block data collection. Engine state is
in memory; after a restart at most one extra alert can occur (cooldown
resets).

### GUI

- Dashboard notification card: toggle + duration picker (until closing /
  2 h / always), threshold slider (saved on release), status line
  ("On below 25 % until 21:00 · set from phone"). Settings are reloaded on
  every data refresh, so phone changes appear within a minute.
- The GUI no longer sends ntfy pushes (the daemon does). It keeps desktop
  popups while it runs, driven by the same settings and engine.
- `CombinedNotifier` is replaced by the desktop-only `SystemNotifier`.

### Config

`[notifications]`: `ntfy_topic`, `ntfy_server`, `cooldown_secs` stay;
new `control_topic`, `opening_grace_minutes` (60), `windows` (list of
`{ days = "weekdays"|"weekends"|"daily", start = "16:00", end = "21:00" }`).
`enabled` and `threshold_percent` move to the database; the config values
are ignored after the migration seeds the row (documented).

## Defaults to confirm

- The committed `config.toml` keeps the current topic name so the running
  Docker daemon and your phone subscription keep working; using a random
  topic name plus an access token is recommended in the docs.
- Opening grace: 60 minutes. Default duration: until closing.

## Tests

- Engine: armed/expiry, window and grace boundaries in gym time (DST days),
  edge trigger, "just armed while below", cooldown — unit + proptest (1000
  cases, e.g. never more than one alert per cooldown, never outside windows).
- Command parser: valid forms, invalid input, proptest round-trip.
- ntfy client: wiremock for publish, auth header, poll parsing, error status.
- DB: settings round-trip, single-row constraint, threshold check
  (`TestDatabase`).
- Migration: seeds the default row; revert drops the table.
