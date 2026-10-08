# GUI load handling

Three problems with how the GUI applies loaded data:

- **Out-of-date results.** Chart history and the week's averages reload on
  range changes, new readings, Refresh and repair. Their results carry no
  hint of which range they were for, so a slow load for an old range can
  overwrite a newer one.
- **"Updating…" ends too early.** One flag, set by the latest-reading fetch,
  Refresh and repair, is cleared by the fetch result alone, while the
  chart and averages may still be loading.
- **One error slot for everything.** A chart failure is hidden by the next
  successful fetch although the chart is still broken; while the gym is
  closed nothing fetches, so an old error stays until Refresh.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Stale results | `LatestOnly` hands out a request number per chart-history and per averages load; the result carries it and is applied only if it is the newest issued |
| "Updating…" | Shown while **any** data load is pending (count, not flag), after the existing 200 ms delay |
| Errors | One error per source; a success clears only that source; the header shows the most recently raised active error. No dismiss button |
| Refresh | Still clears all errors before reloading everything |

## Design

New `app/loads.rs` with pure, unit-tested types:

- `LatestOnly` / `RequestId` — `issue()`, `is_current(id)`.
- `PendingLoads` — `begin(now)`, `finish()`, `is_visible(now)`; `Instant`
  passed in for tests.
- `Errors` keyed by `ErrorSource` (one per load: latest reading, chart,
  week, insights, accuracy, forecast history, model status, model, alert
  settings, export, plus chart-range and repair-date input) — `raise`,
  `clear`, `clear_all`, `latest`.

Pairing loads with results: every database load goes through one helper
that counts it as pending; `Message::finishes_load()` (an exhaustive
match) marks the result messages, and `update` finishes one pending load
for each before dispatching.

## Tests (written first)

- `LatestOnly`: only the newest id is current; results arriving out of
  order are rejected.
- `PendingLoads`: overlapping loads keep it pending until the last
  finishes; hidden before the delay; `finish` never underflows.
- `Errors`: a chart error survives a latest-reading success, a later chart
  success clears it; the newest raised error is shown; re-raising moves a
  source to the front.
- `Message::finishes_load` covers exactly the result messages.

## Verification

fmt, `clippy --workspace --all-targets`, `check --workspace --all-targets`,
`nextest --workspace`, headless screenshots of the four views.
