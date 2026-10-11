# Chart period navigation

Arrows on both sides of the occupancy chart step back and forward through
time, a period at a time; ← / → do the same while the Now view is open.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Views | All ranges: Today steps by a day, 7 days by a week, 30 days by 30 days, Custom by its own length (its date inputs follow) |
| Back limit | Not past the first day with data (`Database::first_reading_time`, measured readings only, so repair anchors do not count) |
| Forward limit | The current period; clicking a range button returns to it |
| Keyboard | ← / → while the Now view is open, unless a text field has focus |
| Label | Toolbar shows the period: "Today", "Fri, 9 Oct", "3 – 9 Oct" |
| Empty period | A past period without readings says so instead of showing an empty chart |
| Forecast | Unchanged: only today shows it (forecasts lie in the future) |

## Design

- `time_range.rs` (pure, unit-tested): `ChartRange::window` takes a period
  offset; `shift_custom` moves custom dates by their length;
  `can_step_back` / `can_step_forward`; `period_label`.
- GUI state: `chart_offset` (periods back, reset when a range is chosen),
  `first_reading` (loaded at start and on Refresh).
- `Message::ChartStep(Step)`, from the arrow buttons and the keyboard
  (window event listener, uncaptured key presses only).

## Tests (written first)

Offsets for each range, DST-crossing day steps, custom shifting, step
limits, labels; `first_reading_time` against `TestDatabase` (ignores
boundary anchors).
