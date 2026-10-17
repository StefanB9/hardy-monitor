# Plan your visit

A "Plan" tab answering "I want to go on Thursday for 90 minutes after
work – when is best?". The Now tab's "Next quiet hour" stays as it is.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Placement | New sidebar tab "Plan", between Insights and Model & Data |
| Day | Today and the next six days, each with its real opening hours (holidays use weekend hours) |
| Visit length | 1 h, 1.5 h or 2 h; 1.5 h preselected |
| Time range | Earliest start and latest end, one for workdays and one for weekends; holidays count as weekends. Default: the whole opening hours |
| Remembered choices | Length and both time ranges, in `plan.json` next to the user `config.toml` (per computer, no database change) |
| Expected occupancy | Live forecast where it reaches (next 24 h, its interval as uncertainty); otherwise the same weekday over the last 28 days per quarter hour (spread across those days as uncertainty) |
| Not enough data | A quarter hour seen on fewer than 2 days (`MIN_DAYS`) is unknown: drawn as a gap, never inside a suggested window |
| Holidays | Use the last 28 days of Sundays, with the note "Holiday – based on recent Sundays" |
| Best window | The quietest window of the chosen length (mean of its quarter hours), starting on a quarter hour, fully inside opening hours and the time range; today not before now. Ties: earliest |
| Alternatives | Best window per part of the day, by start: morning before 11:00, midday 11–14, afternoon 14–17, evening from 17:00; a part with no window is left out |
| Day curve | Expected occupancy over the opening hours with the uncertainty band, the best window highlighted, the switch from forecast to recent weeks marked |
| Compare days | The day chips show each day's best expected occupancy |

## Design

**Core (`hardy-core`)**
- `db`: `quarter_hour_averages(since, until)` – per gym-local weekday and
  quarter hour: mean, standard deviation of the per-day means, number of
  distinct gym-local days. `.sqlx` regenerated.
- `analytics::visit`: no ML types (dependency boundary). Input: one
  `SlotEstimate { start, expected: Option<f64>, low, high, source }` per
  quarter hour of the day (the GUI fills it from forecast and recent
  weeks). `best_windows(estimates, length, earliest, latest, not_before)`
  returns the best window and the best per part of the day. A window's
  expectation is the mean of its quarter hours, its range the mean of their
  lows and highs, its source "forecast" if any quarter hour is forecast.
- `recent_weeks_estimates(day, schedule, averages)`: the day's quarter
  hours from the averages, Sunday's for holidays.

**GUI (`hardy-gui`)**
- `ViewMode::Plan`, `views/plan/` (controls, day chips, best window and
  alternatives, day curve canvas).
- `plan_settings.rs`: load/save `plan.json` (missing or unreadable file →
  defaults, logged).
- Loads the averages with the other analytics data, refreshed with them;
  estimates are rebuilt when the forecast changes.

## PRs
1. Core: query, estimates, window search; unit and property tests
   (windows stay inside opening hours and the range, have the requested
   length, contain no unknown quarter hour, and none is quieter).
2. GUI: Plan tab, settings file; headless screenshots.
