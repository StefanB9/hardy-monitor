# Windows as ranges

The Insights lists show single hours, so one quiet Saturday evening fills
four of five rows ("Sat 20:00, 19:00, 18:00, 17:00"). Busiest and
quietest times become ranges of consecutive hours.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Method | Take the 15 busiest (quietest) hours with enough data (`MIN_DAYS`; quietest also above 0%), join hours of the same weekday that follow each other into ranges, strongest first (by average) |
| Length | No limit: "Sat 16:00–21:00" says the whole evening is quiet (the 3-hour "Best workout window" and the Plan tab stay as they are) |
| Value | Average and span: "~6% (5–7%)", the range's average and its quietest and busiest hour |
| Insights lists | Busiest times and Quietest times list the 5 strongest ranges |
| Findings | "Busiest times to avoid" quotes the busiest ranges; the quietest day's "Best time" quotes that day's quietest range; the busiest day quotes that day's busiest range instead of its peak hour |
| Week heatmap | Outlines every busy and quiet range (all of them, not only 5), from the heatmap's own period, so they move with the period selector |

## Design

**Core (`analytics::slots`)**
- `SlotRange { weekday, start_hour, end_hour, average, min, max }` (end
  after the last hour).
- `busiest_ranges(data, top_hours)` / `quietest_ranges(data, top_hours)`:
  pick the top hours, join consecutive hours of a weekday, sort strongest
  first. `TOP_HOURS = 15`, `LISTED_RANGES = 5`.
- Insights findings use them; `find_peak_hours` / `find_quiet_hours` go
  if nothing else uses them.

**GUI**
- Insights: the two cards list ranges: "Sat 16:00–21:00" and a badge
  "~6% (5–7%)".
- Week: the heatmap draws an outline (busy / quiet colour) around each
  range's cells, with a small legend.

## PR
One PR: core with unit and property tests (ranges are disjoint, each hour
of a range is among the top hours, consecutive and of one weekday, every
top hour is in exactly one range, min ≤ average ≤ max), findings tests,
GUI label tests, headless screenshots of Insights and Week.
