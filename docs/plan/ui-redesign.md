# UI Redesign

## Findings (from screenshots of the current GUI)

- **Global:** labels at 10–12 px; inconsistent font sizes; black secondary
  buttons; no shared colour scale (gauge, heatmap, badges and legend each
  use their own); cards stretch with large empty areas.
- **Dashboard:** "Best time on Tuesdays 06:00" is opening time (always
  empty) and shown after it has passed; alert controls cramped under the
  gauge; chart spans 00:00–24:00 although the gym is closed half of it, has
  no legend, and the forecast is a lone dotted segment; the toolbar mixes
  ranges, dates, Export CSV and a tiny "ML" checkbox.
- **Heatmap:** days without data render as near-black cells (read as "full");
  opening hours hard-coded; no values on cells; unlabeled "Mon 23:00 /
  Tue 06:00" footer; legend colours don't match the gradient; default
  "This Week" is mostly empty.
- **Insights:** "Daily Patterns" is seven near-identical squares; quietest
  times "0% Sat 21:00 / Tue 23:00" and "Mon 23:00 +590%" are artefacts of
  Data Repair's 0 % boundary rows; insight cards ragged with unexplained
  numbered badges; ML card duplicates the Predictions view.
- **Predictions:** with one forecast left, next/peak/quietest show the same
  value.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Structure | Four views: **Now**, **Week**, **Insights**, **Model & Data** |
| Best-time card | **Next quiet window** from now (today, else tomorrow) |
| Chart default | **Opening hours only** for today; 7/30 days keep full range |

## Design system (`style` + `views/components`)

- Colour tokens: app/sidebar/card/elevated backgrounds, border, three text
  levels, one accent. One **occupancy scale** (quiet → moderate → busy →
  packed) used by gauge, heatmap, badges, legends and the sidebar status; a
  distinct **no data** treatment.
- Type scale: display 44, title 24, heading 16, body 14, caption 12 —
  nothing smaller than 12.
- Spacing scale 4/8/12/16/24/32; cards share padding and radius.
- Components: `card(title, content)` with optional header actions,
  `stat_tile`, `segmented` control, `badge`, primary/secondary/ghost button
  styles, `empty_state`.

## Views

**Now** — top row: *live status* (large gauge on the occupancy scale,
open/closed with next opening/closing time, change vs. 15 min ago), *next
quiet window* (time range, expected %, forecast vs. averages source, list of
the next hours), *alerts* (toggle, duration segmented control, threshold,
status). Below: occupancy chart with segmented Today/7 days/30 days and a
legend (actual, forecast, 80 % range); Today shows opening → closing with
the forecast band.

**Week** — segmented range (default *Last 4 weeks*); heatmap using the
configured schedule, values on cells, closed hours dimmed, no-data cells
distinct; gradient legend matching the cell colours; daily-pattern bar chart
with values; quietest/busiest slot summary.

**Insights** — stat tiles (trend, average, range, consistency); busiest and
quietest lists with occupancy badges; full-width key insights with category
colour and icon instead of numbered badges. The duplicate ML card is
removed.

**Model & Data** — model card (status, improvement over averages, trained
when, error by hours ahead, retrain, last error), data repair, CSV export.

**Shell** — sidebar with icons and a live occupancy summary at the bottom;
header with page title, gym-local date and update status.

## Data fixes (bugs surfaced by the screenshots)

- Hourly averages exclude Data Repair `boundary` rows (artificial 0 % at
  opening/closing) — fixes the 0 % quietest slots and the +590 % insight.
- "Significant increase/decrease" insights require a meaningful baseline
  level so tiny values don't produce huge percentages.

## Quiet-window rule

Candidate one-hour windows on a 15-minute grid within opening hours, from now
until the end of today; if less than an hour of opening remains, tomorrow
from one hour after opening (same grace as alerts). Expected occupancy from
the forecast where it covers the window, otherwise the slot averages.
Lowest expected value wins; ties go to the earliest.

## Verification

Pure logic (quiet window, occupancy scale, chart range) is unit- and
property-tested. Each view is rendered headless (Xvfb + iced's tiny-skia
renderer, temporary Linux features not committed) against eight weeks of
synthetic data and inspected via screenshots before committing.
