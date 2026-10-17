# Reliability per hour

Which hourly averages are a good guide and which are not. On the dev data
96 of 115 hours vary by less than 5 points between days, but Sunday and
Friday evenings vary by 13–26 points: "19%" may as well be 0% or 45%.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Measure | Days the hour was seen on, and how much those days differ: the standard deviation of the days' averages, in points |
| Unpredictable | Spread over 10 points (`UNPREDICTABLE_SPREAD`) |
| Heatmap tooltip | "28% · seen on 3 days · varies ±19 pts" instead of the reading count |
| Heatmap cells | A small white dot in the top-right corner of unpredictable cells; legend "Varies a lot" |
| Insights ranges | A range is flagged when any of its hours is unpredictable: the badge is followed by "varies" |
| Plan tab | The best window adds "· unpredictable" when its expected range is wider than ±10 points (forecast or recent weeks) |

## Design

**Core**
- `HourlyAverage.std_dev`: per hour, each gym-local day is averaged first;
  `std_dev` is the sample standard deviation of those day averages (0 with
  one day). `avg_percentage` stays the average over readings (weighted by
  each day's readings), so the numbers shown today do not change.
  `.sqlx` regenerated.
- `analytics::UNPREDICTABLE_SPREAD = 10.0` and `is_unpredictable(spread)`.
- `SlotRange.max_spread`: the largest `std_dev` of its hours;
  `SlotRange::varies()`.

**GUI**
- Heatmap: `WeekGrid` keeps days and spread per cell; tooltip text; the
  corner dot and its legend entry.
- Insights: "varies" after the badge of a flagged range.
- Plan: "· unpredictable" when (high − low) / 2 of the best window exceeds
  the threshold.

## PR
One PR: database test for the spread (and unchanged averages), unit tests
for the threshold, range flags, tooltip and Plan texts; headless
screenshots of Week, Insights and Plan.
