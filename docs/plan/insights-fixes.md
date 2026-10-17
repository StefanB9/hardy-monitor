# Insights fixes

Bugs found in the review of the Insights tab. Feature work (plan your
visit, windows as ranges, reliability per hour, …) comes later, in its own
plans.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Trend | The average change in percentage points over the hours with enough data in both periods: busier above +1 point, quieter below −1, else stable. The text quotes that number, with the relative change of the overall average in brackets |
| Period | The last 28 days up to now (every weekday four times), compared with the 28 days before |

## Fixes

| # | Bug | Fix |
|---|---|---|
| 1 | The trend averages per-hour *relative* changes, so near-empty hours (2% → 3% = +50%) decide it | Points, as above |
| 2 | The trend's label and its text come from different formulas and can disagree in sign | Both use the same number |
| 3 | A quiet window running until closing ends at "24:00" | It ends after its last hour |
| 4 | "Enough data" counts minute readings (every hour has 100+), so one day passes | `HourlyAverage` gains `days` (distinct gym-local days with readings); "enough" means at least 2 days, everywhere |
| 5 | The busiest day is picked before the data check, so it can vanish | Filter, then pick (as for the quietest day) |
| 6 | Quiet windows join hours across gaps | A window only spans consecutive hours |
| 7 | "Last 4 weeks" is 3 weeks plus the current partial week | Rolling 28 days, as above |
| 8 | % and points mixed; "9:00" and "09:00" | Changes in points (relative change in brackets where useful); hours always `HH:00` |

Biggest changes are ranked and shown in points ("Sun 10:00: 21% → 33%,
+12 points"); only hours with enough days and at least 10% before count,
as today.
