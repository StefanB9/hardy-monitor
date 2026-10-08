# A better non-ML baseline

The baseline (GUI forecast without a model, the yardstick of the training
quality gate, and the `baseline` column of the forecast log) was the plain
mean of every reading per weekday × hour. Measured with the quality gate's
own setup on local data (last 7 days and three rolling weeks), a
**15-minute, recency-weighted profile plus a "busier than usual"
correction** cut its error by about 35 % (3.11 → 2.05 pts; 3.32 → 2.16).
Requested by the user directly.

## Decisions

| Topic | Decision |
|---|---|
| Profile | Weekday × 15-min slots (672); weighted mean and spread, weight `0.5^(age / 14 days)` relative to the newest reading. Falls back to the weekday × hour slot, then the overall mean |
| Correction | `typical(target) + carry[h] · (mean of the last 15 min − typical(now))`; `carry[h]` per whole-hour horizon, least squares over the fitting history, clamped to `[0, 1]`; no correction without a reading in the last 15 min |
| ML model | Unchanged: its features keep the hourly `SlotProfile`, so stored models stay valid (no feature or format version bump). Its fallback without a current reading keeps the stored 56-day profile, which knows more than any short GUI history |
| Quality gate | Baseline fitted on data before the cutoff; current state read from readings at or before each anchor (no leakage) |
| History | Baseline needs `BASELINE_HISTORY_DAYS = 28`; GUI forecasting and the daemon's hourly forecast log load that much (was 8) |
| Wording | GUI "averages" → "baseline" where it means this forecast |

## Tests (written first)

Quarter-hour resolution, recency weighting, carry-over of a persistent
anomaly, no correction without a current reading, carry clamped to `[0, 1]`;
proptest (1000 cases): forecasts within 0–100 for arbitrary histories.
