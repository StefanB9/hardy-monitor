# ML Predictions View Redesign

## Context

The ML predictions UI has several UX problems:
1. **Status confusion** — shows "Training..." while old predictions are visible in chart and table below, contradicting itself
2. **Useless table** — the 7-column row-by-row prediction table doesn't add value over the chart
3. **Model details too prominent** — hyperparameters and CV scores always visible, overwhelming for daily use
4. **Chart lacks context** — predictions shown in isolation with no actual occupancy history for comparison

## Decisions (User-Approved)

1. **4-state status system** — distinguish "collecting data", "training initial model", "active", and "retraining (showing previous predictions)"
2. **Prediction Highlights card** replaces the table — next hour, peak, quietest, avg confidence
3. **Model details collapsed by default** — compact R² badge in status line, toggle to expand full details
4. **Chart overlays history** — pass actual `OccupancyLog` history to HistoryChart (already supports it)

## New Layout

```
┌─ ML Prediction Model ────────────────────────────────────┐
│ Status: ● Active (Random Forest)  R² 0.87   Trained 10:30│
│                                                           │
│ [optional: "Retraining... · Showing previous predictions"]│
│                                                           │
│ ▼ Show model details                                      │
│ ┌─ (expanded, hidden by default) ──────────────────────┐  │
│ │ Algorithm: Random Forest    Samples: 2,456           │  │
│ │ Trees: 150  Depth: 12  Min Leaf: 3  Features: 8     │  │
│ │ RMSE: 4.21 ± 0.35    R²: 0.87 ± 0.03               │  │
│ │ MAE: 3.12 ± 0.28                                    │  │
│ └──────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘

┌─ Occupancy — Actual vs Predicted ────────────────────────┐
│ [Blue line: actual history] → [Cyan dashed: predictions]  │
│ [Cyan band: confidence interval]                          │
│ Height: 280px                                             │
└───────────────────────────────────────────────────────────┘

┌─ Prediction Highlights ──────────────────────────────────┐
│  Next Hour          Peak Today                            │
│  45.2%              78.1%                                 │
│  ±5.3pp  ● High     at 17:00                             │
│                                                           │
│  Quietest           Avg Confidence                        │
│  23.4%              ●●● 82%                               │
│  at 14:00           (12 predictions)                      │
└───────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 0: Save plan to project + create feature branch

Save this plan to `docs/plan/ml-predictions-view-redesign.md` and create branch:
```bash
git checkout dev && git pull origin dev
git checkout -b feature/ml-predictions-view-redesign
```

---

### Step 1: Add `PredictionHighlights` extraction logic + tests (TDD)

**File:** `src/views/ml_predictions.rs`

**1a.** Define internal structs:
```rust
struct PredictionHighlights {
    next_hour: Option<HighlightEntry>,
    peak: Option<HighlightEntry>,
    quietest: Option<HighlightEntry>,
    avg_confidence: f64,
    prediction_count: usize,
}

struct HighlightEntry {
    time: DateTime<Utc>,
    value: f64,
    confidence_low: f64,
    confidence_high: f64,
    confidence_score: f64,
}
```

**1b.** Write `extract_highlights(predictions, now) -> Option<PredictionHighlights>`:
- Return `None` if empty
- **next_hour**: prediction closest to `now + 1h` (use `min_by_key` on absolute second difference)
- **peak**: highest `predicted_value` (use `max_by` with `f64::total_cmp`)
- **quietest**: lowest `predicted_value` (use `min_by` with `f64::total_cmp`)
- **avg_confidence**: mean of `confidence_score` across all predictions
- **prediction_count**: `predictions.len()`

**1c.** Tests (write first, TDD red):
- `test_extract_highlights_empty_returns_none`
- `test_extract_highlights_single_prediction` — single prediction is next_hour, peak, and quietest
- `test_extract_highlights_finds_peak_and_quietest` — multiple predictions, verify correct max/min
- `test_extract_highlights_next_hour_selection` — picks prediction closest to now + 1h
- `test_extract_highlights_avg_confidence` — verify average calculation

---

### Step 2: Add `ModelDetailsToggled` message and `show_model_details` state

**File:** `src/app.rs`

**2a.** Add `ModelDetailsToggled(bool)` to `Message` enum (after `PredictionModeToggled` at line 187).

**2b.** Add `show_model_details: bool` to `UiState` (after `show_ml_prediction` at line 112). Init to `false` (line 280).

**2c.** Handle in `update()` (after `PredictionModeToggled` handler at line 627-630):
```rust
Message::ModelDetailsToggled(expanded) => {
    self.ui.show_model_details = expanded;
    Task::none()
}
```

---

### Step 3: Update `MLPredictionsProps` and props construction

**File:** `src/views/ml_predictions.rs`, `src/app.rs`

**3a.** Add two fields to `MLPredictionsProps`:
```rust
pub history: &'a [OccupancyLog],
pub show_model_details: bool,
```
Both are `Copy`, so the existing `#[derive(Clone, Copy)]` still works.

**3b.** Add `use hardy_monitor::db::OccupancyLog;` to imports.

**3c.** Update props construction in `app.rs` (line ~738):
```rust
history: &self.data.history,
show_model_details: self.ui.show_model_details,
```

---

### Step 4: Rewrite the `view()` function

**File:** `src/views/ml_predictions.rs`

#### Status card — 4-state logic (replacing lines 28-61)

| State | Condition | Display |
|-------|-----------|---------|
| 1 | `!has_model && !training` | "Collecting data" (`TEXT_MUTED`) |
| 2 | `!has_model && training` | "Training initial model..." (`ACCENT_ORANGE`) |
| 3 | `has_model && !training` | "Active ({algo})" (`ACCENT_GREEN`) + R² badge + "Last trained: {time}" |
| 4 | `has_model && training` | "Active ({algo})" (`ACCENT_GREEN`) + second row: "Retraining..." (`ACCENT_ORANGE`) + "Showing previous predictions" (`TEXT_MUTED`) |

R² badge: if `training_info.cv_scores` exists, show `R² {value}` color-coded (green >= 0.8, orange >= 0.5, red < 0.5).

#### Model details toggle (within status card)

Only when `training_info` is `Some`:
- Text button: "Show model details ▼" / "Hide model details ▲"
- Sends `Message::ModelDetailsToggled(!props.show_model_details)`
- When expanded: push existing `build_details_card` content **inline** (not wrapped in its own `card_container`). Rename `build_details_card` → `build_details_content` returning `Column<'_, Message>`.

#### Chart card — enhanced (replacing lines 66-106)

- Title: "Occupancy — Actual vs Predicted"
- Height: 280px (was 220px)
- Pass `history: props.history` (was `&[]`)
- Time range: `max(now - 6h, earliest_history)` to `last_prediction + 30min`
- `HistoryChart` already filters data to `range_start..range_end` internally, renders history as blue line and predictions as cyan dashed — no widget changes needed

#### Prediction Highlights card (replacing table, lines 108-214)

- Title: "Prediction Highlights"
- Call `extract_highlights(props.ml_predictions, props.now)`
- If `None`: show "No predictions available" in `TEXT_MUTED`
- If `Some`: 2x2 grid layout using `row![ column![...].width(FillPortion(1)), column![...].width(FillPortion(1)) ]`
  - **Next Hour**: value (size 20, `TEXT_BRIGHT`), ± range + confidence dot (size 11, `TEXT_MUTED`)
  - **Peak Today**: value (size 20, `TEXT_BRIGHT`), "at HH:MM" (size 11, `TEXT_MUTED`)
  - **Quietest**: value (size 20, `TEXT_BRIGHT`), "at HH:MM" (size 11, `TEXT_MUTED`)
  - **Avg Confidence**: percentage (size 20, color-coded), "(N predictions)" (size 11, `TEXT_MUTED`)

---

### Step 5: Clear ML chart cache on history update

**File:** `src/app.rs`

Add `self.ui.ml_predictions_chart_cache.clear()` in `HistoryLoaded` handler (line 388), since the ML chart now shows history data too.

---

### Step 6: Verify

```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo nextest run
cargo nextest run --no-default-features
```

## Files Modified

| File | Change |
|------|--------|
| `src/views/ml_predictions.rs` | Full rewrite: 4-state status, collapsible details, enhanced chart, highlights card, `PredictionHighlights` + tests |
| `src/app.rs` | `ModelDetailsToggled` message, `show_model_details` state, pass `history` to props, cache invalidation |

No new files. No changes to `history_chart.rs` (already supports history + predictions overlay).

## Reusable Code

- `HistoryChart` widget — already renders history (blue) + predictions (cyan) + confidence band; just needs data passed
- `card_container()` — existing card styling (`views/components/helpers.rs`)
- `build_details_card()` content — refactor to `build_details_content()` returning `Column`, reuse inline
- `PredictionWithConfidence` API — `interval_width()`, `to_simple()`, `confidence_score`
- Checkbox toggle pattern — `PredictionModeToggled` in dashboard (line 627) as reference for `ModelDetailsToggled`
