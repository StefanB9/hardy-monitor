# Performance Optimization Plan

Branch: `refactor/performance-optimizations` from `dev`

## Fixes (execution order)

1. **Fix 5** — DB migration: TEXT → TIMESTAMPTZ (foundational)
2. **Fix 3** — Repair module: reduce DB round-trips + batch ops + eliminate clone
3. **Fix 1** — Heatmap O(n²) → O(1) lookup via 2D array
4. **Fix 6** — Heatmap tooltip cache: conditional clear
5. **Fix 2** — History chart O(n) hover → O(log n) binary search
6. **Fix 4** — Momentum feature extraction: single-pass accumulator
7. **Fix 7** — Easter date memoization

Each fix is an independent commit. See full details in the plan discussion.
