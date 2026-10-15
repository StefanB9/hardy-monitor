# Forecast models GUI pass

After phase 7 the suite has 51 models (46 forecasters, 5 ensembles). The
Model & Data leaderboard and the chart's model lists were flat and in a
fixed order; this pass makes them navigable. GUI only — no database or ML
changes.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Leaderboard sorting | Clickable column headers (Model, ≤ 1 h, 1–3 h, 3–24 h, Overall, In band); clicking the active column reverses it; an arrow marks it. Default: overall error, best first. Rows without a value for the sorted column always go last |
| Families in the leaderboard | Filter chips above the table: All · Baselines · Profile · Statistical · ML · Neural · Ensembles. Each row shows a small family tag |
| Forecast picker and "Compare with…" | Searchable combo boxes (iced `combo_box`): type to filter by model or family name; entries ordered family by family, best overall error first within a family, labelled "Model · Family"; "Best per horizon" stays first in the forecast picker |

## Design

- `leaderboard.rs`: `SortKey` (Name, Bucket, Coverage) + direction,
  `FamilyFilter`; `Leaderboard::view(sort, filter) -> Vec<&Row>` does the
  sorting and filtering (pure, unit- and property-tested: the result is a
  permutation of the filtered rows, sorted by the key, missing values last
  in either direction). `Row` gains its `ModelId` / `Family`.
- Sort and filter state live in the app (session only, not persisted);
  messages `LeaderboardSorted(SortKey)` and `LeaderboardFiltered(FamilyFilter)`.
- Model ordering for the pickers comes from the same leaderboard (family
  order, then overall MAE, unscored last by registry order).
- `combo_box::State` for both lists lives in the app and is rebuilt when the
  available models or the leaderboard change.
- Headless screenshots of both pages before the PR.
