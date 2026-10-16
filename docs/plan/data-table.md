# Data tab

A table of every stored reading, so the raw data behind the charts can be
read point by point.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Rows | Every stored reading (one per minute) of the chosen period: time, occupancy, source (measured, interpolated, boundary, smoothed) |
| Place | New sidebar tab "Data" after Week |
| Delivery | Its own PR after the history backfill (PR 2) merged |

## Details (defaults chosen while planning)

| Topic | Choice |
|---|---|
| Periods | Day or week (Monday to Sunday, gym-local), switched with a segmented control; ‹ › and the arrow keys step back to the first reading and forward to the current period |
| Order | Oldest first, as the chart reads |
| Summary | Above the table: readings shown, and how many were measured, filled in or changed by Data Repair |
| Size | A week is about 7,000 rows. Only the rows in view (plus a margin) are built: the table is a fixed-row-height list inside a scrollable that tracks its scroll offset; spacers stand in for the rows above and below |
| Loading | On entering the tab, on changing or stepping the period, and on Refresh; an older load that returns late is ignored (like the chart) |
| Code | Pure logic (period windows and labels, visible row range, summary counts) in `data_table.rs` with unit and property tests; the view in `views/data.rs` |
| Dependencies | None new |
