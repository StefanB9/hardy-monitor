# Dependency upgrades

## Decisions (agreed)

| Crate | From → to | Notes |
|---|---|---|
| notify-rust, proptest, serde_json, toml | patch/minor | Compatible releases |
| dirs | 6 → 7 | Only `preference_dir` on Windows changed; unused |
| zstd | 0.13 → 0.14 | Same zstd 1.5.7 library; stored data unaffected |
| smartcore | 0.4 → 0.6.15 | API unchanged for us; models from 0.4 still load |
| muda + tray-icon | 0.17 → 0.21, 0.21 → 0.26 | Together; Linux builds need a tray backend (`ksni` for local headless checks only) |
| ndarray | stays 0.16 | linfa 0.8.1 (latest) requires 0.16 |
| bincode → postcard | 2.0.1 → postcard 1.1 | bincode is unmaintained (3.0.0 is a `compile_error!`) |

- **Gate test**: tests the gate with a deliberately weak model (a single
  shallow tree) instead of relying on the forest not beating a near-exact
  baseline, which smartcore 0.6 does by ~0.001 pts.
- **Model format**: postcard, `FORMAT_VERSION` 2. The version is read before
  the rest, so bincode-era models are rejected cleanly ("format version 1").
- **Unreadable stored model**: the daemon treats it as missing and retrains
  right away (with the usual retry spacing) instead of waiting for the night.

## Tests (written first)

Old-format bytes rejected with the format version; an unreadable stored model
starts a training run; full workspace suite on the new versions.
