# Workspace Refactoring Plan

## Context

The project is a single Cargo crate with feature-gated GUI modules. Running the daemon requires `--no-default-features` to avoid compiling iced/ML/widgets, and the binary uses a `--daemon` CLI flag to choose between headless and GUI mode. The user wants to split into a workspace so that the daemon and GUI are independent binaries: `cargo run -p hardy-daemon` and `cargo run -p hardy-gui`.

## Architecture: 3 Workspace Members

```
hardy-monitor/                     (workspace root)
├── Cargo.toml                     [workspace] manifest
├── .cargo/config.toml             (unchanged — applies to all members)
├── rustfmt.toml                   (unchanged)
├── rust-toolchain.toml            (unchanged)
├── config.toml                    (unchanged)
├── Dockerfile                     (updated: builds hardy-daemon)
├── CLAUDE.md                      (updated: workspace commands)
├── assets/                        (reference only — moved into hardy-gui)
├── docs/                          (stays at root)
├── migrations/                    (stays at root — referenced by hardy-core)
├── .sqlx/                         (stays at root — workspace-level offline cache)
│
├── crates/
│   ├── hardy-core/                (library)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── analytics.rs
│   │   │   ├── api.rs
│   │   │   ├── config.rs          (+ MlConfig, MlAlgorithm always compiled)
│   │   │   ├── db.rs              (sqlx::migrate!("../../migrations"))
│   │   │   ├── error.rs           (MlTraining always compiled)
│   │   │   ├── repair.rs
│   │   │   ├── schedule.rs
│   │   │   └── traits.rs          (Clock, Notifier, SystemClock, MockClock, MockNotifier)
│   │   └── tests/
│   │       ├── api.rs
│   │       ├── app_logic.rs
│   │       ├── database.rs
│   │       └── common/mod.rs      (TestDatabase)
│   │
│   ├── hardy-daemon/              (binary)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs            (daemon loop, logging, fetch_and_store)
│   │
│   └── hardy-gui/                 (binary + library)
│       ├── Cargo.toml
│       ├── assets/            (icon.png moved here from root)
│       └── src/
│           ├── main.rs            (entry point, tray icon, iced runner)
│           ├── lib.rs             (module declarations)
│           ├── app.rs
│           ├── style.rs
│           ├── notifier.rs        (SystemNotifier, CombinedNotifier)
│           ├── ml/                (entire ML module tree)
│           ├── widgets/           (gauge, heatmap, history_chart)
│           └── views/             (dashboard, weekly, insights, ml_predictions, repair)
```

**Why 3 crates, not 2 or 4:**
- `hardy-core` = shared library. Both binaries depend on it. Clean boundary already exists in the code.
- `hardy-daemon` = thin binary (~120 lines). Only needs core + tokio + tracing.
- `hardy-gui` = binary + lib. All GUI/ML/widget code. Having `lib.rs` enables proper unit test compilation.
- ML as a 4th crate would add complexity without benefit — it's only consumed by the GUI and has deep coupling to core types.

## Cross-Boundary Decisions

### MlConfig → moves to hardy-core

`AppConfig` always deserializes the `[ml]` section (defaults are set in `config.rs:273-279`). `MlConfig` and `MlAlgorithm` have zero GUI dependencies (just `serde::Deserialize`, `dirs`, `std::path`). The daemon loads the same config file and simply ignores the ML settings.

- Remove `#[cfg(feature = "gui")]` from `AppConfig.ml` field
- Move `MlConfig` + `MlAlgorithm` definitions from `src/ml/config.rs` into `hardy-core/src/config.rs`
- Delete `ml/config.rs` from GUI crate (import `MlConfig` from `hardy_core::config` instead)

### MlTraining error → always compiled in hardy-core

It is a `String`-wrapping variant with no dependencies beyond `thiserror`. Remove the `#[cfg(feature = "gui")]` gate.

### SystemNotifier + CombinedNotifier → hardy-gui/src/notifier.rs

These depend on `notify-rust` (GUI-only). The `Notifier` trait, `MockNotifier`, `Clock`, `SystemClock`, `MockClock` stay in `hardy-core/src/traits.rs`.

**Future note:** If the daemon ever needs ntfy push notifications, extract the HTTP-only ntfy logic into a `NtfyNotifier` in core (it only needs `reqwest`, already a core dep). Not needed now.

### sqlx migrations path

`db.rs` currently uses `sqlx::migrate!("./migrations")`. After moving to `crates/hardy-core/`, this becomes `sqlx::migrate!("../../migrations")` (relative to `CARGO_MANIFEST_DIR`). Run `cargo sqlx prepare --workspace` after the move.

### Import path changes

All GUI-crate files change `use hardy_monitor::` → `use hardy_core::`. Within the GUI crate, ML/widget/view types use `use crate::`. Mechanical find-and-replace.

### comparison_chart.rs

Dead file — exists at `src/widgets/comparison_chart.rs` but not declared in `widgets/mod.rs`. Delete during the refactor.

## Implementation Steps

### Step 0: Save plan + create branch

```bash
git checkout dev && git pull origin dev
git checkout -b refactor/workspace-split
```

Save this plan to `docs/plan/workspace-refactor.md`.

---

### Step 1: Remove feature gates from core types (single crate, still compiles)

**Goal:** Make the single crate compile identically with and without `gui` for all core types.

**`src/config.rs`:**
- Copy `MlConfig` and `MlAlgorithm` from `src/ml/config.rs` (the struct + enum + `Default` + `resolve_model_path()` + serde defaults) into `src/config.rs`
- Remove `#[cfg(feature = "gui")]` from the `ml` field on `AppConfig`
- Change `pub ml: crate::ml::MlConfig` → `pub ml: MlConfig` (now local)
- Update `valid_app_config()` test helper — remove the `#[cfg(feature = "gui")]` on the `ml` field
- Move `MlConfig` tests from `ml/config.rs` into `config.rs` test module

**`src/error.rs`:**
- Remove `#[cfg(feature = "gui")]` from `MlTraining(String)` variant (line 29-31)
- Remove `#[cfg(feature = "gui")]` from its `category()` match arm (line 156-157)

**`src/lib.rs`:**
- Add `pub use config::{MlConfig, MlAlgorithm};` to re-exports (always compiled)
- Keep `pub use ml::...` for GUI-only ML types (OccupancyPredictor, etc.)

**`src/ml/config.rs`:**
- Replace the entire content with re-exports from config: `pub use crate::config::{MlConfig, MlAlgorithm};`
- This keeps existing `use super::config::MlConfig` imports in ML module working

**Verify:**
```bash
cargo check --all-targets
cargo check --no-default-features --all-targets
cargo nextest run
cargo nextest run --no-default-features
```

---

### Step 2: Extract notifiers to separate file (single crate, still compiles)

**`src/notifier.rs`** (new file, `#[cfg(feature = "gui")]`):
- Move `SystemNotifier` and `CombinedNotifier` (with their `impl Notifier`) from `traits.rs`
- Add needed imports: `notify_rust`, `reqwest`, `futures`, `tracing`, `anyhow`

**`src/traits.rs`:**
- Remove `SystemNotifier`, `CombinedNotifier` and their impls
- Remove the `#[cfg(feature = "gui")]` imports they needed (`notify_rust`, `reqwest`)

**`src/lib.rs`:**
- Add `#[cfg(feature = "gui")] mod notifier;`
- Change `#[cfg(feature = "gui")] pub use traits::{CombinedNotifier, SystemNotifier};` → `#[cfg(feature = "gui")] pub use notifier::{CombinedNotifier, SystemNotifier};`

**`src/main.rs`:**
- `use hardy_monitor::{CombinedNotifier, SystemClock}` — no change needed (re-export still works)

**Verify:** `cargo check --all-targets && cargo clippy --all-targets`

---

### Step 3: Create workspace structure + move files

This is the core migration step. The workspace won't fully compile until all three members are populated.

#### 3a. Create directory structure

```bash
mkdir -p crates/hardy-core/src
mkdir -p crates/hardy-core/tests/common
mkdir -p crates/hardy-daemon/src
mkdir -p crates/hardy-gui/src
```

#### 3b. Root Cargo.toml → workspace manifest

Replace the current `Cargo.toml` with:

All dependency versions and shared features are centralized here. Member crates reference them with `{ workspace = true }` only — no version numbers in crate `Cargo.toml` files. Crate-specific features use `{ workspace = true, features = ["extra"] }`.

```toml
cargo-features = ["panic-immediate-abort"]

[workspace]
members = ["crates/hardy-core", "crates/hardy-daemon", "crates/hardy-gui"]
resolver = "2"

[workspace.lints.rust]
unsafe_code = "forbid"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
todo = "deny"
print_stdout = "warn"
print_stderr = "warn"
pedantic = { level = "warn", priority = -1 }
must_use_candidate = "allow"
module_name_repetitions = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"

[workspace.dependencies]
# --- Shared across multiple crates ---
anyhow = { version = "1.0.102", default-features = false, features = ["std"] }
chrono = { version = "0.4.44", default-features = false, features = ["clock", "serde"] }
dirs = { version = "6.0.0", default-features = false }
futures = { version = "0.3.32", default-features = false }
mimalloc = { version = "0.1.48", default-features = false }
reqwest = { version = "0.13.2", default-features = false, features = ["json", "native-tls"] }
serde = { version = "1.0.228", default-features = false, features = ["derive"] }
sqlx = { version = "0.9.0-alpha.1", default-features = false, features = ["chrono", "macros", "migrate", "postgres", "runtime-tokio", "tls-rustls"] }
thiserror = { version = "2.0.18", default-features = false }
tokio = { version = "1.50.0", default-features = false, features = ["macros", "rt-multi-thread"] }
tracing = { version = "0.1.44", default-features = false, features = ["attributes", "std"] }
tracing-appender = { version = "0.2.4", default-features = false }
tracing-subscriber = { version = "0.3.23", default-features = false, features = ["env-filter", "fmt", "std"] }

# --- Core-only ---
config = { version = "0.15.21", default-features = false, features = ["toml"] }
csv = { version = "1.4.0", default-features = false }
dotenvy = { version = "0.15.7", default-features = false }

# --- GUI-only ---
approx = { version = "0.6.0-rc2", default-features = false }
bincode = { version = "2.0.1", default-features = false, features = ["std", "serde"] }
iced = { version = "0.14.0", default-features = false, features = ["advanced", "canvas", "crisp", "tokio", "wgpu"] }
image = { version = "0.25.10", default-features = false, features = ["jpeg", "png"] }
linfa = { version = "0.8.1", default-features = false }
linfa-linear = { version = "0.8.1", default-features = false }
muda = { version = "0.17.1", default-features = false }
ndarray = { version = "0.16.1", default-features = false }
notify-rust = { version = "4.12.0", default-features = false }
rayon = { version = "1.11.0", default-features = false }
smartcore = { version = "0.4.9", default-features = false }
tray-icon = { version = "0.21.3", default-features = false }
zstd = { version = "0.13.3", default-features = false }

# --- Dev dependencies ---
proptest = { version = "1.10.0" }
temp-env = { version = "0.3.6" }
tempfile = { version = "3.27.0" }
toml = { version = "1.0.6", default-features = false, features = ["parse"] }
wiremock = { version = "0.6.5" }

# --- Internal crate ---
hardy-core = { path = "crates/hardy-core" }

[profile.release]
codegen-units = 1
debug = false
debug-assertions = false
incremental = false
lto = true
opt-level = 3
overflow-checks = false
panic = "immediate-abort"
rpath = false
strip = true

[profile.dev]
codegen-units = 256
debug = true
debug-assertions = true
incremental = true
lto = false
opt-level = 0
overflow-checks = true
panic = "unwind"
rpath = true
strip = false
```

#### 3c. Populate hardy-core

**Cargo.toml:** `crates/hardy-core/Cargo.toml` — no versions, all `{ workspace = true }`
```toml
[package]
name = "hardy-core"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { workspace = true }
chrono = { workspace = true }
config = { workspace = true }
csv = { workspace = true }
dirs = { workspace = true }
dotenvy = { workspace = true }
futures = { workspace = true }
reqwest = { workspace = true }
serde = { workspace = true }
sqlx = { workspace = true }
thiserror = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }

[dev-dependencies]
approx = { workspace = true }
proptest = { workspace = true }
temp-env = { workspace = true }
tempfile = { workspace = true }
toml = { workspace = true }
wiremock = { workspace = true }

[lints]
workspace = true
```

**Move files:**
- `src/analytics.rs` → `crates/hardy-core/src/analytics.rs`
- `src/api.rs` → `crates/hardy-core/src/api.rs`
- `src/config.rs` → `crates/hardy-core/src/config.rs` (already has MlConfig from Step 1)
- `src/db.rs` → `crates/hardy-core/src/db.rs` (update `sqlx::migrate!("../../migrations")`)
- `src/error.rs` → `crates/hardy-core/src/error.rs` (already has MlTraining ungated from Step 1)
- `src/repair.rs` → `crates/hardy-core/src/repair.rs`
- `src/schedule.rs` → `crates/hardy-core/src/schedule.rs`
- `src/traits.rs` → `crates/hardy-core/src/traits.rs` (without notifiers, from Step 2)

**Create `crates/hardy-core/src/lib.rs`:**
```rust
pub mod analytics;
pub mod api;
pub mod config;
pub mod db;
pub mod error;
pub mod repair;
pub mod schedule;
pub mod traits;

// Re-exports (same as current lib.rs, minus GUI types)
pub use analytics::{...};  // all current analytics re-exports
pub use api::{GymApiClient, GymResponse};
pub use config::{AppConfig, MlConfig, MlAlgorithm};
pub use db::{Database, HourlyAverage, OccupancyLog};
pub use error::{AppError, DatabaseError, NetworkErrorKind};
pub use repair::{DataRepairer, RepairProgress, RepairSummary};
pub use schedule::{GymSchedule, is_bavarian_holiday};
pub use traits::{Clock, MockClock, MockNotifier, Notifier, SystemClock};
```

**Move integration tests:**
- `tests/api.rs` → `crates/hardy-core/tests/api.rs`
- `tests/app_logic.rs` → `crates/hardy-core/tests/app_logic.rs`
- `tests/database.rs` → `crates/hardy-core/tests/database.rs`
- `tests/common/mod.rs` → `crates/hardy-core/tests/common/mod.rs`
- Update all `use hardy_monitor::` → `use hardy_core::` in test files

**Verify:** `cargo check -p hardy-core`

#### 3d. Populate hardy-daemon

**Cargo.toml:** `crates/hardy-daemon/Cargo.toml` — no versions, all `{ workspace = true }`
```toml
[package]
name = "hardy-daemon"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { workspace = true }
chrono = { workspace = true }
hardy-core = { workspace = true }
mimalloc = { workspace = true }
tokio = { workspace = true, features = ["time"] }
tracing = { workspace = true }
tracing-appender = { workspace = true }
tracing-subscriber = { workspace = true }

[lints]
workspace = true
```

**Create `crates/hardy-daemon/src/main.rs`:**
Extract from current `main.rs`:
- `#[global_allocator]` mimalloc
- No CLI args (minimal — it IS the daemon, no `--daemon` flag needed)
- `setup_logging()` (daemon variant only — no fontdb/wgpu/naga filters)
- `main()` → calls `run_daemon()` directly
- `run_daemon()`, `wait_for_minute_alignment()`, `fetch_and_store()`
- Replace `use hardy_monitor::` → `use hardy_core::`

**Verify:** `cargo check -p hardy-daemon`

#### 3e. Populate hardy-gui

**Cargo.toml:** `crates/hardy-gui/Cargo.toml` — no versions, all `{ workspace = true }`
```toml
[package]
name = "hardy-gui"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { workspace = true }
approx = { workspace = true }
bincode = { workspace = true }
chrono = { workspace = true }
dirs = { workspace = true }
hardy-core = { workspace = true }
iced = { workspace = true }
image = { workspace = true }
linfa = { workspace = true }
linfa-linear = { workspace = true }
mimalloc = { workspace = true }
muda = { workspace = true }
ndarray = { workspace = true }
notify-rust = { workspace = true }
rayon = { workspace = true }
reqwest = { workspace = true }
serde = { workspace = true }
smartcore = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-appender = { workspace = true }
tracing-subscriber = { workspace = true }
tray-icon = { workspace = true }
zstd = { workspace = true }

[lints]
workspace = true
```

**Move files:**
- `assets/` → `crates/hardy-gui/assets/` (icon.png used only by GUI)
- `src/app.rs` → `crates/hardy-gui/src/app.rs`
- `src/style.rs` → `crates/hardy-gui/src/style.rs`
- `src/notifier.rs` → `crates/hardy-gui/src/notifier.rs` (from Step 2)
- `src/ml/` → `crates/hardy-gui/src/ml/` (entire directory)
- `src/widgets/` → `crates/hardy-gui/src/widgets/` (minus `comparison_chart.rs`)
- `src/views/` → `crates/hardy-gui/src/views/` (entire directory)

**Create `crates/hardy-gui/src/lib.rs`:**
```rust
pub mod app;
pub mod ml;
pub mod notifier;
pub mod style;
pub mod views;
pub mod widgets;
```

**Create `crates/hardy-gui/src/main.rs`:**
Extract from current `main.rs`:
- `#[global_allocator]` mimalloc
- `load_icon_async()`, `load_tray_icon_async()`
- `setup_logging()` (GUI variant — with fontdb/wgpu/naga filters)
- `main()` → calls `run_gui()` directly
- `run_gui()`, `update()`, `view()`, `subscription()`, `theme()`
- `include_bytes!("../assets/icon.png")` (assets moved into `crates/hardy-gui/assets/`)
- Replace `use hardy_monitor::` → `use hardy_core::`
- Replace `use crate::app::` → `use hardy_gui::app::` (or keep `use crate::` since main.rs is in the same crate)

**Update all imports in moved GUI files:**
- `use hardy_monitor::` → `use hardy_core::` (for core types: Database, OccupancyLog, AppConfig, etc.)
- `use crate::ml::` stays as `use crate::ml::` (still within same crate)
- `use crate::db::` → `use hardy_core::db::` (in ML modules)
- `use crate::schedule::` → `use hardy_core::schedule::` (in ML modules)
- `use crate::traits::` → `use hardy_core::traits::` (in ML modules)

**Delete `src/ml/config.rs`** from GUI crate (was already replaced with re-exports in Step 1; now import directly from `hardy_core::config::MlConfig`). Update `ml/mod.rs` accordingly.

**Verify:** `cargo check -p hardy-gui`

#### 3f. Clean up old source

- Delete `src/` directory at workspace root
- Delete old `tests/` directory at workspace root
- Delete `src/widgets/comparison_chart.rs` (dead file, not included in move)

**Verify:** `cargo check --workspace --all-targets`

---

### Step 4: Update supporting files

**Dockerfile:**
```dockerfile
# Stage 1 changes:
COPY crates ./crates                    # copy workspace members
COPY Cargo.toml Cargo.lock ./           # workspace manifest
RUN cargo build --release -p hardy-daemon

# Stage 2 changes:
COPY --from=builder /app/target/release/hardy-daemon /app/hardy-daemon
CMD ["./hardy-daemon"]                  # no --daemon flag needed
```

**CLAUDE.md** — update Quick Reference:
```bash
cargo nextest run --workspace              # All workspace tests
cargo nextest run -p hardy-core            # Core-only tests
cargo clippy --workspace --all-targets     # Zero warnings required
cargo check --workspace --all-targets
cargo fmt --all -- --check
cargo build -p hardy-daemon               # Build daemon only
cargo build -p hardy-gui                  # Build GUI only
```

Update project structure section to reflect workspace layout.

**`.sqlx/`** — regenerate:
```bash
cargo sqlx prepare --workspace
```

---

### Step 5: Verify

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo nextest run --workspace
cargo build --release -p hardy-daemon
cargo build --release -p hardy-gui
```

Also verify Docker build still works:
```bash
docker compose -f docker-compose.dev.yaml build
```

## Files Modified

| Location | Change |
|----------|--------|
| `Cargo.toml` (root) | Package manifest → workspace manifest |
| `crates/hardy-core/Cargo.toml` | New — core library dependencies |
| `crates/hardy-core/src/lib.rs` | New — module declarations + re-exports |
| `crates/hardy-core/src/*.rs` | Moved from `src/` — core modules |
| `crates/hardy-core/tests/` | Moved from `tests/` — integration tests |
| `crates/hardy-daemon/Cargo.toml` | New — daemon binary dependencies |
| `crates/hardy-daemon/src/main.rs` | New — extracted daemon logic from `main.rs` |
| `crates/hardy-gui/Cargo.toml` | New — GUI binary + lib dependencies |
| `crates/hardy-gui/src/main.rs` | New — extracted GUI logic from `main.rs` |
| `crates/hardy-gui/src/lib.rs` | New — module declarations |
| `crates/hardy-gui/src/*.rs` | Moved from `src/` — GUI modules |
| `crates/hardy-gui/src/ml/` | Moved from `src/ml/` — ML modules |
| `crates/hardy-gui/src/widgets/` | Moved from `src/widgets/` |
| `crates/hardy-gui/src/views/` | Moved from `src/views/` |
| `Dockerfile` | Updated build target and binary name |
| `CLAUDE.md` | Updated commands and project structure |

## Key Risks

1. **`sqlx::migrate!` path resolution** — `../../migrations` relative to `crates/hardy-core/Cargo.toml`. Well-supported in workspace setups but must verify.
2. **`cargo-features = ["panic-immediate-abort"]`** — nightly feature, must go at top of workspace root `Cargo.toml`. Verify it works at workspace level.
3. **`include_bytes!` for icon** — `assets/` moves into `crates/hardy-gui/assets/`, so path is `../assets/icon.png`. Clean and simple.
4. **Import churn** — many files change `hardy_monitor::` to `hardy_core::`. Mechanical but error-prone. Use find-and-replace carefully.
