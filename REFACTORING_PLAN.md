# Hardy Monitor Refactoring Plan

## Overview

This plan addresses architectural improvements, performance optimizations, and reliability enhancements for the hardy-monitor Rust/Iced application. Changes maintain compatibility with sqlx 0.9.0-alpha.1.

---

## Phase 1: Architectural Refactoring

### 1.1 Type-Safe Errors with `thiserror`

**Files to modify:**
- [ ] `Cargo.toml` - Add thiserror dependency
- [ ] `src/app.rs` - Replace AppError enum (lines 32-44)
- [ ] `src/lib.rs` - Re-export new error types

**Current AppError (lines 32-44 in app.rs):**
```rust
pub enum AppError {
    Network(String),
    Database(String),
    Validation(String),
    Io(String),
    Unknown(String),
}
```

**New implementation:**
```rust
// src/error.rs (new file)
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum AppError {
    #[error("Network error: {message}")]
    Network {
        message: String,
        #[source]
        kind: NetworkErrorKind,
    },

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("API error: {status_code} - {message}")]
    Api {
        status_code: u16,
        message: String,
    },
}

#[derive(Error, Debug, Clone)]
pub enum NetworkErrorKind {
    #[error("Connection timeout")]
    Timeout,
    #[error("Connection refused")]
    ConnectionRefused,
    #[error("DNS resolution failed")]
    DnsFailure,
    #[error("Unknown network error")]
    Unknown,
}

#[derive(Error, Debug, Clone)]
pub enum DatabaseError {
    #[error("Query failed: {query_context}")]
    QueryFailed {
        query_context: String,
        sqlx_message: String,
    },
    #[error("Connection pool exhausted")]
    PoolExhausted,
    #[error("Record not found")]
    NotFound,
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}

impl AppError {
    /// Returns true if this error is likely transient and retrying may succeed
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            AppError::Network { kind: NetworkErrorKind::Timeout, .. }
                | AppError::Database(DatabaseError::PoolExhausted)
        )
    }

    /// Create a database error from sqlx error with context
    pub fn from_sqlx(err: sqlx::Error, context: &str) -> Self {
        AppError::Database(DatabaseError::QueryFailed {
            query_context: context.to_string(),
            sqlx_message: err.to_string(),
        })
    }
}
```

**Cargo.toml addition:**
```toml
thiserror = "2.0"
```

---

### 1.2 Modularize Views

**New directory structure:**
```
src/
├── views/
│   ├── mod.rs
│   ├── dashboard.rs
│   ├── weekly_pattern.rs
│   ├── insights.rs
│   ├── data_repair.rs
│   └── components/
│       ├── mod.rs
│       ├── sidebar.rs
│       ├── header.rs
│       └── date_picker.rs
```

**Files to create:**
- [ ] `src/views/mod.rs` - Module exports
- [ ] `src/views/dashboard.rs` - Dashboard view (from lines 896-1115)
- [ ] `src/views/weekly_pattern.rs` - Weekly pattern view (from lines 1117-1223)
- [ ] `src/views/insights.rs` - Insights view (from lines 1225-1508)
- [ ] `src/views/data_repair.rs` - Data repair view (from lines 1510-1705)
- [ ] `src/views/components/mod.rs` - Shared components
- [ ] `src/views/components/sidebar.rs` - Navigation sidebar
- [ ] `src/views/components/header.rs` - Status header
- [ ] `src/views/components/date_picker.rs` - Reusable date picker

**Files to modify:**
- [ ] `src/app.rs` - Import views, delegate to view modules
- [ ] `src/lib.rs` - Export views module

**View function signatures (accept only needed data):**
```rust
// src/views/dashboard.rs
use crate::app::{MonitorState, UiState, Message};
use crate::db::HourlyAverage;
use crate::style;
use iced::{Element, Length};
use iced::widget::{column, container, row, text, button, scrollable};

/// Data required for dashboard rendering
pub struct DashboardProps<'a> {
    pub occupancy: f64,
    pub history: &'a [crate::db::OccupancyLog],
    pub best_time_today: Option<&'a HourlyAverage>,
    pub predictions: &'a [HourlyAverage],
    pub is_loading: bool,
    pub is_gym_open: bool,
    pub history_start_date: &'a str,
    pub history_end_date: &'a str,
    pub chart_cache: &'a crate::widgets::history_chart::Cache,
    pub gauge_cache: &'a crate::widgets::gauge::Cache,
}

pub fn view(props: DashboardProps<'_>) -> Element<'_, Message> {
    let content = column![
        occupancy_section(props.occupancy, props.gauge_cache, props.is_gym_open),
        best_time_card(props.best_time_today),
        history_section(
            props.history,
            props.chart_cache,
            props.history_start_date,
            props.history_end_date,
        ),
    ]
    .spacing(20)
    .padding(20);

    scrollable(content).into()
}

fn occupancy_section(
    occupancy: f64,
    cache: &crate::widgets::gauge::Cache,
    is_open: bool,
) -> Element<'_, Message> {
    // ... extracted from view_dashboard lines 920-960
}

fn best_time_card(best_time: Option<&HourlyAverage>) -> Element<'_, Message> {
    // ... extracted from view_dashboard lines 962-1010
}

fn history_section(
    history: &[crate::db::OccupancyLog],
    cache: &crate::widgets::history_chart::Cache,
    start_date: &str,
    end_date: &str,
) -> Element<'_, Message> {
    // ... extracted from view_dashboard lines 1012-1115
}
```

```rust
// src/views/insights.rs
use crate::analytics::{DayAnalysis, Insight, OccupancyStats, TrendDirection};
use crate::db::HourlyAverage;
use crate::app::Message;
use iced::Element;

pub struct InsightsProps<'a> {
    pub trend: TrendDirection,
    pub stats: Option<&'a OccupancyStats>,
    pub peak_hours: &'a [(i32, i32, f64)],
    pub quiet_hours: &'a [(i32, i32, f64)],
    pub day_analysis: &'a [DayAnalysis],
    pub insights: &'a [Insight],
}

pub fn view(props: InsightsProps<'_>) -> Element<'_, Message> {
    // ... extracted from view_insights
}
```

```rust
// src/views/mod.rs
pub mod dashboard;
pub mod weekly_pattern;
pub mod insights;
pub mod data_repair;
pub mod components;

pub use dashboard::DashboardProps;
pub use weekly_pattern::WeeklyPatternProps;
pub use insights::InsightsProps;
pub use data_repair::DataRepairProps;
```

---

### 1.3 Refactor Update Logic

**Files to modify:**
- [ ] `src/app.rs` - Extract message handlers to trait implementations

**Strategy:** Create handler traits for complex message groups and implement on state structs.

```rust
// src/app.rs - Add near top of file

/// Trait for handling data fetch completions
trait FetchHandler {
    fn handle_fetch_completed(
        &mut self,
        result: Result<f64, AppError>,
        db: &Arc<Database>,
        config: &Arc<AppConfig>,
        clock: &Arc<dyn Clock>,
        notifier: &Arc<dyn Notifier>,
        notifications: &mut NotificationState,
        ui: &mut UiState,
        schedule: &GymSchedule,
    ) -> Task<Message>;
}

impl FetchHandler for MonitorState {
    fn handle_fetch_completed(
        &mut self,
        result: Result<f64, AppError>,
        db: &Arc<Database>,
        config: &Arc<AppConfig>,
        clock: &Arc<dyn Clock>,
        notifier: &Arc<dyn Notifier>,
        notifications: &mut NotificationState,
        ui: &mut UiState,
        schedule: &GymSchedule,
    ) -> Task<Message> {
        match result {
            Ok(pct) => {
                self.occupancy = pct;
                ui.chart_cache.clear();
                ui.gauge_cache.clear();

                // Handle notification logic
                self.check_and_notify(pct, notifications, notifier, config);

                // Return combined refresh tasks
                Task::batch([
                    load_history(db.clone(), clock.clone()),
                    load_analytics(db.clone(), clock.clone(), ui.analytics_range),
                ])
            }
            Err(e) => {
                // Return error state update
                Task::none()
            }
        }
    }
}

/// Trait for handling repair operations
trait RepairHandler {
    fn handle_repair_completed(
        &mut self,
        result: Result<RepairSummary, AppError>,
    ) -> Task<Message>;

    fn handle_repair_progress(&mut self, progress: RepairProgress);
}

impl RepairHandler for RepairState {
    fn handle_repair_completed(
        &mut self,
        result: Result<RepairSummary, AppError>,
    ) -> Task<Message> {
        self.is_running = false;
        match result {
            Ok(summary) => {
                self.last_result = Some(Ok(summary));
                Task::none()
            }
            Err(e) => {
                self.last_result = Some(Err(e));
                Task::none()
            }
        }
    }

    fn handle_repair_progress(&mut self, progress: RepairProgress) {
        self.progress = Some(progress);
    }
}

/// Trait for insights data processing
trait InsightsHandler {
    fn process_insights_data(
        &mut self,
        current: Vec<HourlyAverage>,
        baseline: Option<Vec<HourlyAverage>>,
    );
}

impl InsightsHandler for MonitorState {
    fn process_insights_data(
        &mut self,
        current: Vec<HourlyAverage>,
        baseline: Option<Vec<HourlyAverage>>,
    ) {
        use crate::analytics::*;

        self.stats = Some(calculate_stats(&current));
        self.day_analysis = analyze_days(&current);
        self.peak_hours = find_peak_hours(&current, 5);
        self.quiet_hours = find_quiet_hours(&current, 5);
        self.baseline_for_comparison = baseline.clone();
        self.trend = baseline
            .as_ref()
            .map(|b| compare_periods(b, &current, ComparisonMode::WeekOverWeek).trend)
            .unwrap_or(TrendDirection::Insufficient);
        self.insights = generate_insights(&current, baseline.as_ref());
    }
}
```

**Refactored update function (partial):**
```rust
fn update(&mut self, message: Message) -> Task<Message> {
    match message {
        Message::FetchCompleted(result) => {
            self.data.handle_fetch_completed(
                result,
                &self.db,
                &self.config,
                &self.clock,
                &self.notifier,
                &mut self.notifications,
                &mut self.ui,
                &self.schedule,
            )
        }

        Message::RepairCompleted(result) => {
            self.repair.handle_repair_completed(result)
        }

        Message::RepairProgress(progress) => {
            self.repair.handle_repair_progress(progress);
            Task::none()
        }

        Message::InsightsDataLoaded { current, baseline } => {
            self.data.process_insights_data(current, baseline);
            Task::none()
        }

        // ... other message handlers
    }
}
```

---

## Phase 2: Performance & Concurrency

### 2.1 Optimize Data Repair with Concurrent Processing

**Files to modify:**
- [ ] `Cargo.toml` - Add futures crate
- [ ] `src/repair.rs` - Implement concurrent day processing
- [ ] `src/app.rs` - Update repair job spawning

**Cargo.toml addition:**
```toml
futures = "0.3"
```

**New concurrent repair implementation:**
```rust
// src/repair.rs

use futures::stream::{self, StreamExt};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Configuration for concurrent repair
pub struct RepairConfig {
    /// Maximum concurrent database operations
    pub max_concurrent_days: usize,
    /// Progress callback channel
    pub progress_tx: Option<mpsc::UnboundedSender<RepairProgress>>,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_concurrent_days: 4, // Conservative for DB connection pool
            progress_tx: None,
        }
    }
}

impl DataRepairer {
    /// Repair data for a date range with concurrent processing
    pub async fn repair_range_concurrent(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        config: RepairConfig,
    ) -> Result<RepairSummary, AppError> {
        let days: Vec<NaiveDate> = std::iter::successors(Some(start), |d| {
            let next = *d + chrono::Duration::days(1);
            if next <= end { Some(next) } else { None }
        })
        .collect();

        let total_days = days.len() as u32;
        let processed_count = Arc::new(AtomicU32::new(0));

        // Atomic counters for aggregating results
        let gaps_filled = Arc::new(AtomicU32::new(0));
        let records_zeroed = Arc::new(AtomicU32::new(0));
        let end_entries_added = Arc::new(AtomicU32::new(0));

        // Process days concurrently with bounded parallelism
        let results: Vec<Result<DayRepairResult, AppError>> = stream::iter(days)
            .map(|date| {
                let db = self.db.clone();
                let schedule = self.schedule.clone();
                let processed = processed_count.clone();
                let progress_tx = config.progress_tx.clone();

                async move {
                    let result = Self::repair_single_day(&db, &schedule, date).await;

                    // Update progress
                    let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(tx) = progress_tx {
                        let _ = tx.send(RepairProgress {
                            current_day: date,
                            total_days,
                            processed_days: count,
                        });
                    }

                    result
                }
            })
            .buffer_unordered(config.max_concurrent_days)
            .collect()
            .await;

        // Aggregate results
        let mut total_gaps = 0u32;
        let mut total_zeroed = 0u32;
        let mut total_entries = 0u32;
        let mut error_count = 0u32;

        for result in results {
            match result {
                Ok(day_result) => {
                    total_gaps += day_result.gaps_filled;
                    total_zeroed += day_result.records_zeroed;
                    total_entries += day_result.end_entries_added;
                }
                Err(_) => {
                    error_count += 1;
                    // Continue processing other days even if one fails
                }
            }
        }

        Ok(RepairSummary {
            days_processed: total_days - error_count,
            gaps_filled: total_gaps,
            records_zeroed: total_zeroed,
            end_entries_added: total_entries,
        })
    }

    /// Process a single day - extracted for concurrent execution
    async fn repair_single_day(
        db: &Arc<Database>,
        schedule: &GymSchedule,
        date: NaiveDate,
    ) -> Result<DayRepairResult, AppError> {
        let open_hour = schedule.get_open_hour(&date);
        let close_hour = schedule.get_close_hour(&date);

        let records = db.get_records_for_date(date)
            .await
            .map_err(|e| AppError::from_sqlx(e, "get_records_for_date"))?;

        let mut gaps_filled = 0u32;
        let mut records_zeroed = 0u32;
        let mut end_entries_added = 0u32;

        // 1. Fill gaps (linear interpolation for gaps <= 5 minutes)
        gaps_filled += Self::fill_gaps_for_day(db, &records, date).await?;

        // 2. Zero outside hours
        records_zeroed += Self::zero_outside_hours(db, &records, open_hour, close_hour).await?;

        // 3. Add closure entries
        end_entries_added += Self::add_closure_entry(db, &records, date, close_hour).await?;

        Ok(DayRepairResult {
            gaps_filled,
            records_zeroed,
            end_entries_added,
        })
    }

    // Helper methods for each repair operation...
    async fn fill_gaps_for_day(
        db: &Arc<Database>,
        records: &[OccupancyLog],
        date: NaiveDate,
    ) -> Result<u32, AppError> {
        // ... implementation
        Ok(0)
    }

    async fn zero_outside_hours(
        db: &Arc<Database>,
        records: &[OccupancyLog],
        open_hour: u32,
        close_hour: u32,
    ) -> Result<u32, AppError> {
        // ... implementation
        Ok(0)
    }

    async fn add_closure_entry(
        db: &Arc<Database>,
        records: &[OccupancyLog],
        date: NaiveDate,
        close_hour: u32,
    ) -> Result<u32, AppError> {
        // ... implementation
        Ok(0)
    }
}

struct DayRepairResult {
    gaps_filled: u32,
    records_zeroed: u32,
    end_entries_added: u32,
}
```

**Updated app.rs repair job spawning:**
```rust
// In update() for Message::StartRepairJob
Message::StartRepairJob => {
    if self.repair.is_running {
        return Task::none();
    }

    let start = match parse_date(&self.repair.start_date) {
        Some(d) => d,
        None => return Task::none(),
    };
    let end = match parse_date(&self.repair.end_date) {
        Some(d) => d,
        None => return Task::none(),
    };

    self.repair.is_running = true;
    self.repair.progress = None;
    self.repair.last_result = None;

    let db = self.db.clone();
    let schedule = self.schedule.clone();

    Task::perform(
        async move {
            let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

            let repairer = DataRepairer::new(db, schedule);
            let config = RepairConfig {
                max_concurrent_days: 4,
                progress_tx: Some(progress_tx),
            };

            // Spawn the repair task
            let repair_handle = tokio::spawn(async move {
                repairer.repair_range_concurrent(start, end, config).await
            });

            // Note: Progress messages are handled separately via subscription
            repair_handle.await.unwrap_or_else(|e| {
                Err(AppError::Unknown(format!("Repair task panicked: {}", e)))
            })
        },
        Message::RepairCompleted,
    )
}
```

---

### 2.2 Non-Blocking IO for Icon Loading

**Files to modify:**
- [ ] `src/main.rs` - Wrap image::load_from_memory in spawn_blocking

**Current blocking code (main.rs ~line 180-190):**
```rust
fn load_icon() -> Icon {
    let icon_bytes = include_bytes!("../assets/icon.png");
    let img = image::load_from_memory(icon_bytes)  // BLOCKING!
        .expect("Failed to load icon")
        .to_rgba8();
    // ...
}
```

**New non-blocking implementation:**
```rust
// src/main.rs

/// Load application icon without blocking the async runtime
async fn load_icon_async() -> Result<Icon, Box<dyn std::error::Error + Send + Sync>> {
    // Offload CPU-intensive image decoding to blocking thread pool
    let (width, height, rgba) = tokio::task::spawn_blocking(|| {
        let icon_bytes = include_bytes!("../assets/icon.png");
        let img = image::load_from_memory(icon_bytes)
            .expect("Failed to load embedded icon")
            .to_rgba8();
        let (width, height) = img.dimensions();
        let rgba = img.into_raw();
        (width, height, rgba)
    })
    .await?;

    Icon::from_rgba(rgba, width, height)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}

/// Synchronous wrapper for contexts where async isn't available
fn load_icon() -> Icon {
    let icon_bytes = include_bytes!("../assets/icon.png");

    // For synchronous contexts, use rayon or accept the blocking call
    // This is acceptable during app initialization
    let img = image::load_from_memory(icon_bytes)
        .expect("Failed to load icon")
        .to_rgba8();

    let (width, height) = img.dimensions();
    let rgba = img.into_raw();

    Icon::from_rgba(rgba, width, height).expect("Failed to create icon")
}

// In run_gui(), prefer async loading:
async fn run_gui(config: AppConfig) -> iced::Result {
    // Load icon asynchronously before starting GUI
    let icon = load_icon_async().await.expect("Failed to load icon");

    // ... rest of GUI initialization
}
```

---

### 2.3 Reduce Allocation in View Methods

**Files to modify:**
- [ ] `src/app.rs` - Audit view method for unnecessary clones
- [ ] `src/views/*.rs` - Use references in view props

**Identified allocation hotspots:**

1. **History vector in dashboard (line ~1050):**
```rust
// BEFORE: Cloning entire history vector
HistoryChart::new(&self.data.history.clone(), &self.ui.chart_cache)

// AFTER: Pass reference directly
HistoryChart::new(&self.data.history, &self.ui.chart_cache)
```

2. **Analytics data in weekly pattern (line ~1140):**
```rust
// BEFORE: Cloning analytics data
Heatmap::new(self.data.analytics_data.clone(), &self.ui.heatmap_cache)

// AFTER: Accept reference in Heatmap::new
Heatmap::new(&self.data.analytics_data, &self.ui.heatmap_cache)
```

**Widget signature updates:**
```rust
// src/widgets/history_chart.rs
impl<'a> HistoryChart<'a> {
    // BEFORE
    pub fn new(history: &[OccupancyLog], cache: &'a Cache) -> Self

    // AFTER - same signature, ensure no internal clone
    pub fn new(history: &'a [OccupancyLog], cache: &'a Cache) -> Self {
        Self {
            history,  // Store reference, not owned Vec
            cache,
        }
    }
}

// src/widgets/heatmap.rs
impl<'a> Heatmap<'a> {
    // BEFORE
    pub fn new(data: Vec<HourlyAverage>, cache: &'a Cache) -> Self

    // AFTER - accept reference
    pub fn new(data: &'a [HourlyAverage], cache: &'a Cache) -> Self {
        Self {
            data,
            cache,
        }
    }
}
```

**View method audit checklist:**
- [ ] `view_dashboard`: Verify `history` passed by reference
- [ ] `view_weekly_pattern`: Verify `analytics_data` passed by reference
- [ ] `view_insights`: Verify `insights`, `day_analysis`, `peak_hours`, `quiet_hours` passed by reference
- [ ] `view_data_repair`: No large data structures (already minimal)

---

## Phase 3: Reliability & UX

### 3.1 Daemon Timer Drift Correction

**Files to modify:**
- [ ] `src/main.rs` - Add periodic time alignment verification

**Current implementation (main.rs ~lines 98-152):**
```rust
async fn run_daemon(config: AppConfig) -> Result<()> {
    // ... setup ...

    // Calculate seconds to next minute ONCE at startup
    let now = Local::now();
    let seconds_to_next_minute = 60 - now.second();
    tokio::time::sleep(Duration::from_secs(seconds_to_next_minute as u64)).await;

    let mut interval = tokio::time::interval(Duration::from_secs(fetch_interval));

    loop {
        interval.tick().await;
        // ... fetch logic ...
    }
}
```

**Problem:** After system sleep/hibernation, the interval continues from where it left off, causing drift from minute boundaries.

**New implementation with drift correction:**
```rust
// src/main.rs

use std::time::Instant;

/// Daemon configuration for timing behavior
struct DaemonTiming {
    /// Target fetch interval in seconds
    fetch_interval_secs: u64,
    /// Maximum allowed drift before re-alignment (in seconds)
    max_drift_secs: u64,
    /// How often to check for drift (in fetch cycles)
    drift_check_interval: u32,
}

impl Default for DaemonTiming {
    fn default() -> Self {
        Self {
            fetch_interval_secs: 60,
            max_drift_secs: 5,
            drift_check_interval: 10, // Check every 10 fetches
        }
    }
}

async fn run_daemon(config: AppConfig) -> Result<()> {
    let db = Database::connect(&config.database.url).await?;
    let api_client = GymApiClient::new(
        &config.gym.api_url,
        config.network.request_timeout_ms,
        config.network.connect_timeout_ms,
    );
    let schedule = GymSchedule::from_config(&config.schedule);
    let timing = DaemonTiming {
        fetch_interval_secs: config.refresh.daemon_fetch_interval_secs,
        ..Default::default()
    };

    // Initial alignment to minute boundary
    align_to_minute_boundary().await;

    let mut fetch_count: u32 = 0;
    let mut last_alignment = Instant::now();

    loop {
        // Periodic drift check
        fetch_count += 1;
        if fetch_count % timing.drift_check_interval == 0 {
            if let Some(drift) = calculate_drift_from_minute() {
                if drift > timing.max_drift_secs {
                    log::warn!(
                        "Timer drift detected: {}s off from minute boundary. Re-aligning.",
                        drift
                    );
                    align_to_minute_boundary().await;
                    last_alignment = Instant::now();
                }
            }
        }

        // Check if we should skip (gym closed)
        let now = Local::now();
        if !schedule.is_open(&now) {
            log::debug!("Gym is closed, skipping fetch");
            sleep_until_next_interval(timing.fetch_interval_secs).await;
            continue;
        }

        // Perform fetch
        match fetch_and_store(&api_client, &db).await {
            Ok(pct) => log::info!("Fetched occupancy: {:.1}%", pct),
            Err(e) => log::error!("Fetch failed: {}", e),
        }

        sleep_until_next_interval(timing.fetch_interval_secs).await;
    }
}

/// Align execution to the next minute boundary
async fn align_to_minute_boundary() {
    let now = Local::now();
    let seconds_to_next_minute = 60 - now.second();
    let nanos_to_next_minute = 1_000_000_000 - now.nanosecond();

    let sleep_duration = Duration::from_secs(seconds_to_next_minute as u64)
        + Duration::from_nanos(nanos_to_next_minute as u64);

    log::debug!(
        "Aligning to minute boundary, sleeping for {:.2}s",
        sleep_duration.as_secs_f64()
    );

    tokio::time::sleep(sleep_duration).await;
}

/// Calculate how many seconds we've drifted from minute boundary
fn calculate_drift_from_minute() -> Option<u64> {
    let now = Local::now();
    let seconds_into_minute = now.second() as u64;

    // We want to be at second 0, so drift is distance from 0 or 60
    let drift = if seconds_into_minute <= 30 {
        seconds_into_minute
    } else {
        60 - seconds_into_minute
    };

    Some(drift)
}

/// Sleep until the next fetch interval, accounting for execution time
async fn sleep_until_next_interval(interval_secs: u64) {
    let now = Local::now();
    let current_second = now.second() as u64;

    // Calculate sleep to hit next interval boundary
    let next_target = ((current_second / interval_secs) + 1) * interval_secs;
    let sleep_secs = if next_target >= 60 {
        (60 - current_second) + (next_target - 60)
    } else {
        next_target - current_second
    };

    tokio::time::sleep(Duration::from_secs(sleep_secs.max(1))).await;
}
```

---

### 3.2 Debounced Loading State

**Files to modify:**
- [ ] `src/app.rs` - Add debounced loading logic to UiState and update handlers

**New UiState fields:**
```rust
// src/app.rs - UiState struct additions

pub struct UiState {
    // ... existing fields ...

    /// Timestamp when a loading operation started
    loading_started_at: Option<Instant>,
    /// Whether to actually show the loading indicator
    /// (only true if loading_started_at is > 200ms ago)
    show_loading_indicator: bool,
    /// Pending operations count (for concurrent loads)
    pending_operations: u32,
}
```

**Debounce implementation:**
```rust
// src/app.rs - New helper module or impl block

use std::time::{Duration, Instant};

const LOADING_DEBOUNCE_MS: u64 = 200;

impl UiState {
    /// Mark that a loading operation has started
    pub fn start_loading(&mut self) {
        self.pending_operations += 1;
        if self.loading_started_at.is_none() {
            self.loading_started_at = Some(Instant::now());
        }
    }

    /// Mark that a loading operation has completed
    pub fn finish_loading(&mut self) {
        self.pending_operations = self.pending_operations.saturating_sub(1);
        if self.pending_operations == 0 {
            self.loading_started_at = None;
            self.show_loading_indicator = false;
        }
    }

    /// Check if we should show the loading indicator
    /// Call this from the Tick handler
    pub fn update_loading_visibility(&mut self) {
        if let Some(started) = self.loading_started_at {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(LOADING_DEBOUNCE_MS) {
                self.show_loading_indicator = true;
            }
        }
    }

    /// Returns true if the loading indicator should be displayed
    pub fn is_visibly_loading(&self) -> bool {
        self.show_loading_indicator
    }
}
```

**Integration in update():**
```rust
fn update(&mut self, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            // Update loading visibility on each tick
            self.ui.update_loading_visibility();

            // ... rest of tick handling
        }

        Message::FetchTick => {
            // Mark loading started
            self.ui.start_loading();

            // ... spawn fetch task
        }

        Message::FetchCompleted(result) => {
            // Mark loading finished
            self.ui.finish_loading();

            // ... handle result
        }

        Message::HistoryLoaded(result) => {
            self.ui.finish_loading();
            // ... handle result
        }

        // ... similar for other async completions
    }
}
```

**View integration:**
```rust
// In header/status display
fn view_header(&self) -> Element<'_, Message> {
    let status_text = if self.ui.is_visibly_loading() {
        text("Updating...").color(style::ACCENT_ORANGE)
    } else if self.schedule.is_open(&self.clock.now_local()) {
        text("Live").color(style::ACCENT_GREEN)
    } else {
        text("Gym Closed").color(style::TEXT_MUTED)
    };

    // ... rest of header
}
```

---

## Implementation Checklist

### Phase 1: Architectural Refactoring

#### 1.1 Type-Safe Errors
- [ ] Add `thiserror = "2.0"` to Cargo.toml
- [ ] Create `src/error.rs` with new AppError enum
- [ ] Add `NetworkErrorKind` enum with variants
- [ ] Add `DatabaseError` enum with variants
- [ ] Implement `is_retryable()` method on AppError
- [ ] Implement `from_sqlx()` helper method
- [ ] Update `src/lib.rs` to export error module
- [ ] Update all `AppError::Database(e.to_string())` to `AppError::from_sqlx(e, "context")`
- [ ] Update all `AppError::Network` usages with proper NetworkErrorKind
- [ ] Update error display in UI to use new structured errors

#### 1.2 Modularize Views
- [ ] Create `src/views/` directory
- [ ] Create `src/views/mod.rs` with module exports
- [ ] Create `src/views/components/mod.rs`
- [ ] Extract `view_dashboard` to `src/views/dashboard.rs`
- [ ] Define `DashboardProps` struct with required references
- [ ] Extract `view_weekly_pattern` to `src/views/weekly_pattern.rs`
- [ ] Define `WeeklyPatternProps` struct
- [ ] Extract `view_insights` to `src/views/insights.rs`
- [ ] Define `InsightsProps` struct
- [ ] Extract `view_data_repair` to `src/views/data_repair.rs`
- [ ] Define `DataRepairProps` struct
- [ ] Extract sidebar to `src/views/components/sidebar.rs`
- [ ] Extract header to `src/views/components/header.rs`
- [ ] Extract date picker to `src/views/components/date_picker.rs`
- [ ] Update `src/app.rs` to use new view modules
- [ ] Update `src/lib.rs` to export views

#### 1.3 Refactor Update Logic
- [ ] Define `FetchHandler` trait
- [ ] Implement `FetchHandler` for `MonitorState`
- [ ] Define `RepairHandler` trait
- [ ] Implement `RepairHandler` for `RepairState`
- [ ] Define `InsightsHandler` trait
- [ ] Implement `InsightsHandler` for `MonitorState`
- [ ] Refactor `Message::FetchCompleted` to use trait
- [ ] Refactor `Message::RepairCompleted` to use trait
- [ ] Refactor `Message::InsightsDataLoaded` to use trait
- [ ] Extract notification logic to separate method

### Phase 2: Performance & Concurrency

#### 2.1 Concurrent Data Repair
- [ ] Add `futures = "0.3"` to Cargo.toml
- [ ] Add `RepairConfig` struct with concurrent settings
- [ ] Implement `repair_range_concurrent` method
- [ ] Extract `repair_single_day` as independent async fn
- [ ] Add atomic counters for result aggregation
- [ ] Implement progress reporting via channel
- [ ] Update `Message::StartRepairJob` handler
- [ ] Add integration tests for concurrent repair

#### 2.2 Non-Blocking Icon Loading
- [ ] Create `load_icon_async` function
- [ ] Wrap `image::load_from_memory` in `spawn_blocking`
- [ ] Update `run_gui` to use async icon loading where possible
- [ ] Verify no other blocking operations in async context

#### 2.3 Reduce Allocations
- [ ] Audit `HistoryChart::new` - ensure reference not clone
- [ ] Audit `Heatmap::new` - change to accept `&[HourlyAverage]`
- [ ] Audit `view_dashboard` for unnecessary clones
- [ ] Audit `view_weekly_pattern` for unnecessary clones
- [ ] Audit `view_insights` for unnecessary clones
- [ ] Update widget lifetimes if needed for reference parameters

### Phase 3: Reliability & UX

#### 3.1 Daemon Timer Drift
- [ ] Create `DaemonTiming` configuration struct
- [ ] Implement `align_to_minute_boundary` function
- [ ] Implement `calculate_drift_from_minute` function
- [ ] Implement `sleep_until_next_interval` function
- [ ] Add drift check to daemon main loop
- [ ] Add logging for drift detection and correction
- [ ] Test with simulated system sleep

#### 3.2 Debounced Loading
- [ ] Add `loading_started_at: Option<Instant>` to UiState
- [ ] Add `show_loading_indicator: bool` to UiState
- [ ] Add `pending_operations: u32` to UiState
- [ ] Implement `start_loading()` method
- [ ] Implement `finish_loading()` method
- [ ] Implement `update_loading_visibility()` method
- [ ] Implement `is_visibly_loading()` method
- [ ] Update `Message::Tick` to call `update_loading_visibility`
- [ ] Update all async operation starts to call `start_loading`
- [ ] Update all async completions to call `finish_loading`
- [ ] Update header view to use `is_visibly_loading()`

---

## Testing Strategy

### Unit Tests
- [ ] Test `AppError::is_retryable()` for all variants
- [ ] Test `AppError::from_sqlx()` context preservation
- [ ] Test `UiState` debounce timing logic
- [ ] Test `calculate_drift_from_minute()` at various seconds

### Integration Tests
- [ ] Test concurrent repair with mock database
- [ ] Test daemon drift correction after simulated delay
- [ ] Test view prop construction doesn't clone data

### Manual Testing
- [ ] Verify UI doesn't flicker during fast operations
- [ ] Verify "Updating..." appears only after 200ms
- [ ] Verify repair progress updates correctly with concurrent processing
- [ ] Verify daemon recovers after system sleep

---

## Migration Notes

1. **Error Handling Migration**: Update all call sites gradually. The new `AppError` is backward compatible for display purposes.

2. **View Module Migration**: Extract one view at a time, starting with the simplest (`data_repair`), then `weekly_pattern`, `dashboard`, and finally `insights`.

3. **Concurrent Repair**: Keep the sequential implementation available as fallback. Add a configuration option to choose between sequential and concurrent modes.

4. **Debounced Loading**: This is additive and won't break existing behavior. Start with a higher debounce threshold (500ms) and tune down to 200ms.
