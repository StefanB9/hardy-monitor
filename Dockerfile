# ============================================
# Stage 1: Build
# ============================================
FROM rustlang/rust:nightly-bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install rust-src for build-std (required for panic-immediate-abort)
RUN rustup component add rust-src

WORKDIR /app

# Copy manifests and cargo config for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo

# Copy workspace member manifests for dependency caching
COPY crates/hardy-core/Cargo.toml crates/hardy-core/Cargo.toml
COPY crates/hardy-daemon/Cargo.toml crates/hardy-daemon/Cargo.toml
COPY crates/hardy-gui/Cargo.toml crates/hardy-gui/Cargo.toml

# Create dummy source files to build dependencies
RUN mkdir -p crates/hardy-core/src && \
    echo "pub fn _dummy() {}" > crates/hardy-core/src/lib.rs && \
    mkdir -p crates/hardy-daemon/src && \
    echo "fn main() {}" > crates/hardy-daemon/src/main.rs && \
    mkdir -p crates/hardy-gui/src && \
    echo "fn main() {}" > crates/hardy-gui/src/main.rs && \
    echo "" > crates/hardy-gui/src/lib.rs

# Copy actual source code
COPY crates ./crates
COPY migrations ./migrations

# Copy sqlx offline query metadata
COPY .sqlx ./.sqlx

# Build the daemon binary
# SQLX_OFFLINE=true enables offline compilation without database connection
ENV SQLX_OFFLINE=true
RUN touch crates/hardy-daemon/src/main.rs && \
    cargo build --release -p hardy-daemon

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/hardy-daemon /app/hardy-daemon

# Copy migrations (needed at runtime for sqlx::migrate!)
COPY --from=builder /app/migrations /app/migrations

# Copy config template (non-secret settings)
COPY config.toml /app/config.toml

# Set environment variables
ENV RUST_LOG=info,hardy_core=debug,hardy_daemon=debug

# Run daemon
CMD ["./hardy-daemon"]
