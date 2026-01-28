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

# Create dummy source files to build dependencies
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "" > src/lib.rs

# Copy actual source code
COPY src ./src
COPY migrations ./migrations

# Copy sqlx offline query metadata
COPY .sqlx ./.sqlx

# Touch main.rs to ensure rebuild, then build the actual application
# SQLX_OFFLINE=true enables offline compilation without database connection
ENV SQLX_OFFLINE=true
RUN touch src/main.rs && \
    cargo build --release --no-default-features

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
COPY --from=builder /app/target/release/hardy-monitor /app/hardy-monitor

# Copy migrations (needed at runtime for sqlx::migrate!)
COPY --from=builder /app/migrations /app/migrations

# Copy config template (non-secret settings)
COPY config.toml /app/config.toml

# Set environment variables
ENV RUST_LOG=info,hardy_monitor=debug

# Run in daemon mode
CMD ["./hardy-monitor", "--daemon"]
