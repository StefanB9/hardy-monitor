-- Create the occupancy_logs table for storing gym occupancy snapshots
CREATE TABLE IF NOT EXISTS occupancy_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,  -- ISO 8601 formatted UTC timestamp (RFC3339)
    percentage DOUBLE PRECISION NOT NULL  -- Occupancy percentage (0.0 - 100.0)
);

-- Index for efficient time-range queries
CREATE INDEX IF NOT EXISTS idx_occupancy_logs_timestamp ON occupancy_logs(timestamp);
