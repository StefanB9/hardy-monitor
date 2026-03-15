-- Convert timestamp column from TEXT to TIMESTAMPTZ for native datetime handling.
-- This eliminates string allocation on every DB read/write and enables native
-- PostgreSQL timestamp comparisons and indexing.
ALTER TABLE occupancy_logs
    ALTER COLUMN timestamp TYPE TIMESTAMPTZ
    USING timestamp::TIMESTAMPTZ;
