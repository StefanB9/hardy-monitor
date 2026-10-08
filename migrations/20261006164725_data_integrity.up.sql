-- Enforce one row per UTC minute, valid percentages and record provenance.
-- Rows violating the new rules are moved to a quarantine table, never deleted.

CREATE TABLE occupancy_logs_quarantine (
    id BIGINT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    percentage DOUBLE PRECISION NOT NULL,
    reason TEXT NOT NULL,
    quarantined_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- NaN sorts above every number in Postgres, so NOT BETWEEN also catches it.
WITH moved AS (
    DELETE FROM occupancy_logs
    WHERE NOT (percentage BETWEEN 0 AND 100)
    RETURNING id, timestamp, percentage
)
INSERT INTO occupancy_logs_quarantine (id, timestamp, percentage, reason)
SELECT id, timestamp, percentage, 'out_of_range' FROM moved;

-- Keep the first-inserted row per minute: the daemon's measurement always
-- precedes rows that Data Repair adds later.
WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY date_trunc('minute', timestamp, 'UTC')
            ORDER BY id
        ) AS rn
    FROM occupancy_logs
),
moved AS (
    DELETE FROM occupancy_logs o
    USING ranked r
    WHERE o.id = r.id AND r.rn > 1
    RETURNING o.id, o.timestamp, o.percentage
)
INSERT INTO occupancy_logs_quarantine (id, timestamp, percentage, reason)
SELECT id, timestamp, percentage, 'duplicate_minute' FROM moved;

UPDATE occupancy_logs
SET timestamp = date_trunc('minute', timestamp, 'UTC')
WHERE timestamp <> date_trunc('minute', timestamp, 'UTC');

ALTER TABLE occupancy_logs
    ADD COLUMN source TEXT NOT NULL DEFAULT 'measured',
    ADD CONSTRAINT occupancy_logs_source_valid
        CHECK (source IN ('measured', 'interpolated', 'boundary', 'smoothed')),
    ADD CONSTRAINT occupancy_logs_percentage_range
        CHECK (percentage BETWEEN 0 AND 100),
    ADD CONSTRAINT occupancy_logs_timestamp_minute
        CHECK (timestamp = date_trunc('minute', timestamp, 'UTC'));

DROP INDEX IF EXISTS idx_occupancy_logs_timestamp;
ALTER TABLE occupancy_logs
    ADD CONSTRAINT occupancy_logs_timestamp_key UNIQUE (timestamp);
