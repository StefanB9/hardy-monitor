-- Revert data_integrity. Quarantined rows are restored; the sub-minute part of
-- truncated timestamps cannot be recovered.

ALTER TABLE occupancy_logs
    DROP CONSTRAINT occupancy_logs_timestamp_key,
    DROP CONSTRAINT occupancy_logs_timestamp_minute,
    DROP CONSTRAINT occupancy_logs_percentage_range,
    DROP CONSTRAINT occupancy_logs_source_valid,
    DROP COLUMN source;

CREATE INDEX IF NOT EXISTS idx_occupancy_logs_timestamp ON occupancy_logs(timestamp);

INSERT INTO occupancy_logs (id, timestamp, percentage)
SELECT id, timestamp, percentage FROM occupancy_logs_quarantine;

DROP TABLE occupancy_logs_quarantine;
