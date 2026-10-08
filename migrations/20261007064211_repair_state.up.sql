-- Progress of the daemon's nightly automatic Data Repair (single row).
CREATE TABLE repair_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    -- Last gym-local day repaired; NULL before the first run.
    repaired_through DATE,
    last_attempt_at TIMESTAMPTZ,
    last_error TEXT
);

INSERT INTO repair_state (id) VALUES (1);
