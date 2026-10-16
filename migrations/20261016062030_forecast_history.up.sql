-- The default forecast of every past open slot, made an hour ahead: at the
-- last full hour at least an hour before. Backfilled once over all history
-- (as it would have been made), then extended by every nightly run.
CREATE TABLE forecast_history (
    target TIMESTAMPTZ PRIMARY KEY,
    made_at TIMESTAMPTZ NOT NULL CHECK (made_at < target),
    -- Key of the model the value came from.
    model TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    -- The interval, once the model had enough errors for one.
    low DOUBLE PRECISION,
    high DOUBLE PRECISION
);
