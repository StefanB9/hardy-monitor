-- Forecasts the daemon made, kept to measure their accuracy once the
-- predicted time has passed.
CREATE TABLE forecast_log (
    made_at TIMESTAMPTZ NOT NULL,
    target TIMESTAMPTZ NOT NULL,
    horizon_hours SMALLINT NOT NULL CHECK (horizon_hours > 0),
    -- Model that produced `predicted`; NULL when it came from plain averages
    -- (no model yet, or no recent reading to start from).
    model_id BIGINT,
    predicted DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    -- Plain slot-average forecast for the same target, for comparison.
    baseline DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (made_at, horizon_hours)
);

CREATE INDEX forecast_log_target ON forecast_log (target);
