-- Every model's live forecast, logged every full hour while the gym is
-- open, to measure each model's real-world accuracy. Element i of each
-- array is the forecast for made_at + 15 min × (i + 1); NULL where the gym
-- is closed (and in low / high where the model had no interval yet).
CREATE TABLE model_forecast_log (
    made_at TIMESTAMPTZ NOT NULL,
    -- Model key; 'selected' is the default forecast.
    model TEXT NOT NULL,
    forecast REAL[] NOT NULL,
    low REAL[] NOT NULL,
    high REAL[] NOT NULL,
    PRIMARY KEY (made_at, model)
);

-- The models each benchmark run selected per horizon bucket (1–3), kept
-- with the run so selection changes can be shown over time.
CREATE TABLE benchmark_selection (
    run_id BIGINT NOT NULL REFERENCES benchmark_runs (id) ON DELETE CASCADE,
    bucket SMALLINT NOT NULL CHECK (bucket BETWEEN 1 AND 3),
    model TEXT NOT NULL,
    PRIMARY KEY (run_id, bucket)
);

-- The current selection is the only one known for earlier runs.
INSERT INTO benchmark_selection (run_id, bucket, model)
SELECT run_id, bucket, model FROM forecast_selection WHERE run_id IS NOT NULL;
