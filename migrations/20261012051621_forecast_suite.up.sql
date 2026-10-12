-- The forecasting suite: every model's latest fit, the nightly benchmark,
-- the selected model per horizon and the live forecasts the GUI shows. It
-- replaces the single stored model (ml_models); ml_state keeps coordinating
-- the nightly run.

-- One row per model: its newest fit (or why there is none).
CREATE TABLE forecast_models (
    model TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('fitted', 'waiting', 'failed')),
    -- Why the model is waiting or failed.
    message TEXT,
    fitted_at TIMESTAMPTZ,
    -- Readings the fit used.
    data_from TIMESTAMPTZ,
    data_until TIMESTAMPTZ,
    -- zstd-compressed model state (hardy-ml format `artifact_version`).
    artifact_version INTEGER,
    artifact BYTEA,
    updated_at TIMESTAMPTZ NOT NULL
);

-- One row per nightly benchmark.
CREATE TABLE benchmark_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    -- Backtested gym-local days.
    first_day DATE,
    last_day DATE,
    days INTEGER NOT NULL CHECK (days >= 0),
    origins INTEGER NOT NULL CHECK (origins >= 0)
);

-- Per model and run: how much it took part.
CREATE TABLE benchmark_models (
    run_id BIGINT NOT NULL REFERENCES benchmark_runs (id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    origins INTEGER NOT NULL CHECK (origins >= 0),
    fitted_days INTEGER NOT NULL CHECK (fitted_days >= 0),
    -- Forecast from enough origins to be selected.
    eligible BOOLEAN NOT NULL,
    failure TEXT,
    PRIMARY KEY (run_id, model)
);

-- Per model, run and horizon bucket (0 = all horizons, 1 = next hour,
-- 2 = 1–3 h, 3 = 3–24 h): its scores and conformal interval offsets.
CREATE TABLE benchmark_scores (
    run_id BIGINT NOT NULL,
    model TEXT NOT NULL,
    bucket SMALLINT NOT NULL CHECK (bucket BETWEEN 0 AND 3),
    mae DOUBLE PRECISION NOT NULL,
    rmse DOUBLE PRECISION NOT NULL,
    mase DOUBLE PRECISION,
    coverage DOUBLE PRECISION,
    pinball DOUBLE PRECISION,
    errors INTEGER NOT NULL CHECK (errors >= 0),
    interval_low DOUBLE PRECISION,
    interval_high DOUBLE PRECISION,
    PRIMARY KEY (run_id, model, bucket),
    FOREIGN KEY (run_id, model) REFERENCES benchmark_models (run_id, model) ON DELETE CASCADE
);

-- The model shown by default for each horizon bucket (1–3).
CREATE TABLE forecast_selection (
    bucket SMALLINT PRIMARY KEY CHECK (bucket BETWEEN 1 AND 3),
    model TEXT NOT NULL,
    run_id BIGINT REFERENCES benchmark_runs (id) ON DELETE SET NULL,
    selected_at TIMESTAMPTZ NOT NULL
);

-- The newest forecast of every model, replaced as new readings arrive.
-- `model` 'selected' holds the default forecast stitched from the selected
-- models; `source` names the model each of its values came from.
CREATE TABLE live_forecasts (
    model TEXT NOT NULL,
    target TIMESTAMPTZ NOT NULL,
    made_at TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION,
    high DOUBLE PRECISION,
    source TEXT,
    PRIMARY KEY (model, target)
);

DROP TABLE ml_models;

-- Logged forecasts name the suite model that made them.
ALTER TABLE forecast_log DROP COLUMN model_id;
ALTER TABLE forecast_log ADD COLUMN model TEXT;
