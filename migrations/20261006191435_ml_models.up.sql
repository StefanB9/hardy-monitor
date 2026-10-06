-- Trained forecasting models, shared by the daemon (writes) and the GUI
-- (reads). Only the newest few are kept by the application.
CREATE TABLE ml_models (
    id BIGSERIAL PRIMARY KEY,
    trained_at TIMESTAMPTZ NOT NULL,
    feature_version INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    training_samples INTEGER NOT NULL CHECK (training_samples >= 0),
    holdout_mae DOUBLE PRECISION NOT NULL,
    baseline_mae DOUBLE PRECISION NOT NULL,
    tuned BOOLEAN NOT NULL,
    -- zstd-compressed bincode model artifact (hardy-ml persistence format).
    model BYTEA NOT NULL
);

CREATE INDEX idx_ml_models_version_trained_at
    ON ml_models (feature_version, trained_at DESC);

-- Training coordination. Exactly one row.
CREATE TABLE ml_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    -- Set by the GUI's "Train model" button, cleared when training starts.
    retrain_requested_at TIMESTAMPTZ NULL,
    last_attempt_at TIMESTAMPTZ NULL,
    -- Why the last attempt did not produce a model; NULL after success.
    last_error TEXT NULL
);

INSERT INTO ml_state DEFAULT VALUES;
