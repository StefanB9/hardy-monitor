ALTER TABLE forecast_log DROP COLUMN model;
ALTER TABLE forecast_log ADD COLUMN model_id BIGINT;

CREATE TABLE ml_models (
    id BIGSERIAL PRIMARY KEY,
    trained_at TIMESTAMPTZ NOT NULL,
    feature_version INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    training_samples INTEGER NOT NULL CHECK (training_samples >= 0),
    holdout_mae DOUBLE PRECISION NOT NULL,
    baseline_mae DOUBLE PRECISION NOT NULL,
    tuned BOOLEAN NOT NULL,
    model BYTEA NOT NULL
);

CREATE INDEX idx_ml_models_version_trained_at
    ON ml_models (feature_version, trained_at DESC);

DROP TABLE live_forecasts;
DROP TABLE forecast_selection;
DROP TABLE benchmark_scores;
DROP TABLE benchmark_models;
DROP TABLE benchmark_runs;
DROP TABLE forecast_models;
