-- Alert settings shared by the daemon (sends alerts) and the GUI / phone
-- commands (change them). Exactly one row.
CREATE TABLE alert_settings (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    enabled BOOLEAN NOT NULL,
    threshold_percent DOUBLE PRECISION NOT NULL
        CHECK (threshold_percent BETWEEN 0 AND 100),
    -- NULL = no expiry while enabled.
    active_until TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT NOT NULL
        CHECK (updated_by IN ('migration', 'gui', 'phone'))
);

INSERT INTO alert_settings (enabled, threshold_percent, updated_by)
VALUES (false, 30, 'migration');
