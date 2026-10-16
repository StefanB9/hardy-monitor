-- Which forecaster is running: at most one row. The holder renews it every
-- minute; a lease not renewed for a while counts as abandoned. A table, not
-- an advisory lock, so it works through connection poolers.
CREATE TABLE forecaster_lease (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    holder TEXT NOT NULL,
    renewed_at TIMESTAMPTZ NOT NULL
);
