-- Apply manually with psql -v ON_ERROR_STOP=1. Reapplication intentionally fails.
BEGIN;

CREATE TABLE IF NOT EXISTS damage_schema_migrations (
    version integer PRIMARY KEY CHECK (version > 0),
    applied_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inspections (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL,
    upload_sha256 text NOT NULL CHECK (upload_sha256 ~ '^[0-9a-f]{64}$'),
    image_key text NOT NULL UNIQUE,
    image_sha256 text NOT NULL CHECK (image_sha256 ~ '^[0-9a-f]{64}$'),
    width integer NOT NULL CHECK (width > 0),
    height integer NOT NULL CHECK (height > 0),
    source_media_type text NOT NULL CHECK (source_media_type IN ('image/jpeg', 'image/png')),
    source_mode text NOT NULL CHECK (source_mode IN ('RGB', 'L'))
);

CREATE TABLE predictions (
    inspection_id uuid PRIMARY KEY REFERENCES inspections(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    mask_key text NOT NULL UNIQUE,
    mask_sha256 text NOT NULL CHECK (mask_sha256 ~ '^[0-9a-f]{64}$'),
    checkpoint_sha256 text NOT NULL CHECK (checkpoint_sha256 ~ '^[0-9a-f]{64}$'),
    model_metadata jsonb NOT NULL CHECK (jsonb_typeof(model_metadata) = 'object'),
    analytics jsonb NOT NULL CHECK (jsonb_typeof(analytics) = 'object')
);

INSERT INTO damage_schema_migrations (version) VALUES (1);
COMMIT;
