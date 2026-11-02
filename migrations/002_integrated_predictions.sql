-- Additive migration; existing Task 05 results remain retrievable with null new fields.
BEGIN;
ALTER TABLE predictions
    ADD COLUMN features jsonb CHECK (features IS NULL OR jsonb_typeof(features) = 'object'),
    ADD COLUMN classification jsonb CHECK (classification IS NULL OR jsonb_typeof(classification) = 'object'),
    ADD COLUMN severity jsonb CHECK (severity IS NULL OR jsonb_typeof(severity) = 'object');
CREATE INDEX inspections_created_id_idx ON inspections (created_at DESC, id DESC);
INSERT INTO damage_schema_migrations (version) VALUES (2);
COMMIT;
