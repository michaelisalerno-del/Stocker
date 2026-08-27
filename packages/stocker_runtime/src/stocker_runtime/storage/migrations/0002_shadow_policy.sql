ALTER TABLE shadow_positions ADD COLUMN policy_json TEXT NOT NULL DEFAULT '{}'
    CHECK(stocker_canonical_json(policy_json) = 1 AND length(CAST(policy_json AS BLOB)) <= 16384);
ALTER TABLE shadow_positions ADD COLUMN policy_hash TEXT NOT NULL DEFAULT
    '0000000000000000000000000000000000000000000000000000000000000000'
    CHECK(length(policy_hash) = 64);

CREATE TRIGGER shadow_positions_policy_immutable
BEFORE UPDATE OF policy_json, policy_hash ON shadow_positions
BEGIN
    SELECT RAISE(ABORT, 'shadow_policy_immutable');
END;
