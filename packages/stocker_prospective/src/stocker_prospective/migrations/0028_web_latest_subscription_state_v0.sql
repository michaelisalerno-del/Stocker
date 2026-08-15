CREATE TABLE IF NOT EXISTS web_latest_subscription_state_v0 (
    run_id TEXT NOT NULL REFERENCES prospective_run(run_id),
    subscription_key TEXT NOT NULL,
    event_id INTEGER NOT NULL REFERENCES subscription_lifecycle_event_v0(id),
    subscription_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY(run_id, subscription_key)
);

INSERT INTO web_latest_subscription_state_v0(
    run_id,
    subscription_key,
    event_id,
    subscription_kind,
    status
)
SELECT event.run_id,
       event.subscription_key,
       event.id,
       event.subscription_kind,
       event.status
FROM subscription_lifecycle_event_v0 AS event
JOIN (
    SELECT run_id,
           subscription_key,
           MAX(id) AS event_id
    FROM subscription_lifecycle_event_v0
    GROUP BY run_id, subscription_key
) AS latest
  ON latest.event_id = event.id
ON CONFLICT(run_id, subscription_key) DO UPDATE SET
    event_id = excluded.event_id,
    subscription_kind = excluded.subscription_kind,
    status = excluded.status
WHERE excluded.event_id > web_latest_subscription_state_v0.event_id;

CREATE TRIGGER IF NOT EXISTS web_latest_subscription_state_insert_v0
AFTER INSERT ON subscription_lifecycle_event_v0
BEGIN
    INSERT INTO web_latest_subscription_state_v0(
        run_id,
        subscription_key,
        event_id,
        subscription_kind,
        status
    ) VALUES (
        NEW.run_id,
        NEW.subscription_key,
        NEW.id,
        NEW.subscription_kind,
        NEW.status
    )
    ON CONFLICT(run_id, subscription_key) DO UPDATE SET
        event_id = excluded.event_id,
        subscription_kind = excluded.subscription_kind,
        status = excluded.status
    WHERE excluded.event_id > web_latest_subscription_state_v0.event_id;
END;

CREATE INDEX IF NOT EXISTS idx_web_latest_subscription_active_v0
ON web_latest_subscription_state_v0(run_id, status, subscription_kind);
