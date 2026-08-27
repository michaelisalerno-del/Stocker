-- Phase 6 query-plan measurements showed full scans or temporary ordering for
-- the three high-cardinality read-model streams. These indexes cover only
-- their timestamp/id keyset order; no domain storage is introduced.

CREATE INDEX market_latest_run_event_time_idx
    ON market_latest(run_id, event_at_us DESC, event_id);

CREATE INDEX idea_instances_activation_time_idx
    ON idea_instances(activated_at_us DESC, instance_id);

CREATE INDEX idea_instances_run_activation_time_idx
    ON idea_instances(run_id, activated_at_us DESC, instance_id);

CREATE INDEX idea_outputs_instance_time_web_idx
    ON idea_outputs(instance_id, as_of_at_us DESC, output_id);
