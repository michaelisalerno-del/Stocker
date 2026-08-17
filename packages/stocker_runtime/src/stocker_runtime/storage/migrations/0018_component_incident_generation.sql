ALTER TABLE incidents ADD COLUMN recorder_generation INTEGER
    CHECK(recorder_generation IS NULL OR recorder_generation >= 0);
