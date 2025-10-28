-- database/schema.sql

-- Drop table if it exists
DROP TABLE IF EXISTS traffic_data;

-- Main table to store time-series traffic speed data collected from HERE API
CREATE TABLE traffic_data (
    -- Timestamp of the traffic reading (aligned to 5-minute intervals, UTC)
    -- TIMESTAMPTZ stores the timestamp along with timezone information.
    ts TIMESTAMPTZ NOT NULL,

    -- The PeMS sensor ID this reading corresponds to (based on the mapping file)
    -- VARCHAR(20) should be sufficient for PeMS IDs.
    sensor_id VARCHAR(20) NOT NULL,

    -- The actual speed in MPH fetched from the HERE API for the mapped segment/direction
    -- REAL (equivalent to float4) provides sufficient precision for speed data.
    -- Stored as NULL if data was missing from HERE API for this interval/sensor.
    actual_speed REAL,

    -- Composite Primary Key: Ensures that for any given timestamp,
    -- there is only one speed entry per sensor_id. Prevents duplicates.
    PRIMARY KEY (ts, sensor_id)
);

-- Indexes to significantly speed up common queries:

-- Index optimized for fetching time-series data for a specific sensor.
-- Ordering by ts DESC is often useful for getting the latest data quickly.
CREATE INDEX idx_traffic_data_sensor_id_ts ON traffic_data (sensor_id, ts DESC);

-- Index optimized for fetching data within a specific time range across all sensors,
-- and for the data deletion (janitor) process.
-- Ordering by ts DESC helps when querying recent time ranges.
CREATE INDEX idx_traffic_data_ts ON traffic_data (ts DESC);