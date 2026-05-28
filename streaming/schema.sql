-- streaming/schema.sql
-- Run once before starting the streaming job, or let ensure_tables() in job.py do it.
--
--   psql "$POSTGRES_DSN" -f streaming/schema.sql

-- ── All scored transactions ────────────────────────────────────────────────────
-- Stores every event that passes through the Spark job, whether flagged or not.
-- The API's GET /transactions endpoint reads from here; the dashboard plots
-- score distributions; the evaluator computes precision/recall against ground_truth.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    TEXT            PRIMARY KEY,
    user_id           TEXT            NOT NULL,
    amount            NUMERIC(12, 2)  NOT NULL,
    merchant_category TEXT            NOT NULL,
    -- event_time is the producer-stamped timestamp (ISO-8601 string from the
    -- Kafka message), parsed by Postgres into a timezone-aware instant.
    -- Using the event timestamp (not ingestion time) is critical for correct
    -- time-series plots and for any future windowed aggregations in the API.
    event_time        TIMESTAMPTZ     NOT NULL,
    country           TEXT            NOT NULL,
    -- IsolationForest decision_function score: lower = more anomalous.
    -- Stored as DOUBLE PRECISION because sklearn returns float64 and we want
    -- full precision for threshold tuning without round-trip loss.
    score             DOUBLE PRECISION NOT NULL,
    -- Derived flag: score < ANOMALY_THRESHOLD at the time of Spark processing.
    -- Can be recomputed by the API if the threshold changes.
    is_flagged        BOOLEAN         NOT NULL,
    -- Ground-truth label injected by the synthetic producer.
    -- NULL in production (no labels); present here for offline evaluation.
    ground_truth      BOOLEAN,
    anomaly_type      TEXT,           -- "amount_spike" | "geo_jump" | "odd_hour_burst" | NULL
    processed_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Descending event_time index: API queries always ORDER BY event_time DESC LIMIT n.
CREATE INDEX IF NOT EXISTS idx_tx_event_time ON transactions (event_time DESC);
-- Partial index on flagged rows: GET /anomalies scans only the small flagged subset.
CREATE INDEX IF NOT EXISTS idx_tx_is_flagged  ON transactions (is_flagged)
    WHERE is_flagged = TRUE;

-- ── Flagged anomalies only ─────────────────────────────────────────────────────
-- Mirrors the alerts Kafka topic: every row here was also published to the
-- "alerts" topic so downstream consumers (API SSE, dashboard) can subscribe.
-- Keeping a separate table avoids a full-scan on transactions when the API
-- serves GET /anomalies or the SSE endpoint polls for new alerts.
CREATE TABLE IF NOT EXISTS alerts (
    transaction_id    TEXT            PRIMARY KEY,
    user_id           TEXT            NOT NULL,
    amount            NUMERIC(12, 2)  NOT NULL,
    merchant_category TEXT            NOT NULL,
    event_time        TIMESTAMPTZ     NOT NULL,
    country           TEXT            NOT NULL,
    score             DOUBLE PRECISION NOT NULL,
    anomaly_type      TEXT,
    alerted_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Most-recent-first index: SSE polling queries "alerts WHERE alerted_at > $last_seen".
CREATE INDEX IF NOT EXISTS idx_alerts_alerted_at ON alerts (alerted_at DESC);
