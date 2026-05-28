"""
streaming/job.py — PySpark Structured Streaming anomaly scoring job
====================================================================

Run via the package entry point (recommended):
    python -m streaming

Or directly:
    python streaming/job.py

══════════════════════════════════════════════════════════════════════════════
KAFKA-SPARK CONNECTOR DEPENDENCY
══════════════════════════════════════════════════════════════════════════════

PySpark does not bundle the Kafka connector.  You need:
    org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0

This job sets spark.jars.packages in the SparkConf so the connector is
downloaded automatically from Maven Central on the first run (~30 s) and
cached in ~/.ivy2 on every subsequent run (instant).

If you prefer to supply the JAR yourself:
    # Download once
    mvn dependency:get \
        -Dartifact=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0

    # Then run
    spark-submit \
        --jars ~/.m2/repository/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar \
        streaming/job.py

Connector version matrix (the Scala cross-build suffix _2.12 stays fixed for
Spark 3.x; the version number must match your installed pyspark exactly):
    pyspark==3.3.x  →  spark-sql-kafka-0-10_2.12:3.3.4
    pyspark==3.4.x  →  spark-sql-kafka-0-10_2.12:3.4.3
    pyspark==3.5.x  →  spark-sql-kafka-0-10_2.12:3.5.0  ← default here

══════════════════════════════════════════════════════════════════════════════
STREAMING SEMANTICS — RATIONALE FOR EVERY CHOICE
══════════════════════════════════════════════════════════════════════════════

OUTPUT MODE: append
    We are appending new scored rows to Postgres; we never update or delete
    existing ones.  Spark Structured Streaming enforces that "complete" mode
    requires a bounded aggregation (GROUP BY result fits in memory — not true
    for an infinite stream), and "update" mode requires stateful aggregation
    too.  "append" is the only valid choice for our projection-only pipeline
    (no aggregation, no deduplication state).

TRIGGER: processingTime("5 seconds")
    A 5-second micro-batch gives ≈5 s end-to-end latency (Kafka → Postgres
    → API).  For a human-reviewed fraud dashboard, 5 s is well within the
    acceptable window.  The batch is large enough (≈50 tx/batch at default
    10 tx/s) to amortize Python process overhead while small enough to feel
    live on the dashboard.

    Trade-off knobs:
      • Lower latency  → reduce to "1 second" (more overhead, same throughput)
      • Higher throughput → "30 seconds" (batch-like; good for backfill)
      • Continuous processing (no trigger) → use Trigger.Continuous("1 second")
        but note that only simple stateless operations are supported there.

WATERMARK: not applied on the scoring path
    Watermarks are required only when Spark needs to bound state for a
    stateful aggregation — e.g., "count transactions per user per 60-second
    sliding window" requires knowing when Spark can safely evict old user
    state.  Our pipeline scores each transaction independently; there is no
    aggregation and no accumulated state, so a watermark adds no value and
    would only introduce artificial latency.

    If you add velocity checks (e.g., "flag if a user made >10 transactions
    in 60 s") you would add:
        .withWatermark("event_time", "10 minutes")
    before the .groupBy("user_id").agg(...) to tell Spark it can drop events
    arriving more than 10 minutes late.

CHECKPOINTING: streaming/checkpoints/
    Spark writes committed Kafka offsets to the checkpoint directory after
    each micro-batch.  On restart the job resumes exactly where it left off —
    no events lost during downtime, no re-processing unless you delete the
    checkpoint directory.  The Postgres INSERT … ON CONFLICT DO NOTHING makes
    writes idempotent, so even if a batch partially fails and re-runs, rows
    are not duplicated.  Together these give effectively-exactly-once
    semantics on this single-node setup.

FOREACHBATCH vs. STRUCTURED SINKS
    We use foreachBatch rather than separate DataStreamWriter sinks for three
    reasons:
    1. Fan-out: one micro-batch must write to two Postgres tables AND one
       Kafka topic.  foreachBatch handles this in a single Python callback
       without spawning three independent streaming queries.
    2. Idempotent writes: ON CONFLICT DO NOTHING requires psycopg2, which
       the built-in JDBC sink does not expose without custom SQL.
    3. Scoring lives here too: the model artifacts are on the driver, so
       toPandas() + score_batch() is the most direct path for a single-node
       demo.

    Production scaling path: replace toPandas() + score_batch() with a
    pandas_udf (see scorer.py for the UDF snippet) so scoring distributes
    across Spark executors.  Keep foreachBatch only for the sink fan-out.

STARTING OFFSETS: latest
    The job reads only new messages arriving after it starts.  Use "earliest"
    when you want to process the full topic history (e.g., after a model
    retrain) — but that will replay potentially millions of events; pair it
    with a fresh checkpoint directory to avoid offset conflicts.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from confluent_kafka import Producer as KafkaProducer
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# ── Project root on sys.path so `model.*` imports work from any CWD ──────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.scorer import load_artifacts, score_batch  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("streaming.job")

# ── Configuration ─────────────────────────────────────────────────────────────

BROKERS   = os.getenv("REDPANDA_BROKERS", "localhost:9092")
PG_DSN    = os.getenv("POSTGRES_DSN", "postgresql://anomaly:anomaly@localhost:5432/anomaly_db")
THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "-0.1"))

_model_env = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "isolation_forest.pkl"))
MODEL_DIR  = Path(_model_env).parent  # scorer.load_artifacts() takes a directory

CHECKPOINT_DIR = str(PROJECT_ROOT / "streaming" / "checkpoints")
TX_TOPIC       = "transactions"
ALERT_TOPIC    = "alerts"

_SPARK_KAFKA_PKG = f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}"

# ── Spark schema for incoming transaction JSON ─────────────────────────────────
# Must match the fields emitted by producer/generator.py:make_transaction().
# StringType for timestamp: we parse it in Python (via _parse_hour in features.py)
# rather than casting to TimestampType so the feature pipeline stays self-contained.
TX_SCHEMA = StructType([
    StructField("transaction_id",    StringType(),  nullable=False),
    StructField("user_id",           StringType(),  nullable=False),
    StructField("amount",            DoubleType(),  nullable=False),
    StructField("merchant_category", StringType(),  nullable=False),
    StructField("timestamp",         StringType(),  nullable=False),
    StructField("country",           StringType(),  nullable=False),
    StructField("is_anomaly",        BooleanType(), nullable=True),  # ground-truth label
    StructField("anomaly_type",      StringType(),  nullable=True),  # nullable
])

# ── Database helpers ───────────────────────────────────────────────────────────

def ensure_tables(pg_dsn: str) -> None:
    """Create Postgres tables and indexes from schema.sql if they don't exist.

    Reads schema.sql so the DDL has a single source of truth.  Call once at
    job startup before the streaming query begins.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    ddl = schema_path.read_text()
    conn = psycopg2.connect(pg_dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        log.info("Postgres tables verified/created")
    finally:
        conn.close()


def _write_transactions(conn: Any, rows: list[dict]) -> None:
    """Bulk-upsert a list of scored transaction dicts into the transactions table.

    ON CONFLICT DO NOTHING makes this idempotent: if Spark replays a
    micro-batch after a checkpoint failure, duplicate transaction_ids are
    silently dropped rather than raising an error.
    """
    psycopg2.extras.execute_values(
        conn.cursor(),
        """
        INSERT INTO transactions
            (transaction_id, user_id, amount, merchant_category,
             event_time, country, score, is_flagged, ground_truth, anomaly_type)
        VALUES %s
        ON CONFLICT (transaction_id) DO NOTHING
        """,
        [
            (
                r["transaction_id"],
                r["user_id"],
                float(r["amount"]),
                r["merchant_category"],
                r["timestamp"],      # Postgres parses ISO-8601 into TIMESTAMPTZ
                r["country"],
                float(r["score"]),
                bool(r["is_flagged"]),
                # ground_truth: preserve the generator's label for evaluation.
                # Coerce None/NaN to Python None so psycopg2 writes SQL NULL.
                bool(r["is_anomaly"]) if r.get("is_anomaly") is not None
                                      and not _is_nan(r["is_anomaly"])
                                      else None,
                r.get("anomaly_type"),
            )
            for r in rows
        ],
    )


def _write_alerts(conn: Any, rows: list[dict]) -> None:
    """Bulk-upsert flagged rows into the alerts table."""
    psycopg2.extras.execute_values(
        conn.cursor(),
        """
        INSERT INTO alerts
            (transaction_id, user_id, amount, merchant_category,
             event_time, country, score, anomaly_type)
        VALUES %s
        ON CONFLICT (transaction_id) DO NOTHING
        """,
        [
            (
                r["transaction_id"],
                r["user_id"],
                float(r["amount"]),
                r["merchant_category"],
                r["timestamp"],
                r["country"],
                float(r["score"]),
                r.get("anomaly_type"),
            )
            for r in rows
        ],
    )


def _is_nan(v: Any) -> bool:
    """Return True only if v is a float NaN; safe for non-numeric types."""
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


# ── Kafka helpers ──────────────────────────────────────────────────────────────

def _ensure_alert_topic(brokers: str) -> None:
    """Create the 'alerts' Kafka topic if it doesn't exist.

    Redpanda auto-creates topics on first produce, but being explicit lets us
    control partition count and avoids a race on job startup.
    """
    admin = AdminClient({"bootstrap.servers": brokers})
    meta  = admin.list_topics(timeout=10)
    if ALERT_TOPIC not in meta.topics:
        fs = admin.create_topics([
            NewTopic(ALERT_TOPIC, num_partitions=3, replication_factor=1)
        ])
        for t, fut in fs.items():
            try:
                fut.result()
                log.info("Created Kafka topic '%s'", t)
            except Exception as exc:
                log.warning("Could not create topic '%s': %s", t, exc)
    else:
        log.info("Kafka topic '%s' already exists", ALERT_TOPIC)


def _publish_alerts(producer: KafkaProducer, rows: list[dict]) -> None:
    """Publish each flagged transaction as a JSON message to the alerts topic.

    Key: transaction_id (bytes) — ensures all alert messages for the same
    transaction land on the same partition, which matters if downstream
    consumers need deduplication via log compaction.

    Why confluent-kafka here instead of the Spark Kafka sink?
    • We are already inside foreachBatch (driver-side); no need to create a
      second streaming query just to write alerts.
    • confluent-kafka is already a dependency (producer uses it too).
    • We get delivery callbacks, compression, and batching for free via the
      librdkafka C layer.
    """
    for r in rows:
        payload = {
            "transaction_id":    r["transaction_id"],
            "user_id":           r["user_id"],
            "amount":            float(r["amount"]),
            "merchant_category": r["merchant_category"],
            "timestamp":         r["timestamp"],
            "country":           r["country"],
            "score":             float(r["score"]),
            "anomaly_type":      r.get("anomaly_type"),
        }
        producer.produce(
            ALERT_TOPIC,
            key=r["transaction_id"].encode(),
            value=json.dumps(payload).encode(),
        )
    producer.flush()


# ── Batch processor ────────────────────────────────────────────────────────────

def make_batch_processor(model, pipeline, pg_dsn: str, alert_producer: KafkaProducer, threshold: float):
    """Return a foreachBatch callback closed over the model artifacts and connections.

    Why a factory?  foreachBatch receives (DataFrame, batch_id) only; we need
    the model, pipeline, PG DSN, Kafka producer, and threshold too.  A closure
    captures them without making them global.
    """

    def process_batch(spark_df: DataFrame, batch_id: int) -> None:
        if spark_df.isEmpty():
            log.debug("Batch %d: empty, skipping", batch_id)
            return

        # toPandas() is safe for this workload:
        #   • We run in local[*] (single-node) — driver == executor
        #   • Default rate: 10 tx/s × 5 s trigger = ~50 rows per batch
        #   • Each row is ~300 bytes → batch is <20 KB in memory
        # If you scale to a multi-worker Spark cluster, replace this with a
        # pandas_udf on the Spark DataFrame (see scorer.py for the pattern)
        # so scoring is distributed across executors.
        df: pd.DataFrame = spark_df.toPandas()
        n = len(df)

        # ── Scoring ───────────────────────────────────────────────────────────
        # score_batch accepts a DataFrame and returns an np.ndarray of floats.
        # The module-level cache in scorer.py ensures model + pipeline are
        # loaded from disk only on the first call per Python process.
        scores          = score_batch(df, model, pipeline)
        df["score"]     = scores.astype(float)
        df["is_flagged"] = df["score"] < threshold

        n_flagged = int(df["is_flagged"].sum())
        log.info(
            "Batch %d: %d events, %d flagged (threshold=%.3f, min_score=%.4f)",
            batch_id, n, n_flagged, threshold,
            float(scores.min()) if len(scores) else float("nan"),
        )

        rows         = df.to_dict("records")
        flagged_rows = [r for r in rows if r["is_flagged"]]

        # ── Postgres: write all transactions ──────────────────────────────────
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn:               # transaction context manager
                _write_transactions(conn, rows)
                if flagged_rows:
                    _write_alerts(conn, flagged_rows)
        finally:
            conn.close()

        # ── Kafka: publish flagged events to alerts topic ─────────────────────
        # Done AFTER Postgres commit so we only emit alerts for rows that
        # are durably persisted.  A partial Postgres failure means we may
        # miss publishing some alerts, but we never publish an alert that
        # wasn't saved — the safe failure direction for a fraud system.
        if flagged_rows:
            _publish_alerts(alert_producer, flagged_rows)
            log.info(
                "Batch %d: published %d alerts to Kafka topic '%s'",
                batch_id, n_flagged, ALERT_TOPIC,
            )

    return process_batch


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading model artifacts from %s", MODEL_DIR)
    model, pipeline = load_artifacts(MODEL_DIR)
    log.info(
        "Model loaded: %d trees, threshold=%.3f",
        model.n_estimators, THRESHOLD,
    )

    log.info("Ensuring Postgres tables exist")
    ensure_tables(PG_DSN)

    log.info("Ensuring Kafka alert topic exists")
    _ensure_alert_topic(BROKERS)

    # Kafka producer for the alerts fan-out.  Created once here on the driver
    # so the foreachBatch callback reuses a single persistent connection.
    alert_producer = KafkaProducer({
        "bootstrap.servers":      BROKERS,
        "queue.buffering.max.ms": 50,
        "compression.type":       "snappy",
    })

    # ── SparkSession ──────────────────────────────────────────────────────────
    # spark.jars.packages is set programmatically so the job needs no external
    # spark-submit flags.  Maven coordinates are resolved on startup and cached
    # in ~/.ivy2; first run takes ~30 s, subsequent runs are instant.
    spark = (
        SparkSession.builder
        .appName("stream-anomaly-detector")
        .master("local[*]")                  # all available CPU cores on the driver
        .config("spark.jars.packages", _SPARK_KAFKA_PKG)
        # Silence noisy Spark / Kafka connector INFO logs so our logs are readable
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        # Reduce shuffle partitions: default 200 is wasteful for micro-batches
        # of ~50 rows.  2× CPU cores is a sensible starting point.
        .config("spark.sql.shuffle.partitions", "4")
        # Driver log level: keep INFO for our logger; suppress Spark noise below
        .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    log.info("SparkSession started")

    # ── Kafka source ──────────────────────────────────────────────────────────
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BROKERS)
        .option("subscribe", TX_TOPIC)
        # "latest": only new messages arriving after job start.
        # Switch to "earliest" when you want to reprocess the full topic
        # (e.g. after a model retrain), paired with a fresh checkpoint dir.
        .option("startingOffsets", "latest")
        # Allow Spark to continue if Kafka data has been deleted (log
        # compaction / retention).  In production, set to "true" to surface
        # data-loss bugs early.
        .option("failOnDataLoss", "false")
        # Read at most 500 records per partition per micro-batch to prevent
        # a spike from overwhelming the driver on startup catch-up.
        .option("maxOffsetsPerTrigger", "500")
        .load()
    )

    # ── JSON parsing ──────────────────────────────────────────────────────────
    # Kafka delivers messages as (key BINARY, value BINARY, topic, partition,
    # offset, timestamp, timestampType, headers).  We only need value.
    # from_json with a strict schema drops unknown fields and sets missing
    # optional fields (is_anomaly, anomaly_type) to NULL.
    parsed = (
        raw
        .select(
            from_json(col("value").cast("string"), TX_SCHEMA).alias("tx"),
            col("timestamp").alias("kafka_ts"),   # broker ingestion timestamp
        )
        .select("tx.*", "kafka_ts")
        # Drop rows where JSON was malformed (from_json returns NULL struct)
        .filter(col("transaction_id").isNotNull())
    )

    # ── Streaming query ───────────────────────────────────────────────────────
    batch_fn = make_batch_processor(
        model, pipeline, PG_DSN, alert_producer, THRESHOLD
    )

    query = (
        parsed.writeStream
        .foreachBatch(batch_fn)
        # OUTPUT MODE: append — see module docstring for rationale.
        .outputMode("append")
        # TRIGGER: 5-second micro-batches — see module docstring for rationale.
        .trigger(processingTime="5 seconds")
        # CHECKPOINT: enables resume-from-offset on restart.
        .option("checkpointLocation", CHECKPOINT_DIR)
        .queryName("anomaly-scorer")
        .start()
    )

    log.info(
        "Streaming query started — reading from topic '%s', "
        "writing to Postgres + Kafka topic '%s'",
        TX_TOPIC, ALERT_TOPIC,
    )
    log.info("Checkpoint directory: %s", CHECKPOINT_DIR)
    log.info("Press Ctrl-C to stop")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        log.info("Interrupted — stopping streaming query")
        query.stop()
    finally:
        alert_producer.flush()
        spark.stop()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
