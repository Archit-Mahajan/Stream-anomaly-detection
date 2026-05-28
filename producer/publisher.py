"""
Kafka producer: serializes synthetic transactions to JSON and publishes them
to the configured topic on Redpanda (or any Kafka-compatible broker).

Keying messages by user_id ensures all transactions for the same user land on
the same partition, which preserves per-user event ordering.  The Spark
streaming job relies on this ordering to compute velocity features (e.g.
seconds since the same user's last transaction in a different country).
"""

import json
import logging
import os
import time

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

from .generator import get_config, make_transaction

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TOPIC   = os.getenv("KAFKA_TOPIC",      "transactions")
BROKERS = os.getenv("REDPANDA_BROKERS", "localhost:9092")


def _ensure_topic(brokers: str, topic: str) -> None:
    """Create the topic if it does not already exist.

    Redpanda can auto-create topics on first produce, but being explicit gives
    us control over partition count (3 here — enough for parallelism without
    waste on a single-node dev cluster) and avoids a race on first startup.
    """
    admin = AdminClient({"bootstrap.servers": brokers})
    meta  = admin.list_topics(timeout=10)
    if topic not in meta.topics:
        fs = admin.create_topics([
            NewTopic(topic, num_partitions=3, replication_factor=1)
        ])
        for t, fut in fs.items():
            try:
                fut.result()
                log.info("Created topic '%s'", t)
            except Exception as exc:
                log.warning("Could not create topic '%s': %s", t, exc)
    else:
        log.info("Topic '%s' already exists", topic)


def _delivery_report(err, msg) -> None:
    """Async delivery callback — called once the broker acks or rejects."""
    if err:
        log.error("Delivery failed  key=%s  error=%s", msg.key(), err)


def run() -> None:
    config = get_config()
    log.info(
        "Starting producer  brokers=%s  topic=%s  rate=%d/s  "
        "anomaly_rate=%.1f%%  anomaly_types=%s",
        BROKERS, TOPIC,
        config["events_per_second"],
        config["anomaly_rate"] * 100,
        config["anomaly_types"],
    )

    _ensure_topic(BROKERS, TOPIC)

    producer = Producer({
        "bootstrap.servers": BROKERS,
        # Batch for up to 50 ms or 500 messages before sending, reducing
        # syscall overhead at higher rates while keeping latency <100 ms.
        "queue.buffering.max.ms":  50,
        "batch.num.messages":      500,
        "compression.type":        "snappy",
        # Retry transient broker errors; combined with default acks=1 this
        # gives at-least-once delivery, acceptable for a dev stream.
        "message.send.max.retries": 5,
        "retry.backoff.ms":         200,
    })

    eps      = max(1, config["events_per_second"])
    interval = 1.0 / eps
    sent     = 0
    n_anomaly = 0

    try:
        while True:
            t0 = time.monotonic()

            tx      = make_transaction(config)
            payload = json.dumps(tx, default=str).encode("utf-8")

            producer.produce(
                TOPIC,
                key=tx["user_id"].encode("utf-8"),
                value=payload,
                callback=_delivery_report,
            )
            # Non-blocking poll: drains the delivery-report callback queue
            # without waiting for new events.
            producer.poll(0)

            sent += 1
            if tx["is_anomaly"]:
                n_anomaly += 1

            if sent % 200 == 0:
                log.info(
                    "sent=%d  anomalies=%d  observed_rate=%.2f%%",
                    sent, n_anomaly, 100.0 * n_anomaly / sent,
                )

            elapsed   = time.monotonic() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("Interrupted — flushing remaining messages…")
    finally:
        producer.flush(timeout=30)
        log.info(
            "Shutdown complete  total_sent=%d  anomalies=%d  observed_rate=%.2f%%",
            sent, n_anomaly,
            100.0 * n_anomaly / sent if sent else 0.0,
        )
