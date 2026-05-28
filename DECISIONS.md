# Engineering Decisions — stream-anomaly-detector

Six decisions I made deliberately, with the trade-offs I'd defend in an interview.

---

## 1. Redpanda over Kafka + Zookeeper

**Decision:** Use Redpanda as the message broker instead of Apache Kafka.

**Why it works:**
Redpanda is a single binary that speaks the Kafka wire protocol verbatim. On a developer laptop it starts in under 2 seconds and consumes ~150 MB RAM, versus Kafka + Zookeeper which needs ~800 MB and 30–60 seconds to become healthy. Because the wire protocol is identical, no client code changes when swapping to Confluent Cloud in production — the same `confluent-kafka` Python client works without modification.

**What I gave up:**
Redpanda's ecosystem (connectors, MirrorMaker, ksqlDB) is thinner than Kafka's. Benchmark ceiling is lower on very high partition counts. For a demo that tops out at a few thousand events/second these are not real constraints.

**Interview angle:** "I would swap to Confluent Cloud for production — the producer and Spark job don't change at all because they speak the Kafka protocol, not the Redpanda API."

---

## 2. PySpark Structured Streaming over Flink or a custom consumer

**Decision:** Use PySpark's `readStream / foreachBatch` for the scoring job.

**Why it works:**
PySpark handles both batch (model training on historical data) and micro-batch streaming (online scoring) in one framework, eliminating the need for two separate runtimes. `foreachBatch` lets me fan-out to two Postgres tables and a Kafka topic in a single callback. Checkpointing to local disk gives resume-from-offset semantics with near-exactly-once guarantees for free. Most interviewers know Spark; Flink's Python API (PyFlink) is less mature and less commonly understood.

**What I gave up:**
Spark's mini-batch model adds inherent latency (5 s trigger here). True low-latency streaming (sub-second) would need Flink with its record-at-a-time processing model. The `toPandas()` call inside `foreachBatch` is single-node; at scale I would replace it with a `pandas_udf` so scoring distributes across Spark executors.

**Interview angle:** "The production scaling path is one change: replace `toPandas() + score_batch()` with a `pandas_udf`. `foreachBatch` stays for the fan-out logic."

---

## 3. Isolation Forest over deep-learning anomaly detectors

**Decision:** Use scikit-learn's `IsolationForest` as the scoring model.

**Why it works:**
IsolationForest is O(n log n) and not distance-based, so it degrades gracefully in high dimensions — unlike LOF or DBSCAN which suffer from the curse of dimensionality. Critically, `decision_function()` returns a *continuous score* (not just a binary label), which lets the dashboard slide the threshold dynamically without retraining. The model trains on 50 000 synthetic rows in under 5 seconds on a CPU; retraining on a rolling window is feasible in production.

**What I gave up:**
IsolationForest captures linear anomalies well (spikes, rare categories) but misses sequential patterns — e.g., "10 transactions in 60 seconds" requires stateful sessionisation, which would be a Spark windowed aggregation. Autoencoders can learn complex non-linear patterns but take minutes to train and require GPU infrastructure.

**Measured performance on holdout (10 000 events, 2.84% anomaly rate):**

| Metric | Value |
|--------|-------|
| Precision | 0.914 |
| Recall | 0.863 |
| F1 | 0.888 |
| AUPRC | 0.954 |
| geo_jump recall | 1.000 |
| odd_hour_burst recall | 1.000 |
| amount_spike recall | 0.567 |

Amount-spike recall is lower because a small fraction of spikes land only 2–3× above normal (within the training distribution's tail). Raising the multiplier in `generator.py` or adding a rolling per-user z-score feature would fix this.

---

## 4. foreachBatch fan-out instead of multiple streaming sinks

**Decision:** Write both Postgres tables and the Kafka alerts topic inside a single `foreachBatch` callback instead of using three independent `DataStreamWriter` sinks.

**Why it works:**
Three independent sinks would spawn three separate streaming queries, each with its own offset tracking and checkpoint. A partial failure in the second query (Postgres write) would not roll back the first (Kafka publish), creating phantom alerts that reference rows that don't exist in Postgres. `foreachBatch` runs atomically inside a single Python callback: Postgres is committed first, then — and only then — alerts are published to Kafka. A failure in the Postgres write prevents any alert from being published, which is the safe failure direction for a fraud system (miss an alert rather than create a spurious one).

**What I gave up:**
The batch callback runs on the Spark driver, so it cannot be parallelised across executors. This is fine for the ~50 row/batch workload here; at 10 000 rows/batch I would split the sink logic into a `pandas_udf` for scoring + a JDBC foreachBatch for persistence.

---

## 5. Postgres as the query sink instead of a time-series store

**Decision:** Persist all scored transactions and alerts in Postgres with two B-tree indexes.

**Why it works:**
Postgres gives structured queries (filter by anomaly type, score range, time window), supports ACID transactions for the idempotent `ON CONFLICT DO NOTHING` inserts, and is universally understood. The `alerted_at DESC` index on `alerts` makes the SSE polling query (`WHERE alerted_at > $last_seen LIMIT 30`) a fast index scan. The schema is already TimescaleDB-compatible: converting `event_time` into a hypertable is a one-line `SELECT create_hypertable(...)` call, giving automatic time-partitioning and compression if query latency degrades at scale.

**What I gave up:**
Postgres is not optimised for append-heavy time-series workloads at high volume. ClickHouse or TimescaleDB would give 10–50× faster analytical scans on millions of rows. For a demo generating ~10 events/second (36 000 rows/hour) Postgres is comfortably within its sweet spot.

---

## 6. SSE over WebSockets for the live dashboard

**Decision:** Use HTTP Server-Sent Events (SSE) for the dashboard's live feed rather than WebSockets.

**Why it works:**
The dashboard is unidirectional: the server pushes new alerts; the client never sends data back. SSE is a plain HTTP GET that stays open, proxies trivially through any load balancer or CDN (no Upgrade header negotiation), and reconnects automatically on network interruption. The `EventSource` browser API is three lines of JavaScript. WebSockets add bidirectional framing, a custom upgrade handshake, and stateful connection management — none of which are needed here.

**What I gave up:**
SSE is HTTP/1.1 by default; HTTP/2 multiplexing eliminates the one-connection-per-stream limit for free, but most reverse proxies still treat SSE as a special case. For a use case that genuinely needs bidirectional messaging (e.g., the dashboard sending threshold changes back to the API), WebSockets would be the right choice.

**Note on the current implementation:** The dashboard polls `/alerts` and `/stats` via `setInterval` rather than a true SSE stream, which is simpler and equally correct for a 2-second update cadence. A full SSE endpoint (`GET /stream`) is wired in the API (`routes.py`) and ready to replace polling when sub-second latency is needed.
