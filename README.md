# stream-anomaly-detector

Real-time financial transaction anomaly detection pipeline — Redpanda → PySpark → IsolationForest → Postgres → FastAPI → live dashboard.

---

## Architecture

```mermaid
flowchart LR
    subgraph Producer["producer/ (Python)"]
        G[generator.py\nsynthetic txs\n~10 tx/s]
        P[publisher.py\nconfluent-kafka]
        G --> P
    end

    subgraph Broker["Redpanda (Docker)"]
        T[(topic:\ntransactions)]
        A[(topic:\nalerts)]
    end

    subgraph Spark["streaming/ (PySpark)"]
        R[readStream\nKafka source]
        S[score_batch\nIsolationForest]
        FB[foreachBatch\nfan-out sink]
        R --> S --> FB
    end

    subgraph DB["Postgres (Docker)"]
        TX[(transactions)]
        AL[(alerts)]
    end

    subgraph API["api/ (FastAPI)"]
        EP["/alerts\n/stats\n/score\n/health"]
    end

    subgraph UI["dashboard/"]
        D[index.html\npolls every 2s]
    end

    P -->|JSON| T
    T -->|Kafka wire| R
    FB -->|psycopg2\nON CONFLICT DO NOTHING| TX
    FB -->|psycopg2| AL
    FB -->|confluent-kafka| A
    TX -->|SQL| EP
    AL -->|SQL| EP
    EP -->|HTTP/JSON| D
```

---

## Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Broker | Redpanda | Kafka-API compatible, single binary, <2 s start, ~150 MB RAM vs ~800 MB for Kafka+ZK |
| Stream processor | PySpark Structured Streaming | Mature Python API, handles batch training + streaming scoring, exactly-once via checkpointing |
| Model | scikit-learn IsolationForest | Continuous score (not just a label), O(n log n), fast enough to retrain on a rolling window |
| API | FastAPI | Async-native, auto OpenAPI docs, Pydantic validation, trivially testable |
| Sink | Postgres | Structured queries, ACID, TimescaleDB upgrade path for time-series at scale |
| Kafka client | confluent-kafka | Wraps librdkafka (C), ~10× faster than kafka-python, production-safe rebalancing |
| Containerisation | Docker Compose | One-command data plane; no Kubernetes overhead for a single-node demo |

See [DECISIONS.md](DECISIONS.md) for the full trade-off analysis and interview rationale.

---

## Model Performance

Trained on 50 000 synthetic transactions (2.5% anomaly rate).
Evaluated on a labelled 10 000-event holdout:

| Metric | Value |
|--------|-------|
| Precision | **0.914** |
| Recall | **0.863** |
| F1 | **0.888** |
| AUPRC | **0.954** |
| geo_jump recall | 1.000 |
| odd_hour_burst recall | 1.000 |
| amount_spike recall | 0.567 |

Features: `amount_log_zscore`, `hour_sin`, `hour_cos`, `merchant_freq`, `is_home_country`

---

## Prerequisites

- Docker + Docker Compose v2
- Python 3.10+

---

## Quickstart (Makefile)

```bash
# 1. Clone and enter the project
cd "Stream anomaly"

# 2. Create and activate a virtual environment
make venv
source .venv/bin/activate

# 3. Install Python dependencies
make install

# 4. Copy environment config
cp .env.example .env

# 5. Start everything (Docker + train + all services in background)
make all
```

Then open **http://localhost:8000/dashboard**.

Logs are tailed at `/tmp/producer.log`, `/tmp/streaming.log`, `/tmp/api.log`.

---

## Manual Run Order

Open five terminal tabs. Run each from the project root with the venv active.

### 0 — Start Docker services

```bash
make up
make wait        # blocks until Redpanda + Postgres are healthy
```

### 1 — Train the model

```bash
make train
# Writes model/isolation_forest.pkl, model/feature_pipeline.pkl, model/metrics.json
```

### 2 — Start the producer

```bash
make produce     # or: ANOMALY_RATE=0.05 EVENTS_PER_SECOND=20 python -m producer
```

Publishes synthetic transactions to Redpanda topic `transactions` at ~10 tx/s.

### 3 — Start the Spark streaming job

```bash
make stream
# First run downloads the Spark-Kafka connector from Maven (~30 s); subsequent runs instant.
```

Reads from `transactions`, scores each event, writes to Postgres + Kafka `alerts` topic.

### 4 — Start the API

```bash
make api         # uvicorn api.main:app --reload --port 8000
```

### 5 — Open the dashboard

```
http://localhost:8000/dashboard
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/alerts` | Recent anomaly alerts; filterable by `limit`, `offset`, `since`, `anomaly_type`, `max_score` |
| POST | `/score` | Score a single transaction on demand (no DB required) |
| GET | `/stats` | Rolling-window counts, anomaly rate, model metrics; `?window_minutes=10` |
| GET | `/health` | Liveness + DB connectivity |
| GET | `/docs` | Auto-generated OpenAPI UI (Swagger) |
| GET | `/dashboard` | Live HTML dashboard |

Quick smoke tests:

```bash
# Health
curl -s localhost:8000/health | python -m json.tool

# 10-minute stats
curl -s "localhost:8000/stats?window_minutes=10" | python -m json.tool

# Last 5 alerts
curl -s "localhost:8000/alerts?limit=5" | python -m json.tool

# Score a suspicious transaction on demand
curl -s -X POST localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 9500, "merchant_category": "atm", "country": "SG"}' \
  | python -m json.tool

# Filter: only geo_jump alerts
curl -s "localhost:8000/alerts?anomaly_type=geo_jump&limit=20" | python -m json.tool
```

---

## Tests

```bash
make test        # runs tests/ with pytest; no broker or DB required
```

Three test modules:

| Module | Coverage |
|--------|----------|
| `tests/test_producer.py` | Transaction shape, field contract, all three anomaly-injection modes |
| `tests/test_model.py` | Artifact load, score range, normal vs. anomaly ordering, batch scoring, metrics sanity |
| `tests/test_api.py` | All four endpoints return 200, schema validation, edge cases; DB patched in-process |

---

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Redpanda (Kafka) | 9092 | Producer / Spark wire protocol |
| Redpanda Console | 8080 | Browser UI for topic inspection |
| Postgres | 5432 | Transaction and alert storage |
| FastAPI + dashboard | 8000 | REST API, OpenAPI docs, live dashboard |

---

## Environment Variables

Copy `.env.example` to `.env` (never commit):

```bash
REDPANDA_BROKERS=localhost:9092
POSTGRES_DSN=postgresql://anomaly:anomaly@localhost:5432/anomaly_db
ANOMALY_THRESHOLD=-0.1        # IsolationForest scores below this are flagged
MODEL_PATH=model/isolation_forest.pkl
ANOMALY_RATE=0.025            # fraction of events injected as anomalies
EVENTS_PER_SECOND=10          # producer publish rate
```

---

## Project Layout

```
.
├── producer/           Phase 1 — synthetic transaction generator + Kafka publisher
│   ├── generator.py    transaction factory (amount_spike / geo_jump / odd_hour_burst)
│   ├── publisher.py    confluent-kafka producer loop
│   └── __main__.py     entry point: python -m producer
├── model/              Phase 2 — IsolationForest training, features, scorer, evaluation
│   ├── features.py     FeaturePipeline (log-amount, cyclical time, merchant_freq, is_home)
│   ├── train.py        fit + serialise model + pipeline
│   ├── scorer.py       load_artifacts(), score_transaction(), score_batch()
│   ├── evaluate.py     precision / recall / AUPRC on holdout → metrics.json
│   ├── isolation_forest.pkl
│   ├── feature_pipeline.pkl
│   └── metrics.json
├── streaming/          Phase 3 — PySpark Structured Streaming job
│   ├── job.py          Kafka source → foreachBatch → Postgres + Kafka alerts
│   ├── schema.sql      DDL for transactions and alerts tables
│   └── __main__.py     entry point: python -m streaming
├── api/                Phase 4 — FastAPI service
│   ├── main.py         app setup, CORS, static dashboard mount
│   ├── routes.py       /alerts, /score, /stats, /health
│   ├── db.py           psycopg2 queries
│   └── schemas.py      Pydantic request / response models
├── dashboard/          Phase 5 — single-page HTML + vanilla JS live dashboard
│   └── index.html
├── tests/              Offline test suite (no broker / DB required)
│   ├── test_producer.py
│   ├── test_model.py
│   └── test_api.py
├── Makefile            One-command targets: up, train, produce, stream, api, test, all
├── docker-compose.yml  Redpanda + Postgres data plane
├── requirements.txt    Python dependencies with inline rationale
├── .env.example        Environment variable template
├── DECISIONS.md        Six key engineering trade-offs (interview cheat-sheet)
└── CLAUDE.md           Architecture decisions, session context
```

---

## Dashboard Screenshot

_Start the full pipeline (`make all`) then open http://localhost:8000/dashboard to see:_

- **Live stats card** — transaction count, alert count, anomaly rate (10-min rolling window)
- **Model metrics card** — F1, precision, recall, AUPRC, per-type recall from holdout evaluation
- **Anomaly rate chart** — 6-minute canvas sparkline, sampled every 5 seconds
- **Alerts table** — last 30 flagged events with amount, score, type badge, and TX ID; new rows highlighted green

---

## Scaling the Pipeline

| Bottleneck | Current | Production path |
|------------|---------|-----------------|
| Scoring throughput | `toPandas()` on driver | Replace with `pandas_udf` — distributes across Spark executors |
| Postgres write latency | `psycopg2` sync write per batch | asyncpg connection pool + `COPY` for bulk inserts |
| Time-series query performance | B-tree index on `event_time` | `SELECT create_hypertable(...)` → TimescaleDB automatic partitioning |
| Model staleness | Static serialised artifact | Trigger retrain when rolling AUPRC drops below threshold; serve new artifact via MLflow model registry |
| Broker scalability | Single-node Redpanda | Confluent Cloud multi-partition cluster; no client code changes |
