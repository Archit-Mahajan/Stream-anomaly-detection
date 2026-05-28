# producer/

Synthetic Kafka producer that generates a realistic financial transaction stream
and publishes each event as a JSON message to the **`transactions`** topic on
Redpanda.

---

## Files

| File | Purpose |
|------|---------|
| `generator.py` | Transaction factory — log-normal amounts, realistic hour distribution, configurable anomaly injection |
| `publisher.py` | Confluent-Kafka producer loop — throttling, delivery callbacks, auto-topic creation |
| `__main__.py`  | Entry-point so the package runs with `python -m producer` |

---

## Prerequisites

```bash
# From the repo root — starts Redpanda (and Postgres)
docker compose up -d redpanda

# Install Python dependencies (recommended: activate a virtualenv first)
pip install -r requirements.txt
```

---

## Running

```bash
# Defaults: 10 events/s, 2.5 % anomaly rate, all three anomaly types
python -m producer

# Slow to 1 event/s for easy manual inspection
EVENTS_PER_SECOND=1 python -m producer

# Only amount-spike anomalies at 5 %
ANOMALY_RATE=0.05 ANOMALY_TYPES=amount_spike python -m producer

# All knobs
REDPANDA_BROKERS=localhost:9092 \
  KAFKA_TOPIC=transactions \
  EVENTS_PER_SECOND=50 \
  ANOMALY_RATE=0.03 \
  ANOMALY_TYPES=amount_spike,geo_jump,odd_hour_burst \
  python -m producer
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDPANDA_BROKERS` | `localhost:9092` | Kafka/Redpanda broker address |
| `KAFKA_TOPIC` | `transactions` | Topic to publish to |
| `EVENTS_PER_SECOND` | `10` | Target publish rate (use 1 on a slow laptop) |
| `ANOMALY_RATE` | `0.025` | Fraction of events injected as anomalies (0.0–1.0) |
| `ANOMALY_TYPES` | `amount_spike,geo_jump,odd_hour_burst` | Comma-separated anomaly types to enable |

---

## Verifying messages land in the topic

**Option 1 — rpk CLI (bundled inside the Redpanda container)**

```bash
docker exec -it redpanda rpk topic consume transactions --num 5
```

Sample output (one JSON line per message):

```json
{"transaction_id":"3f2a...","user_id":"u_0042","amount":73.41,"merchant_category":"grocery","timestamp":"2026-05-28T14:22:01Z","country":"US","is_anomaly":false,"anomaly_type":null}
```

**Option 2 — Redpanda Console (browser UI)**

Navigate to [http://localhost:8080](http://localhost:8080) → **Topics** →
**transactions** → **Messages**.  You should see live messages arriving.

**Option 3 — curl via Pandaproxy (no extra tooling)**

```bash
# 1. Create consumer instance
curl -s -X POST http://localhost:8082/consumers/test-group \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -d '{"name":"ci","format":"json","auto.offset.reset":"earliest"}'

# 2. Subscribe to topic
curl -s -X POST http://localhost:8082/consumers/test-group/instances/ci/subscription \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -d '{"topics":["transactions"]}'

# 3. Consume (call repeatedly to page through messages)
curl -s http://localhost:8082/consumers/test-group/instances/ci/records \
  -H "Accept: application/vnd.kafka.json.v2+json"
```

**Checking the anomaly injection rate**

```bash
# Count true/false across 1 000 messages (requires jq and rpk on the host)
docker exec -it redpanda rpk topic consume transactions --num 1000 \
  | jq -r '.value | fromjson | .is_anomaly' \
  | sort | uniq -c
```

At the default `ANOMALY_RATE=0.025` you should see roughly **25 `true`** and
**975 `false`**.

---

## Message schema

```json
{
  "transaction_id":    "uuid-v4 string",
  "user_id":           "u_NNNN",
  "amount":            73.41,
  "merchant_category": "grocery",
  "timestamp":         "2026-05-28T14:22:01Z",
  "country":           "US",
  "is_anomaly":        false,
  "anomaly_type":      null
}
```

`anomaly_type` is one of `"amount_spike"`, `"geo_jump"`, `"odd_hour_burst"`,
or `null` for normal transactions.

---

## Anomaly types

| Type | What is mutated | Why the model should detect it |
|------|----------------|-------------------------------|
| `amount_spike` | amount ×10–50 of normal | Sits well past the 99th pct of the log-normal distribution (~$479 at σ=0.8) |
| `geo_jump` | country → JP / SG / AU / ZA (≥8 time zones from every home country) | IsolationForest isolates rare country values; Spark velocity window confirms impossible transit time |
| `odd_hour_burst` | timestamp hour → 02–04 | Hour weights at 2–4 AM are 1/10 of peak; sin/cos encoding places this far from the normal spending cluster |

---

## Why labeled anomalies matter (quick reference)

Without `is_anomaly` we can only eyeball individual scored transactions.
With it we can compute **precision** (what fraction of our alerts are real) and
**recall** (what fraction of real anomalies we catch), plot a full PR curve,
and measure model drift over time — essential for deciding when to retrain.
