"""
Synthetic transaction generator for the stream-anomaly-detector pipeline.

WHY LABELED ANOMALIES MATTER FOR EVALUATION
────────────────────────────────────────────
An IsolationForest (or any unsupervised scorer) produces a continuous anomaly
score for every event.  Without ground-truth labels we can only inspect
individual flagged transactions by hand — we have no objective way to compare
two model versions or choose a score threshold.

With ground-truth labels (is_anomaly, anomaly_type) we can:

  • Sweep the score threshold and plot a Precision-Recall curve, then pick the
    operating point that best balances false alarms vs. missed fraud.
  • Compute AUPRC (area under the PR curve) — a single number summarising
    quality across all thresholds, robust to severe class imbalance (~2-3%
    anomaly rate means accuracy is useless as a metric).
  • Detect model drift: if precision@k drops significantly over a rolling
    evaluation window the data distribution has shifted and we should retrain.
  • Measure type-specific recall: "we catch 90 % of amount_spikes but only
    60 % of geo_jumps" tells the team which engineered features need work.

In production, ground-truth labels come from human fraud analysts confirming or
dismissing flagged transactions.  Here we inject them synthetically so that
Phases 2-3 (model training + streaming scoring) can be validated end-to-end on
a laptop without a labelled fraud dataset.
"""

import os
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ─── Static pools ─────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery", "gas", "restaurant", "entertainment",
    "travel", "online", "atm", "pharmacy",
]

# User home countries — all western-hemisphere / Europe so geo-jump anomalies
# are unambiguous (a US user transacting in Singapore is suspicious; a US user
# transacting in Canada is normal).
_HOME_COUNTRIES = ["US", "CA", "GB", "DE", "FR"]

# Countries ≥ 8 time zones from every home country above.  When a geo_jump
# anomaly fires, the destination is drawn from this pool.  A real velocity
# check in Spark would confirm the previous transaction was in the home country
# seconds earlier, making this "physically impossible" to reach by plane.
_FAR_COUNTRIES = ["JP", "SG", "AU", "ZA"]

# Normal amount distribution: log-normal so the bulk of transactions look like
# retail ($30–$200) with a long right tail for high-value purchases.
#   exp(4.0)        ≈ $55  median
#   exp(4.0 + 0.8)  ≈ $123  84th-pct
#   exp(4.0 + 1.6)  ≈ $333  97th-pct
#   99th-pct        ≈ $479
_AMOUNT_MU    = 4.0
_AMOUNT_SIGMA = 0.8

# Hour-of-day weights reproduce a realistic retail spending curve:
# low overnight (0-6), building through morning, peak at 10-15, tapering off.
# Phase 2 feature engineering will encode this via sin/cos cyclical encoding
# so IsolationForest does not see a discontinuity at midnight.
_HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 10, 10, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 1]
_HOURS        = list(range(24))

# ─── User pool ────────────────────────────────────────────────────────────────

def _build_user_pool(n: int = 500) -> dict[str, str]:
    """Return {user_id: home_country} for n deterministic synthetic users."""
    rng = random.Random(42)  # fixed seed → reproducible pool across runs
    return {f"u_{i:04d}": rng.choice(_HOME_COUNTRIES) for i in range(n)}


_USER_POOL: dict[str, str] = _build_user_pool()
_USER_IDS:  list[str]      = list(_USER_POOL.keys())

# ─── Configuration ────────────────────────────────────────────────────────────

_VALID_ANOMALY_TYPES = frozenset({"amount_spike", "geo_jump", "odd_hour_burst"})


def _parse_anomaly_types(raw: str) -> list[str]:
    types = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = set(types) - _VALID_ANOMALY_TYPES
    if unknown:
        raise ValueError(
            f"Unknown ANOMALY_TYPES: {unknown!r}.  Valid: {_VALID_ANOMALY_TYPES}"
        )
    return types


def get_config() -> dict:
    """Read producer configuration from environment variables.

    ANOMALY_RATE      float  Fraction of events injected as anomalies. Default 0.025.
    ANOMALY_TYPES     str    Comma-separated subset of the three anomaly types.
    EVENTS_PER_SECOND int    Target publish rate.  Default 10.
    """
    return {
        "anomaly_rate": float(os.getenv("ANOMALY_RATE", "0.025")),
        "anomaly_types": _parse_anomaly_types(
            os.getenv("ANOMALY_TYPES", "amount_spike,geo_jump,odd_hour_burst")
        ),
        "events_per_second": int(os.getenv("EVENTS_PER_SECOND", "10")),
    }


# ─── Transaction factory ──────────────────────────────────────────────────────

def _sample_amount() -> float:
    return round(float(np.random.lognormal(_AMOUNT_MU, _AMOUNT_SIGMA)), 2)


def make_transaction(config: dict, *, now: Optional[datetime] = None) -> dict:
    """Generate one synthetic transaction, optionally injecting a labeled anomaly.

    The returned dict always includes:
      is_anomaly  (bool)       — ground-truth label consumed by the evaluator
      anomaly_type (str|None)  — granular type for per-class precision/recall

    Downstream consumers may strip these before inserting to the public-facing
    transactions table, but must preserve them in the anomalies table so
    evaluation metrics can be computed.
    """
    user_id      = random.choice(_USER_IDS)
    home_country = _USER_POOL[user_id]
    ts           = now or datetime.now(timezone.utc)

    amount       = _sample_amount()
    country      = home_country
    anomaly_type: Optional[str] = None

    # ── Anomaly injection ────────────────────────────────────────────────────
    if config["anomaly_types"] and random.random() < config["anomaly_rate"]:
        chosen = random.choice(config["anomaly_types"])

        if chosen == "amount_spike":
            # ×10–50 of normal; turns a median $55 purchase into $550–$2 750,
            # well past the 99th-pct ($479) of the log-normal distribution.
            amount = round(amount * random.uniform(10.0, 50.0), 2)
            anomaly_type = "amount_spike"

        elif chosen == "geo_jump":
            # Transaction appears in a country ≥8 time zones from the user's
            # home.  IsolationForest detects this via the country feature;
            # Spark can additionally compute velocity (seconds since same user's
            # last transaction) to flag impossible transit times.
            country = random.choice(_FAR_COUNTRIES)
            anomaly_type = "geo_jump"

        elif chosen == "odd_hour_burst":
            # Burst of activity in the 02–04 window (_HOUR_WEIGHTS value = 1,
            # vs 10 at peak hours).  Sin/cos cyclical encoding places this hour
            # far from the normal spending cluster without ordinal discontinuity.
            hour = random.randint(2, 4)
            ts   = ts.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
            )
            anomaly_type = "odd_hour_burst"

    return {
        "transaction_id":    str(uuid.uuid4()),
        "user_id":           user_id,
        "amount":            amount,
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "timestamp":         ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "country":           country,
        "is_anomaly":        anomaly_type is not None,
        "anomaly_type":      anomaly_type,  # None | "amount_spike" | "geo_jump" | "odd_hour_burst"
    }
