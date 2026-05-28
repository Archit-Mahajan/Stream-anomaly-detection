"""
Scorer: load trained artifacts and score individual or batch transactions.

This module is the inference interface imported by:
  • streaming/job.py  — Spark pandas_udf wraps score_batch() for micro-batches
  • api/routes.py     — optional real-time scoring via the REST API
  • model/evaluate.py — batch evaluation against the labelled holdout

SCORE CONVENTION
─────────────────
IsolationForest.decision_function() returns a value near 0 for typical
transactions and increasingly negative for anomalies.  The threshold
(ANOMALY_THRESHOLD env var, default -0.1) divides the space:

    score < threshold  →  flagged as anomaly
    score ≥ threshold  →  classified as normal

decision_function (not score_samples) is used because it is already
offset-corrected: 0 corresponds exactly to the contamination percentile of
the training set, giving a calibrated starting point before you adjust the
threshold for your precision/recall target.

CACHING
────────
load_artifacts() caches model + pipeline in module globals.  In a Spark
executor, each Python worker process calls load_artifacts() once on its first
micro-batch, then reuses the in-memory objects for all subsequent batches.
This avoids repeated disk I/O, which would dominate latency at high throughput.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `python model/scorer.py` invocation from the project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import FeaturePipeline

_DEFAULT_MODEL_DIR = Path(__file__).parent

# Module-level cache so Spark workers load artifacts only once per process
_pipeline: FeaturePipeline | None = None
_model = None
_loaded_from: Path | None = None


def load_artifacts(model_dir: Path | str | None = None):
    """Return (IsolationForest, FeaturePipeline), loading from disk if needed.

    Args:
        model_dir: directory containing isolation_forest.pkl and
                   feature_pipeline.pkl.  Defaults to the model/ package dir.
                   Passing a different path forces a fresh load (useful in tests).
    """
    global _pipeline, _model, _loaded_from

    model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

    if _model is None or _loaded_from != model_dir:
        _pipeline = joblib.load(model_dir / "feature_pipeline.pkl")
        _model    = joblib.load(model_dir / "isolation_forest.pkl")
        _loaded_from = model_dir

    return _model, _pipeline


def score_transaction(
    tx: dict,
    model=None,
    pipeline: FeaturePipeline | None = None,
) -> float:
    """Score a single transaction dict.

    Returns:
        float: anomaly score.  Lower (more negative) → more anomalous.
               Typical range is roughly [-0.5, 0.5] after calibration.
               A score below ANOMALY_THRESHOLD (default -0.1) flags the
               transaction as an anomaly.
    """
    if model is None or pipeline is None:
        model, pipeline = load_artifacts()
    X = pipeline.transform_one(tx)
    return float(model.decision_function(X)[0])


def score_batch(
    txs: list[dict] | pd.DataFrame,
    model=None,
    pipeline: FeaturePipeline | None = None,
) -> np.ndarray:
    """Score a list of transaction dicts or a DataFrame.

    Returns:
        np.ndarray of floats, shape (N,).  Same sign convention as
        score_transaction(): lower = more anomalous.

    This function is the entry point for the Spark pandas_udf:

        @pandas_udf("double")
        def score_udf(txs: pd.Series) -> pd.Series:
            dicts = txs.apply(json.loads).tolist()
            scores = score_batch(dicts)
            return pd.Series(scores)
    """
    if model is None or pipeline is None:
        model, pipeline = load_artifacts()

    df = pd.DataFrame(txs) if not isinstance(txs, pd.DataFrame) else txs
    X = pipeline.transform(df)
    return model.decision_function(X)


def is_anomaly(score: float, threshold: float | None = None) -> bool:
    """Return True if score indicates an anomaly (score < threshold)."""
    if threshold is None:
        threshold = float(os.getenv("ANOMALY_THRESHOLD", "-0.1"))
    return score < threshold


# ── CLI for quick sanity checks ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Score a transaction dict")
    parser.add_argument(
        "--tx",
        default=None,
        help='JSON string of a transaction dict.  Omit to run a built-in smoke test.',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("ANOMALY_THRESHOLD", "-0.1")),
        help="Anomaly threshold (default from ANOMALY_THRESHOLD env or -0.1)",
    )
    args = parser.parse_args()

    if args.tx:
        tx = json.loads(args.tx)
    else:
        # Smoke test: one obviously-normal and one obviously-anomalous transaction
        from producer.generator import make_transaction

        cfg_normal  = {"anomaly_rate": 0.0, "anomaly_types": [], "events_per_second": 1}
        cfg_anomaly = {"anomaly_rate": 1.0, "anomaly_types": ["amount_spike"], "events_per_second": 1}
        normal_tx  = make_transaction(cfg_normal)
        anomaly_tx = make_transaction(cfg_anomaly)

        model, pipeline = load_artifacts()
        for label, tx in [("normal", normal_tx), ("anomaly", anomaly_tx)]:
            score = score_transaction(tx, model, pipeline)
            flag  = is_anomaly(score, args.threshold)
            print(f"[{label:7s}]  amount={tx['amount']:8.2f}  "
                  f"score={score:+.4f}  flagged={flag}")
        sys.exit(0)

    model, pipeline = load_artifacts()
    score = score_transaction(tx, model, pipeline)
    flag  = is_anomaly(score, args.threshold)
    print(f"score={score:+.4f}  flagged={flag}  threshold={args.threshold}")
