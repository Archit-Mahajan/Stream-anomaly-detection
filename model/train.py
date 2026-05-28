"""
Train the IsolationForest anomaly detector.

Run from the project root:
    python -m model.train                            # generate 50k synthetic records
    python -m model.train --sample-file model/sample_transactions.jsonl
    python -m model.train --n-records 100000 --contamination 0.025

To capture real Redpanda data for training:
    rpk topic consume transactions --brokers localhost:9092 \\
        --num 50000 > model/sample_transactions.jsonl

WHY ISOLATION FOREST FOR UNSUPERVISED STREAMING ANOMALY DETECTION
──────────────────────────────────────────────────────────────────
IsolationForest randomly partitions the feature space using an ensemble of
isolation trees.  Anomalous points — rare and structurally distinct — need
fewer random splits to become isolated from the rest of the data.  The
anomaly score is the normalised mean path length across all trees; shorter
path → more anomalous → lower decision_function score → flags as outlier.

KEY ADVANTAGES FOR THIS USE CASE

1. O(n log n) training, O(log n) per-record inference.
   Trains on 50k records in ~2 seconds on a laptop CPU.  A 100-tree forest
   scores each incoming Kafka event in microseconds — fast enough to run
   synchronously inside a Spark micro-batch without becoming the bottleneck.
   This also means we can retrain on a rolling window (last N transactions)
   every few minutes to adapt to concept drift, without pausing the pipeline.

2. Not distance-based — scales in high dimensions.
   LOF, kNN-based methods, and One-Class SVM (RBF kernel) all rely on
   pairwise distances that converge in high-dimensional spaces (the curse of
   dimensionality makes all points equidistant).  IsolationForest uses random
   axis-aligned splits, so adding features costs only training time, not
   detection quality.

3. Continuous score, not just a binary label.
   decision_function returns a value ∈ (−∞, +∞) centred near 0.  The ops
   team can tune ANOMALY_THRESHOLD (env var) at runtime to trade precision
   vs. recall without retraining.  An AUPRC curve lets you pick the operating
   point for your false-alarm budget.

4. Truly unsupervised.
   Labelled anomalies are rare and expensive to obtain in fraud detection.
   IsolationForest trains on normal data alone and uses the `contamination`
   hyperparameter to set the decision boundary at the expected outlier
   percentile.  We happen to have labels in this synthetic pipeline, but the
   model does NOT use them during training — labels are reserved for the
   evaluation holdout.

TRADEOFFS vs ALTERNATIVES

vs One-Class SVM (OC-SVM)
  OC-SVM fits a maximum-margin hypersphere in kernel (RBF) space.  The
  resulting boundary can be highly non-linear and tight — good for clean,
  well-structured normal data.
  Disadvantages:
    • Training complexity O(n²) to O(n³): fitting 50k samples with RBF takes
      minutes; fitting 500k is intractable on a laptop.
    • Cannot retrain on a rolling window in real-time.
    • Sensitive to the ν (nu) hyperparameter and kernel bandwidth γ; both
      require cross-validation.
    • Sklearn's OC-SVM supports predict() on new data, but the decision surface
      is expensive to compute for large test sets too.
  IsolationForest is strictly faster and more hyperparameter-robust for our
  5-feature, ~50k record regime.

vs Autoencoder (AE)
  A deep autoencoder learns to compress and reconstruct normal data; anomalies
  produce high reconstruction error.  In principle it can model complex
  non-linear distributions that IsolationForest misses.
  Disadvantages:
    • Requires PyTorch or TensorFlow — heavy dependency for a streaming job.
    • Training takes minutes to hours; incompatible with rolling retraining.
    • Reconstruction error threshold is dataset-specific and hard to calibrate
      without labels.
    • Black-box: cannot explain WHY a transaction is anomalous (path length in
      an isolation tree is at least partially interpretable — a short path means
      the amount or hour was so unusual it was isolated immediately).
    • Overkill for 5 engineered numerical features; autoencoders shine on
      high-dimensional raw inputs (images, sequences, embeddings).
  The right place for an autoencoder in this pipeline would be if we added raw
  text features (merchant name free text) or transaction sequences per user.

vs Local Outlier Factor (LOF)
  LOF computes a local density ratio: points in sparse regions relative to
  their neighbours are anomalies.
  Disadvantages:
    • No native predict() for new points (sklearn raises NotImplementedError).
      Every inference call must recompute distances against the full training
      set: O(n × d) per new record — completely incompatible with streaming.
    • Memory: must retain all training points in RAM (n × d floats).
    • Degrades in high dimensions for the same reason as OC-SVM (distance
      concentration).
  LOF is useful for offline batch outlier detection in a static dataset;
  not for a live Kafka consumer scoring 1000 events/second.

CONTAMINATION HYPERPARAMETER
─────────────────────────────
We set contamination=0.025 to match the generator's ANOMALY_RATE (2.5 %).
This tells IsolationForest to set its decision_function threshold such that
2.5 % of training samples are labelled as outliers.  In production without
labels, set contamination='auto' (sklearn default = 0.1) or estimate the rate
from fraud analyst reports.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Allow running as `python model/train.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import FeaturePipeline

MODEL_DIR = Path(__file__).parent


def _load_or_generate(sample_file: str | None, n_records: int) -> pd.DataFrame:
    if sample_file and Path(sample_file).exists():
        print(f"  Loading transactions from {sample_file}")
        return pd.read_json(sample_file, lines=True)

    if sample_file:
        print(f"  Sample file {sample_file!r} not found — falling back to synthetic data")

    print(f"  Generating {n_records:,} synthetic transactions...")
    from producer.generator import make_transaction, get_config
    cfg = get_config()
    rows = [make_transaction(cfg) for _ in range(n_records)]
    return pd.DataFrame(rows)


def train(
    df: pd.DataFrame,
    contamination: float,
    n_estimators: int = 100,
) -> tuple[FeaturePipeline, IsolationForest]:
    """Fit feature pipeline and IsolationForest on the given DataFrame."""
    pipeline = FeaturePipeline()
    pipeline.fit(df)
    X = pipeline.transform(df)

    model = IsolationForest(
        n_estimators=n_estimators,
        # 'auto' uses min(256, n_samples) for max_samples, balancing
        # diversity across trees vs. enough points to learn local structure.
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,  # use all CPU cores; each tree is independent
    )
    model.fit(X)
    return pipeline, model


def main():
    parser = argparse.ArgumentParser(
        description="Train IsolationForest anomaly detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-file",
        default=None,
        help="JSONL of historical transactions (e.g. from rpk consume). "
             "If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=50_000,
        help="Synthetic records to generate when no sample file is provided (default 50000)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.025,
        help="Expected anomaly rate in training data (default 0.025 = 2.5%%)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of isolation trees (default 100; 200 gives ~1%% improvement "
             "at 2× cost)",
    )
    args = parser.parse_args()

    print("\n── Loading data ──────────────────────────────────────────")
    df = _load_or_generate(args.sample_file, args.n_records)
    n_anomalies = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else "?"
    print(f"  Records: {len(df):,}   Labelled anomalies: {n_anomalies}")

    print("\n── Training ──────────────────────────────────────────────")
    pipeline, model = train(df, args.contamination, args.n_estimators)
    print(f"  IsolationForest: {args.n_estimators} trees  "
          f"contamination={args.contamination}")

    print("\n── Saving artifacts ──────────────────────────────────────")
    pipeline_path = MODEL_DIR / "feature_pipeline.pkl"
    model_path    = MODEL_DIR / "isolation_forest.pkl"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(model,    model_path)
    print(f"  {pipeline_path}")
    print(f"  {model_path}")

    # Sanity check: flagged rate on training set should approximate contamination
    X = pipeline.transform(df)
    preds = model.predict(X)           # +1 = normal, -1 = anomaly
    flagged_rate = float((preds == -1).mean())
    print(f"\n  Training set flagged rate: {flagged_rate:.3f}  "
          f"(target ≈ {args.contamination:.3f})")

    # Save a brief training summary alongside the model
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_records": len(df),
        "n_anomalies_in_training": n_anomalies,
        "contamination": args.contamination,
        "n_estimators": args.n_estimators,
        "flagged_rate_training": round(flagged_rate, 4),
        "features": list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else ["amount_log_zscore", "hour_sin", "hour_cos", "merchant_freq", "is_home_country"],
    }
    summary_path = MODEL_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path}")
    print("\nTraining complete.\n")


if __name__ == "__main__":
    main()
