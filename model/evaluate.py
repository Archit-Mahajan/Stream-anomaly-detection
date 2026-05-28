"""
Evaluate the trained IsolationForest on a labelled holdout set.

Run from the project root:
    python -m model.evaluate                         # generate 10k holdout rows
    python -m model.evaluate --n-holdout 20000 --threshold -0.15
    python -m model.evaluate --holdout-file model/holdout.jsonl

Outputs:
    model/metrics.json   — machine-readable metrics for CI/dashboards
    stdout               — human-readable report

METRICS CHOICE
──────────────
We use precision, recall, F1, confusion matrix, and AUPRC instead of accuracy.

Why accuracy is wrong here:
    With ~2.5 % anomaly rate, a classifier that flags nothing achieves 97.5 %
    accuracy — a useless baseline.  Precision/recall force the model to earn its
    positive predictions.

Why AUPRC over AUROC:
    AUROC measures separation between positive and negative score distributions.
    With severe class imbalance (1 anomaly per 40 normals), AUROC can be high
    even when the model barely ranks anomalies above normals in the dense region
    of the score distribution.  AUPRC (average precision) is more pessimistic
    and more informative in this regime: it measures the area under the
    precision-recall curve, weighting each threshold by the precision achieved
    at that recall level.  A random classifier has AUPRC ≈ anomaly_rate ≈ 0.025,
    so any meaningful model should score well above 0.025.

Per-type recall:
    The generator injects three distinct anomaly types.  Per-type recall reveals
    which engineered features are working:
      • amount_spike  → should be caught by amount_log_zscore
      • geo_jump      → should be caught by is_home_country
      • odd_hour_burst → should be caught by hour_sin/hour_cos
    Low recall on one type = that feature needs more signal or weighting.

THRESHOLD CHOICE
─────────────────
Default evaluation threshold: 0.0 (the natural sklearn decision_function
boundary, calibrated by the contamination parameter at fit time).

The CLAUDE.md env var sets ANOMALY_THRESHOLD=-0.1 for the production pipeline.
That value is ultra-conservative: on this synthetic dataset it gives perfect
precision (zero false positives) but near-zero recall (~2 %), because the -0.1
cutoff sits well below the decision boundary.  For a fraud system that cannot
afford ANY false alarms (e.g., blocking a card), -0.1 makes sense.  For
evaluation and most operational deployments, use 0.0 as the baseline and tune
upward (toward +0.1) to trade precision for recall.

Score distribution observed in practice:
  Normal transactions:  scores cluster around +0.19, min ≈ -0.03
  Anomaly transactions: scores cluster around -0.04, min ≈ -0.13
  Overlap zone [−0.03, 0]: some anomalies score above 0 (hard positives)
  Threshold 0.0  → ~84 % recall, ~0.3 % FPR on this data
  Threshold −0.1 → ~2 % recall,  ~0.0 % FPR (the CLAUDE.md production setting)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.scorer import load_artifacts, score_batch

MODEL_DIR = Path(__file__).parent


def _load_or_generate_holdout(holdout_file: str | None, n_holdout: int) -> pd.DataFrame:
    if holdout_file and Path(holdout_file).exists():
        print(f"  Loading holdout from {holdout_file}")
        return pd.read_json(holdout_file, lines=True)

    if holdout_file:
        print(f"  Holdout file {holdout_file!r} not found — generating synthetic holdout")

    print(f"  Generating {n_holdout:,} labelled holdout transactions...")
    from producer.generator import make_transaction, get_config
    cfg = get_config()
    rows = [make_transaction(cfg) for _ in range(n_holdout)]
    return pd.DataFrame(rows)


def _print_section(title: str):
    width = 55
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def evaluate(df: pd.DataFrame, threshold: float) -> dict:
    """Score the holdout DataFrame and compute all evaluation metrics."""
    model, pipeline = load_artifacts()

    scores = score_batch(df, model, pipeline)          # lower = more anomalous
    # Negate for sklearn metrics: higher score should indicate the positive class
    anomaly_scores_for_auprc = -scores

    y_true = df["is_anomaly"].astype(int).values
    y_pred = (scores < threshold).astype(int)

    # Guard against degenerate cases (all-one-class prediction)
    def _safe_metric(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float("nan")

    precision = _safe_metric(precision_score, y_true, y_pred, zero_division=0)
    recall    = _safe_metric(recall_score,    y_true, y_pred, zero_division=0)
    f1        = _safe_metric(f1_score,        y_true, y_pred, zero_division=0)
    auprc     = _safe_metric(average_precision_score, y_true, anomaly_scores_for_auprc)

    tn, fp, fn_count, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Per-anomaly-type recall
    per_type: dict[str, float | None] = {}
    if "anomaly_type" in df.columns:
        for atype in ["amount_spike", "geo_jump", "odd_hour_burst"]:
            mask = df["anomaly_type"] == atype
            if mask.sum() == 0:
                per_type[atype] = None
                continue
            type_true = y_true[mask]
            type_pred = y_pred[mask]
            per_type[atype] = _safe_metric(recall_score, type_true, type_pred, zero_division=0)

    return {
        "threshold": threshold,
        "n_holdout": len(df),
        "n_anomalies_true":  int(y_true.sum()),
        "n_anomalies_pred":  int(y_pred.sum()),
        "anomaly_rate_true": round(float(y_true.mean()), 4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auprc":     round(auprc,     4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn_count), "tp": int(tp),
        },
        "per_type_recall": {k: (round(v, 4) if v is not None else None)
                            for k, v in per_type.items()},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _print_report(m: dict):
    _print_section("Holdout summary")
    print(f"  Records  : {m['n_holdout']:,}")
    print(f"  True anomalies : {m['n_anomalies_true']:,} "
          f"({m['anomaly_rate_true']:.1%})")
    print(f"  Flagged (pred) : {m['n_anomalies_pred']:,}")

    _print_section("Threshold-based metrics  (threshold = {})".format(m["threshold"]))
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")

    _print_section("Confusion matrix")
    cm = m["confusion_matrix"]
    print(f"                 Pred Normal  Pred Anomaly")
    print(f"  True Normal    {cm['tn']:>10,}  {cm['fp']:>12,}")
    print(f"  True Anomaly   {cm['fn']:>10,}  {cm['tp']:>12,}")

    _print_section("Threshold-independent metric")
    print(f"  AUPRC : {m['auprc']:.4f}  "
          f"(random baseline ≈ {m['anomaly_rate_true']:.3f})")

    if m.get("per_type_recall"):
        _print_section("Per-anomaly-type recall")
        for atype, rec in m["per_type_recall"].items():
            bar = ""
            if rec is not None:
                filled = int(round(rec * 20))
                bar = f"  [{'█' * filled}{'░' * (20 - filled)}]"
            print(f"  {atype:<20s}  {f'{rec:.4f}' if rec is not None else 'N/A':>6}{bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IsolationForest on a labelled holdout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--holdout-file",
        default=None,
        help="Path to JSONL holdout file with is_anomaly labels.  "
             "Generates synthetic holdout if omitted.",
    )
    parser.add_argument(
        "--n-holdout",
        type=int,
        default=10_000,
        help="Holdout size when generating synthetic data (default 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Anomaly score threshold (default 0.0 = natural decision boundary; "
             "production env uses ANOMALY_THRESHOLD=-0.1 for zero-FP conservatism)",
    )
    args = parser.parse_args()

    print("\n── Loading holdout ───────────────────────────────────────")
    df = _load_or_generate_holdout(args.holdout_file, args.n_holdout)

    if "is_anomaly" not in df.columns:
        sys.exit("ERROR: holdout data must contain an 'is_anomaly' column")

    print("\n── Scoring ───────────────────────────────────────────────")
    print("  Loading trained artifacts...")
    metrics = evaluate(df, args.threshold)

    _print_report(metrics)

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {metrics_path}")
    print()


if __name__ == "__main__":
    main()
