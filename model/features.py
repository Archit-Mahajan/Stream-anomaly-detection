"""
Feature engineering pipeline for the stream-anomaly-detector.

FEATURE DESIGN RATIONALE
─────────────────────────
Five features cover all three injected anomaly types while staying well below
the ~20-feature sweet spot for IsolationForest (more features dilute isolation-
tree splits via the curse of dimensionality):

  amount_log_zscore — log1p(amount), then z-normalised using training stats.
    log1p flattens the right-skewed lognormal distribution so that normal
    purchases cluster tightly near z≈0.  Amount-spike anomalies (×10–50 the
    median) land 4–10σ out, which isolation trees isolate in very few splits.

  hour_sin / hour_cos — cyclical encoding of hour-of-day on the unit circle.
    Without this, hour=0 and hour=23 are 23 integers apart but only 1 hour
    apart in reality.  Anomalous "odd_hour_burst" transactions (02–04) land
    in a low-density arc of the circle far from the 10–15 peak cluster.

  merchant_freq — empirical frequency of the merchant category in training
    data, normalised to [0, 1].  In real fraud data, merchant categories are
    highly skewed (grocery >> crypto-exchange); an unusually rare category
    is a weak but additive signal.  In our uniform synthetic data this feature
    is near-flat — documented limitation, right approach for real data.

  is_home_country — 1 if transaction.country == user's home country, else 0.
    Proxy for geo-velocity: all geo_jump anomalies land in countries ≥8 time
    zones from the home, so this binary flag perfectly separates them.
    True geo-velocity (travel_km / seconds_since_last_tx) requires a stateful
    per-user join across events — doable in Spark with watermarked sessionisation
    (Phase 3); here we use the per-user mode of country in normal training rows
    as the home-country ground truth.

STATEFULNESS REQUIREMENT
─────────────────────────
amount_mean, amount_std, and merchant_freq_map are learned from the training
corpus and must be serialised alongside the model.  Recomputing them on the
inference batch (micro-batch in Spark) would give wrong z-scores if the
inference distribution drifts — a silent but catastrophic failure mode.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "amount_log_zscore",
    "hour_sin",
    "hour_cos",
    "merchant_freq",
    "is_home_country",
]


def _parse_hour(ts: str) -> int:
    """Extract hour-of-day (0–23) from ISO-8601 string 'YYYY-MM-DDTHH:MM:SSZ'."""
    try:
        return int(ts[11:13])
    except (IndexError, ValueError, TypeError):
        return 12  # fallback: noon is the least biased guess


class FeaturePipeline:
    """Stateful transformer: fit on training data, call transform/transform_one
    at inference time.  Serialised with joblib so train and serve stats match."""

    def __init__(self):
        self.amount_mean: float = 0.0
        self.amount_std: float = 1.0
        self.merchant_freq_map: dict[str, float] = {}
        self.merchant_freq_fallback: float = 0.0
        # {user_id: home_country} — inferred from normal rows at fit time
        self.user_home_country: dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Learn normalization stats from the training DataFrame.

        Learns from ALL rows (normal + anomalies) because in production there
        is no clean separation.  The low contamination rate (~2.5%) means
        anomalies barely move the mean/std.  If you have labelled data, passing
        only normal rows gives tighter z-score calibration.
        """
        log_amounts = np.log1p(df["amount"].astype(float))
        self.amount_mean = float(log_amounts.mean())
        # max(..., 1e-6) guards against a degenerate single-value training set
        self.amount_std = max(float(log_amounts.std()), 1e-6)

        freq = df["merchant_category"].value_counts(normalize=True)
        self.merchant_freq_map = freq.to_dict()
        self.merchant_freq_fallback = float(freq.min())

        # Build user→home_country from the normal subset so anomalous geo_jump
        # rows (where country is far) don't corrupt the mode calculation.
        if {"user_id", "country"}.issubset(df.columns):
            src = (
                df[~df["is_anomaly"].astype(bool)]
                if "is_anomaly" in df.columns
                else df
            )
            if not src.empty:
                self.user_home_country = (
                    src.groupby("user_id")["country"]
                    .agg(lambda s: s.mode().iloc[0])
                    .to_dict()
                )

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised transform; returns (N, 5) float32 feature matrix."""
        log_amt = np.log1p(df["amount"].astype(float))
        amt_z = (log_amt - self.amount_mean) / self.amount_std

        hours = df["timestamp"].map(_parse_hour).astype(float)
        h_sin = np.sin(2 * math.pi * hours / 24)
        h_cos = np.cos(2 * math.pi * hours / 24)

        m_freq = (
            df["merchant_category"]
            .map(self.merchant_freq_map)
            .fillna(self.merchant_freq_fallback)
            .astype(float)
        )

        is_home = df.apply(
            lambda r: 1.0
            if self.user_home_country.get(r["user_id"], "") == r["country"]
            else 0.0,
            axis=1,
        )

        return np.column_stack([amt_z, h_sin, h_cos, m_freq, is_home]).astype(np.float32)

    def transform_one(self, tx: dict) -> np.ndarray:
        """Transform a single transaction dict; returns (1, 5) float32 array.

        Used by scorer.py for per-record inference in the Spark UDF and API.
        """
        amt_z = (math.log1p(float(tx["amount"])) - self.amount_mean) / self.amount_std

        hour = tx.get("hour") or _parse_hour(tx.get("timestamp", ""))
        h_sin = math.sin(2 * math.pi * int(hour) / 24)
        h_cos = math.cos(2 * math.pi * int(hour) / 24)

        cat = tx.get("merchant_category", "")
        m_freq = self.merchant_freq_map.get(cat, self.merchant_freq_fallback)

        user_id = tx.get("user_id", "")
        country = tx.get("country", "")
        is_home = 1.0 if self.user_home_country.get(user_id, "") == country else 0.0

        return np.array([[amt_z, h_sin, h_cos, m_freq, is_home]], dtype=np.float32)
