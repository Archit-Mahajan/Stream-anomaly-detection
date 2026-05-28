"""
tests/test_model.py — model load, feature pipeline, and scoring contract

Runs offline — only needs the serialised artifacts in model/.
"""
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).parent.parent / "model"


@pytest.fixture(scope="module")
def artifacts():
    from model.scorer import load_artifacts
    return load_artifacts(MODEL_DIR)


@pytest.fixture(scope="module")
def normal_tx():
    from producer.generator import make_transaction
    cfg = {"anomaly_rate": 0.0, "anomaly_types": [], "events_per_second": 1}
    return make_transaction(cfg)


@pytest.fixture(scope="module")
def spike_tx():
    # Hardcoded $25 000 ATM withdrawal — ~450× the median training amount.
    # log1p(25000) ≈ 10.1 → z-score ≈ +7.6 under the training distribution.
    # Deterministic so the test never draws a borderline spike from the RNG.
    return {
        "transaction_id": "test-spike-fixture",
        "user_id": "u_0001",
        "amount": 25000.0,
        "merchant_category": "atm",
        "timestamp": "2024-01-15T02:00:00Z",
        "country": "US",
        "is_anomaly": True,
        "anomaly_type": "amount_spike",
    }


class TestArtifactsLoad:
    def test_model_loads(self, artifacts):
        model, _ = artifacts
        assert model is not None

    def test_pipeline_loads(self, artifacts):
        _, pipeline = artifacts
        assert pipeline is not None

    def test_model_has_estimators(self, artifacts):
        model, _ = artifacts
        assert model.n_estimators > 0

    def test_artifacts_exist_on_disk(self):
        assert (MODEL_DIR / "isolation_forest.pkl").exists()
        assert (MODEL_DIR / "feature_pipeline.pkl").exists()


class TestScoreContract:
    def test_score_transaction_returns_float(self, artifacts, normal_tx):
        from model.scorer import score_transaction
        model, pipeline = artifacts
        score = score_transaction(normal_tx, model, pipeline)
        assert isinstance(score, float)

    def test_score_in_reasonable_range(self, artifacts, normal_tx):
        from model.scorer import score_transaction
        model, pipeline = artifacts
        score = score_transaction(normal_tx, model, pipeline)
        assert -1.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_normal_tx_higher_score_than_spike(self, artifacts, normal_tx, spike_tx):
        from model.scorer import score_transaction
        model, pipeline = artifacts
        normal_score = score_transaction(normal_tx, model, pipeline)
        spike_score  = score_transaction(spike_tx,  model, pipeline)
        # IsolationForest: lower score = more anomalous; spike should score lower
        assert spike_score < normal_score, (
            f"Expected spike ({spike_score:.4f}) < normal ({normal_score:.4f})"
        )

    def test_spike_scores_below_zero(self, artifacts, spike_tx):
        from model.scorer import score_transaction
        model, pipeline = artifacts
        score = score_transaction(spike_tx, model, pipeline)
        # Amount spikes score below 0 (the calibrated contamination boundary).
        # Full recall at -0.1 is 56.7% by design — the test verifies the score
        # is anomalous relative to baseline, not that every spike clears the
        # production threshold.
        assert score < 0, f"Amount spike should score < 0; got {score:.4f}"

    def test_score_batch_returns_array(self, artifacts):
        from producer.generator import make_transaction, get_config
        from model.scorer import score_batch
        model, pipeline = artifacts
        cfg = get_config()
        txs = [make_transaction(cfg) for _ in range(10)]
        scores = score_batch(txs, model, pipeline)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (10,)
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_score_batch_from_dataframe(self, artifacts):
        import pandas as pd
        from producer.generator import make_transaction, get_config
        from model.scorer import score_batch
        model, pipeline = artifacts
        cfg = get_config()
        df = pd.DataFrame([make_transaction(cfg) for _ in range(5)])
        scores = score_batch(df, model, pipeline)
        assert scores.shape == (5,)


class TestMetrics:
    """Sanity check that the persisted evaluation metrics are plausible."""

    def test_metrics_file_exists(self):
        assert (MODEL_DIR / "metrics.json").exists()

    def test_f1_above_threshold(self):
        import json
        m = json.loads((MODEL_DIR / "metrics.json").read_text())
        assert m["f1"] > 0.70, f"F1 too low: {m['f1']}"

    def test_precision_and_recall_present(self):
        import json
        m = json.loads((MODEL_DIR / "metrics.json").read_text())
        assert "precision" in m and "recall" in m and "auprc" in m
