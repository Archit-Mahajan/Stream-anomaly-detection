"""
tests/test_producer.py — producer output shape and field contract

These tests run offline (no Kafka / Docker required).
"""
import re
from datetime import datetime, timezone

import pytest

from producer.generator import MERCHANT_CATEGORIES, _USER_IDS, get_config, make_transaction

CONFIG = get_config()
REQUIRED_FIELDS = {
    "transaction_id",
    "user_id",
    "amount",
    "merchant_category",
    "timestamp",
    "country",
    "is_anomaly",
    "anomaly_type",
}

ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _normal_config():
    return {"anomaly_rate": 0.0, "anomaly_types": [], "events_per_second": 1}


def _anomaly_config(anomaly_type: str):
    return {"anomaly_rate": 1.0, "anomaly_types": [anomaly_type], "events_per_second": 1}


class TestTransactionShape:
    def test_all_required_fields_present(self):
        tx = make_transaction(CONFIG)
        assert REQUIRED_FIELDS.issubset(tx.keys()), f"Missing fields: {REQUIRED_FIELDS - tx.keys()}"

    def test_amount_is_positive(self):
        for _ in range(20):
            tx = make_transaction(_normal_config())
            assert tx["amount"] > 0

    def test_timestamp_iso8601(self):
        tx = make_transaction(CONFIG)
        assert ISO8601_RE.match(tx["timestamp"]), f"Bad timestamp: {tx['timestamp']!r}"

    def test_user_id_in_pool(self):
        tx = make_transaction(CONFIG)
        assert tx["user_id"] in _USER_IDS

    def test_merchant_category_valid(self):
        for _ in range(10):
            tx = make_transaction(CONFIG)
            assert tx["merchant_category"] in MERCHANT_CATEGORIES

    def test_normal_transaction_not_flagged(self):
        cfg = _normal_config()
        for _ in range(30):
            tx = make_transaction(cfg)
            assert tx["is_anomaly"] is False
            assert tx["anomaly_type"] is None


class TestAnomalyInjection:
    def test_amount_spike_large(self):
        cfg = _anomaly_config("amount_spike")
        for _ in range(20):
            tx = make_transaction(cfg)
            assert tx["is_anomaly"] is True
            assert tx["anomaly_type"] == "amount_spike"
            # spike multiplies by 10–50×; even a $1 base gives >$10
            assert tx["amount"] > 100, f"Spike amount too small: {tx['amount']}"

    def test_geo_jump_foreign_country(self):
        cfg = _anomaly_config("geo_jump")
        foreign = {"JP", "SG", "AU", "ZA"}
        for _ in range(20):
            tx = make_transaction(cfg)
            assert tx["is_anomaly"] is True
            assert tx["anomaly_type"] == "geo_jump"
            assert tx["country"] in foreign, f"Expected foreign country, got {tx['country']!r}"

    def test_odd_hour_burst_window(self):
        cfg = _anomaly_config("odd_hour_burst")
        for _ in range(20):
            tx = make_transaction(cfg)
            assert tx["is_anomaly"] is True
            assert tx["anomaly_type"] == "odd_hour_burst"
            hour = datetime.strptime(tx["timestamp"], "%Y-%m-%dT%H:%M:%SZ").hour
            assert 2 <= hour <= 4, f"Expected 02–04, got hour {hour}"

    def test_transaction_id_unique(self):
        cfg = get_config()
        ids = [make_transaction(cfg)["transaction_id"] for _ in range(100)]
        assert len(set(ids)) == 100, "Duplicate transaction IDs detected"
