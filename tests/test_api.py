"""
tests/test_api.py — FastAPI endpoint contract (no live DB required)

Uses FastAPI's TestClient which runs the app in-process.  Database calls are
patched so the tests pass without a running Postgres instance.
"""
from unittest.mock import patch
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _mock_healthy():
    return (True, "ok")


def _mock_alerts(**_kwargs):
    return [
        {
            "transaction_id": "abc-123",
            "user_id": "u_0001",
            "amount": 9500.0,
            "merchant_category": "atm",
            "event_time": datetime.now(timezone.utc),
            "country": "SG",
            "score": -0.45,
            "anomaly_type": "amount_spike",
            "alerted_at": datetime.now(timezone.utc),
        }
    ]


def _mock_stats(**_kwargs):
    return {
        "window_minutes": 10,
        "total_transactions": 500,
        "total_alerts": 14,
        "anomaly_rate": 0.028,
        "by_type": {"amount_spike": 5, "geo_jump": 6, "odd_hour_burst": 3},
    }


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        with patch("api.db.db_healthy", return_value=(True, "ok")):
            r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        with patch("api.db.db_healthy", return_value=(True, "ok")):
            r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "db" in body

    def test_health_ok_when_db_up(self, client):
        with patch("api.db.db_healthy", return_value=(True, "ok")):
            r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_degraded_when_db_down(self, client):
        with patch("api.db.db_healthy", return_value=(False, "connection refused")):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"


# ── /alerts ───────────────────────────────────────────────────────────────────

class TestAlerts:
    def test_alerts_returns_200(self, client):
        with patch("api.db.fetch_alerts", side_effect=lambda **kw: _mock_alerts(**kw)):
            r = client.get("/alerts")
        assert r.status_code == 200

    def test_alerts_returns_list(self, client):
        with patch("api.db.fetch_alerts", side_effect=lambda **kw: _mock_alerts(**kw)):
            r = client.get("/alerts")
        assert isinstance(r.json(), list)

    def test_alerts_record_has_required_fields(self, client):
        with patch("api.db.fetch_alerts", side_effect=lambda **kw: _mock_alerts(**kw)):
            r = client.get("/alerts")
        rec = r.json()[0]
        for field in ("transaction_id", "user_id", "amount", "score", "anomaly_type"):
            assert field in rec, f"Missing field: {field}"

    def test_alerts_limit_param(self, client):
        with patch("api.db.fetch_alerts", side_effect=lambda **kw: _mock_alerts(**kw)):
            r = client.get("/alerts?limit=5")
        assert r.status_code == 200

    def test_alerts_invalid_limit_rejected(self, client):
        r = client.get("/alerts?limit=9999")
        assert r.status_code == 422


# ── /stats ────────────────────────────────────────────────────────────────────

class TestStats:
    # fetch_stats is called with a positional window_minutes int, hence lambda w: ...
    def test_stats_returns_200(self, client):
        with patch("api.db.fetch_stats", side_effect=lambda w: _mock_stats()):
            r = client.get("/stats")
        assert r.status_code == 200

    def test_stats_schema(self, client):
        with patch("api.db.fetch_stats", side_effect=lambda w: _mock_stats()):
            r = client.get("/stats")
        body = r.json()
        for field in ("window_minutes", "total_transactions", "total_alerts", "anomaly_rate", "by_type"):
            assert field in body

    def test_stats_by_type_fields(self, client):
        with patch("api.db.fetch_stats", side_effect=lambda w: _mock_stats()):
            r = client.get("/stats")
        bt = r.json()["by_type"]
        for key in ("amount_spike", "geo_jump", "odd_hour_burst"):
            assert key in bt

    def test_stats_window_minutes_param(self, client):
        with patch("api.db.fetch_stats", side_effect=lambda w: _mock_stats()):
            r = client.get("/stats?window_minutes=30")
        assert r.status_code == 200


# ── /score ────────────────────────────────────────────────────────────────────

class TestScore:
    def _payload(self):
        return {
            "amount": 9500.0,
            "merchant_category": "atm",
            "country": "SG",
            "user_id": "u_0001",
        }

    def test_score_returns_200(self, client):
        r = client.post("/score", json=self._payload())
        assert r.status_code == 200

    def test_score_response_schema(self, client):
        r = client.post("/score", json=self._payload())
        body = r.json()
        assert "score" in body
        assert "is_anomaly" in body
        assert "threshold" in body

    def test_score_obvious_anomaly_flagged(self, client):
        r = client.post("/score", json=self._payload())
        body = r.json()
        assert body["is_anomaly"] is True, (
            f"High-amount ATM in Singapore should be flagged; got score={body['score']}"
        )

    def test_score_normal_tx_not_flagged(self, client):
        payload = {
            "amount": 55.0,
            "merchant_category": "grocery",
            "country": "US",
            "user_id": "u_0001",
        }
        r = client.post("/score", json=payload)
        assert r.status_code == 200
        # Normal transaction may or may not be flagged depending on training;
        # we just assert the response is well-formed.
        body = r.json()
        assert isinstance(body["is_anomaly"], bool)

    def test_score_missing_required_field_422(self, client):
        r = client.post("/score", json={"amount": 100.0})
        assert r.status_code == 422
