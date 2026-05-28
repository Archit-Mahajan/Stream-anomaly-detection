import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api import db
from api.schemas import AlertRecord, HealthOut, ScoreOut, StatsOut, TransactionIn, TypeBreakdown

router = APIRouter()

_METRICS_PATH = Path(__file__).parent.parent / "model" / "metrics.json"
_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "-0.1"))


def _load_metrics() -> dict:
    try:
        return json.loads(_METRICS_PATH.read_text())
    except Exception:
        return {}


# ── Ops ───────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthOut, tags=["ops"])
def health():
    ok, msg = db.db_healthy()
    return HealthOut(status="ok" if ok else "degraded", db=msg)


@router.get(
    "/stats",
    response_model=StatsOut,
    tags=["ops"],
    summary="Counts, anomaly rate, and model metrics for the rolling window",
)
def stats(
    window_minutes: int = Query(10, ge=1, le=1440, description="Rolling window size in minutes"),
):
    try:
        s = db.fetch_stats(window_minutes)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    return StatsOut(
        window_minutes=s["window_minutes"],
        total_transactions=s["total_transactions"],
        total_alerts=s["total_alerts"],
        anomaly_rate=s["anomaly_rate"],
        by_type=TypeBreakdown(**s["by_type"]),
        model_metrics=_load_metrics(),
    )


# ── Data ──────────────────────────────────────────────────────────────────────

@router.get(
    "/alerts",
    response_model=list[AlertRecord],
    tags=["data"],
    summary="Recent anomaly alerts, paginated and filterable",
)
def list_alerts(
    limit: int = Query(50, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    since: Optional[datetime] = Query(
        None,
        description="Return only alerts after this ISO-8601 timestamp",
    ),
    anomaly_type: Optional[str] = Query(
        None,
        description="Filter: amount_spike | geo_jump | odd_hour_burst",
    ),
    max_score: Optional[float] = Query(
        None,
        description="Severity filter: only return alerts with score ≤ this value (lower = more severe)",
    ),
):
    try:
        rows = db.fetch_alerts(
            limit=limit,
            offset=offset,
            since=since,
            anomaly_type=anomaly_type,
            max_score=max_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    return [AlertRecord(**r) for r in rows]


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post(
    "/score",
    response_model=ScoreOut,
    tags=["inference"],
    summary="Score a single transaction on demand using the serialised model",
)
def score(tx: TransactionIn):
    # Lazy import: model artifacts are large; only load on first /score call.
    try:
        from model.scorer import is_anomaly, load_artifacts, score_transaction
        model, pipeline = load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}")

    tx_dict = tx.model_dump()
    if not tx_dict.get("timestamp"):
        tx_dict["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        s = score_transaction(tx_dict, model, pipeline)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scoring failed: {exc}")

    return ScoreOut(
        score=round(float(s), 6),
        is_anomaly=is_anomaly(s, _THRESHOLD),
        threshold=_THRESHOLD,
        features={
            "amount": tx_dict["amount"],
            "merchant_category": tx_dict["merchant_category"],
            "country": tx_dict["country"],
            "user_id": tx_dict["user_id"],
            "timestamp": tx_dict["timestamp"],
        },
    )
