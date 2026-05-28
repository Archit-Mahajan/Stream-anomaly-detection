from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AlertRecord(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    event_time: datetime
    country: str
    score: float
    anomaly_type: Optional[str]
    alerted_at: datetime

    model_config = {"from_attributes": True}


class TransactionIn(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Merchant category (grocery, gas, restaurant, etc.)")
    timestamp: str = Field(
        default="",
        description="ISO-8601 event time, e.g. 2024-01-15T14:23:01Z. Defaults to now.",
    )
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour override (0-23); takes precedence over timestamp")
    user_id: str = Field(default="u_0000", description="User ID; must be in the training pool for home-country signal")
    country: str = Field(default="US", description="ISO-3166-1 alpha-2 country code")


class ScoreOut(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float
    features: dict


class HealthOut(BaseModel):
    status: str  # "ok" | "degraded"
    db: str      # "ok" | error text


class TypeBreakdown(BaseModel):
    amount_spike: int = 0
    geo_jump: int = 0
    odd_hour_burst: int = 0


class StatsOut(BaseModel):
    window_minutes: int
    total_transactions: int
    total_alerts: int
    anomaly_rate: float
    by_type: TypeBreakdown
    model_metrics: dict
