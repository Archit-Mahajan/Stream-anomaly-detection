import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PG_DSN = os.getenv("POSTGRES_DSN", "postgresql://anomaly:anomaly@localhost:5432/anomaly_db")


def _conn():
    return psycopg2.connect(PG_DSN)


def db_healthy() -> tuple[bool, str]:
    try:
        conn = _conn()
        conn.close()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def fetch_alerts(
    limit: int = 50,
    offset: int = 0,
    since: Optional[datetime] = None,
    anomaly_type: Optional[str] = None,
    max_score: Optional[float] = None,
) -> list[dict]:
    """Return alert rows ordered by alerted_at DESC."""
    clauses: list[str] = []
    params: list = []

    if since:
        clauses.append("alerted_at > %s")
        params.append(since)
    if anomaly_type:
        clauses.append("anomaly_type = %s")
        params.append(anomaly_type)
    if max_score is not None:
        # Lower score = more severe; max_score acts as a severity floor.
        clauses.append("score <= %s")
        params.append(max_score)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]

    sql = f"""
        SELECT transaction_id, user_id, amount::float AS amount,
               merchant_category, event_time, country,
               score, anomaly_type, alerted_at
        FROM   alerts
        {where}
        ORDER  BY alerted_at DESC
        LIMIT  %s OFFSET %s
    """
    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def fetch_stats(window_minutes: int = 60) -> dict:
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM transactions WHERE processed_at > %s", (since,))
            total_tx: int = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM alerts WHERE alerted_at > %s", (since,))
            total_alerts: int = cur.fetchone()[0]

            cur.execute(
                """
                SELECT anomaly_type, COUNT(*) AS cnt
                FROM   alerts
                WHERE  alerted_at > %s AND anomaly_type IS NOT NULL
                GROUP  BY anomaly_type
                """,
                (since,),
            )
            by_type = {row[0]: int(row[1]) for row in cur.fetchall()}
    finally:
        conn.close()

    rate = round(total_alerts / total_tx, 6) if total_tx else 0.0
    return {
        "window_minutes": window_minutes,
        "total_transactions": total_tx,
        "total_alerts": total_alerts,
        "anomaly_rate": rate,
        "by_type": {
            "amount_spike": by_type.get("amount_spike", 0),
            "geo_jump": by_type.get("geo_jump", 0),
            "odd_hour_burst": by_type.get("odd_hour_burst", 0),
        },
    }
