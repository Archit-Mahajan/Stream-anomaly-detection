from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="Stream Anomaly Detector",
    description=(
        "Real-time financial transaction anomaly detection. "
        "Scores via IsolationForest; alerts persisted in Postgres."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)

_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if _DASHBOARD_DIR.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_DASHBOARD_DIR), html=True),
        name="dashboard",
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard")
