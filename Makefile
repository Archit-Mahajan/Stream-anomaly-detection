# stream-anomaly-detector — one-command targets
# Usage: make <target>   (default: help)
#
# Prerequisites: Docker + Docker Compose v2, Python 3.10+, a working venv
# Activate your venv first: source .venv/bin/activate

PYTHON   ?= python3
UVICORN  ?= uvicorn
PIP      ?= pip

.PHONY: help venv install up down wait train produce stream api test all clean

# ── Meta ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make up        Start Redpanda + Postgres in Docker (detached)"
	@echo "  make down      Stop and remove Docker containers"
	@echo "  make wait      Block until both services report healthy"
	@echo "  make venv      Create .venv (skip if already exists)"
	@echo "  make install   pip install -r requirements.txt into active venv"
	@echo "  make train     Train IsolationForest + evaluate; writes model/metrics.json"
	@echo "  make produce   Start the transaction producer (Ctrl-C to stop)"
	@echo "  make stream    Start the PySpark streaming job (Ctrl-C to stop)"
	@echo "  make api       Start FastAPI on port 8000 (Ctrl-C to stop)"
	@echo "  make test      Run the test suite (no broker/DB required)"
	@echo "  make all       up → wait → train → launch produce+stream+api in bg"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────

venv:
	@[ -d .venv ] && echo ".venv already exists — skipping" || $(PYTHON) -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	$(PIP) install -r requirements.txt

# ── Data plane ────────────────────────────────────────────────────────────────

up:
	docker compose up -d
	@echo "Redpanda + Postgres starting. Run 'make wait' to block until healthy."

down:
	docker compose down

wait:
	@echo "Waiting for Redpanda…"
	@until docker exec redpanda rpk cluster health 2>/dev/null | grep -q "Healthy:.*true"; do \
		printf '.'; sleep 2; \
	done
	@echo ""
	@echo "Waiting for Postgres…"
	@until docker exec postgres pg_isready -U anomaly -d anomaly_db 2>/dev/null | grep -q "accepting"; do \
		printf '.'; sleep 2; \
	done
	@echo ""
	@echo "Both services healthy."

# ── Application layers ────────────────────────────────────────────────────────

train:
	$(PYTHON) -m model.train
	$(PYTHON) -m model.evaluate
	@echo "Model artifacts written to model/. Metrics:"
	@cat model/metrics.json

produce:
	$(PYTHON) -m producer

stream:
	$(PYTHON) -m streaming

api:
	$(UVICORN) api.main:app --reload --port 8000

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ── Full pipeline (background processes) ──────────────────────────────────────

all: up wait train
	@echo "Starting producer in background (logs → /tmp/producer.log)…"
	$(PYTHON) -m producer >> /tmp/producer.log 2>&1 &
	@echo "Starting Spark streaming job in background (logs → /tmp/streaming.log)…"
	$(PYTHON) -m streaming >> /tmp/streaming.log 2>&1 &
	@echo "Starting API in background (logs → /tmp/api.log)…"
	$(UVICORN) api.main:app --port 8000 >> /tmp/api.log 2>&1 &
	@echo ""
	@echo "All processes running. Dashboard: http://localhost:8000/dashboard"
	@echo "Logs: /tmp/producer.log  /tmp/streaming.log  /tmp/api.log"
	@echo "Kill all: pkill -f 'python -m producer'; pkill -f 'python -m streaming'; pkill -f 'uvicorn api'"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf streaming/checkpoints
	@echo "Cleaned pycache, pytest cache, and Spark checkpoints."
