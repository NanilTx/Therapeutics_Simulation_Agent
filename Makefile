PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

# Server configuration
HOST ?= 0.0.0.0
PORT ?= 8000
RELOAD ?= --reload

.PHONY: setup run start stop smoke test fmt kill-port

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	PYTHONPATH=src $(UVICORN) tsa.api.app:app $(RELOAD) --host $(HOST) --port $(PORT)

# Start server in background, write PID + logs
PID_FILE := .server.$(PORT).pid
LOG_FILE := .server.$(PORT).log

start:
	@if [ -f $(PID_FILE) ] && kill -0 `cat $(PID_FILE)` 2>/dev/null; then \
		echo "Server already running (PID `cat $(PID_FILE)`) on port $(PORT)."; \
		exit 0; \
	fi
	@echo "Starting API on $(HOST):$(PORT)..."; \
	PYTHONPATH=src $(UVICORN) tsa.api.app:app $(RELOAD) --host $(HOST) --port $(PORT) --log-level info > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE); \
	sleep 1; \
	if kill -0 `cat $(PID_FILE)` 2>/dev/null; then \
		echo "Started (PID `cat $(PID_FILE)`). Logs: $(LOG_FILE)"; \
	else \
		echo "Failed to start. See logs: $(LOG_FILE)"; \
		exit 1; \
	fi

stop:
	@if [ -f $(PID_FILE) ]; then \
		pid=`cat $(PID_FILE)`; \
		if kill -0 $$pid 2>/dev/null; then \
			echo "Stopping server PID $$pid..."; \
			kill $$pid || true; \
			sleep 1; \
			if kill -0 $$pid 2>/dev/null; then \
				echo "Force killing PID $$pid..."; \
				kill -9 $$pid || true; \
			fi; \
		else \
			echo "PID file present but process not running."; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		pids=`lsof -ti tcp:$(PORT)`; \
		if [ -n "$$pids" ]; then \
			echo "Killing PIDs on port $(PORT): $$pids"; \
			kill -9 $$pids || true; \
		else \
			echo "No server PID file and nothing on port $(PORT)."; \
		fi; \
	fi

# Starts the server, waits for it to come up, runs smoke tests, then stops it
smoke:
	PYTHONPATH=src $(UVICORN) tsa.api.app:app --host 127.0.0.1 --port $(PORT) --log-level warning & echo $$! > .server.pid
	@echo "Waiting for API to be ready..."; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		$(PY) -c "import sys,requests;\
		\
import time;\
\
import os;\
\
		url='http://127.0.0.1:$(PORT)/status';\
\
import requests;\
\
resp=None;\
\
try: resp=requests.get(url, timeout=1);\
\
except Exception: pass;\
\
sys.exit(0 if (resp and resp.ok) else 1)" && break || (sleep 1); \
	done
	$(PY) scripts/smoke_test.py
	-kill `cat .server.pid` 2>/dev/null || true
	rm -f .server.pid

test:
	$(VENV)/bin/pytest -q

# Kill any process listening on $(PORT)
kill-port:
	-@pids=`lsof -ti tcp:$(PORT)`; \
	if [ -n "$$pids" ]; then \
		echo "Killing PIDs on port $(PORT): $$pids"; \
		kill -9 $$pids || true; \
	else \
		echo "No process is listening on port $(PORT)"; \
	fi

# CLI helpers
.PHONY: cli-plan cli-run
cli-plan:
	PYTHONPATH=src $(PY) -m tsa.cli plan -n 6

cli-run:
	PYTHONPATH=src $(PY) -m tsa.cli run -n 6

.PHONY: cli-api
cli-api:
	PYTHONPATH=src $(PY) -m tsa.cli api --host $(HOST) --port $(PORT) $(RELOAD)
