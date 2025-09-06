PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

.PHONY: setup run smoke test fmt

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	PYTHONPATH=src $(UVICORN) tsa.api.app:app --reload --host 0.0.0.0 --port 8000

# Starts the server, waits for it to come up, runs smoke tests, then stops it
smoke:
	PYTHONPATH=src $(UVICORN) tsa.api.app:app --host 127.0.0.1 --port 8000 --log-level warning & echo $$! > .server.pid
	@echo "Waiting for API to be ready..."; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		$(PY) -c "import sys,requests;\
		\
import time;\
\
import os;\
\
url='http://127.0.0.1:8000/status';\
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

