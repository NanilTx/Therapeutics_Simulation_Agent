# Therapeutics Simulation Agent (MVP)

A minimal multi‑agent pipeline for therapeutic hypothesis generation and simulation, exposed as a FastAPI service. The system proposes candidate perturbations (target, dose, duration), simulates expected biomarker deltas with uncertainty, scores alternatives, and returns a selected plan along with a simple retrospective validation metric.

- Multi‑agent orchestration (data → hypothesis → simulation → critic → validation)
- Synthetic dataset bootstrap for quick demos
- REST API via FastAPI with a few endpoints
- Docker image for easy deployment
- Lightweight tests and an example notebook

## Quickstart

Prereqs: Python 3.11+

1) Create env and install deps

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

2) Run the API

```
uvicorn tsa.api.app:app --reload --host 0.0.0.0 --port 8000
```

3) Bootstrap synthetic data (optional – API will auto‑ensure if missing)

```
curl -X POST http://localhost:8000/bootstrap_synth_data
```

4) Run the full pipeline

```
curl -X POST http://localhost:8000/run_pipeline -H 'Content-Type: application/json' -d '{"n": 6}'
```

## API

- `GET /status` – Health check `{ "ok": true }`
- `POST /bootstrap_synth_data` – Generates CSVs under `TSA_DATA_DIR` (see Config)
- `POST /plan` – Body: `{ "n": int }` → returns `proposals`
- `POST /run_pipeline` – Body: `{ "n": int }` → runs end‑to‑end and returns `proposals`, `simulations`, `selected`, `validation`

Example request/response shape for `run_pipeline`:

```
{
  "proposals": [{"target":"MAPT","direction":-1.0,"dose":0.5,"duration":14}, ...],
  "simulations": [{"biomarker_delta":[...],"uncertainty":[...]}, ...],
  "selected": {"choice": { ... }, "score": 1.23},
  "validation": {"rmse": 0.42}
}
```

## Docker

Build and run the API in Docker:

```
docker build -t tsa:latest -f docker/Dockerfile .
docker run --rm -p 8000:8000 -e TSA_DATA_DIR=/app/data tsa:latest
```

## Testing

```
pytest -q
```

## Configuration

Environment variables (with defaults):

- `TSA_DATA_DIR` – output directory for generated CSVs (default `./data`)
- `TSA_SEED` – RNG seed for synthetic data and simulations (default `1337`)
- `TSA_LATENT_DIM` – latent dimensionality for the simulator (default `16`)

You can set these in your shell, or provide them to Docker via `-e` flags. An `.env.example` is included as a reference for values; export them manually when running locally.

## Architecture

High‑level flow orchestrated by `Orchestrator`:

1. DataAgent – ensures data availability; can generate synthetic CSVs.
2. HypothesisAgent – proposes `n` candidate perturbations over targets/dose/duration.
3. SimulationAgent – uses a simple latent model to predict biomarker deltas + uncertainty.
4. CriticAgent – scores simulations by effect/uncertainty and selects the best.
5. ValidationAgent – returns a simple retrospective RMSE metric.

Key modules (under `src/tsa`):

- `api/app.py` – FastAPI app and endpoints
- `agents/` – agents and `Orchestrator`
- `agent/simulator.py` – latent perturbation simulator
- `data/synthetic.py` – synthetic data generators and writers
- `eval/validation.py` – simple retrospective validation
- `config.py` – paths and env‑driven configuration

## Project Structure

```
src/
  tsa/
    api/app.py
    agents/
    agent/simulator.py
    data/
    eval/
    config.py
tests/
docker/Dockerfile
notebooks/TSA_Demo.ipynb
requirements.txt
```

## Notes

- This is an MVP intended for demonstration and iteration, not clinical use.
- The simulation and validation are intentionally simple and stochastic.
- See `tests/` for minimal examples of usage.
