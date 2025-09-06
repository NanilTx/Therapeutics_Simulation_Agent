# Therapeutic Simulation Agent (TSA)

A compact, multi‑agent pipeline for therapeutic hypothesis generation and simulation — available as a CLI and a FastAPI service. TSA proposes candidate perturbations (target, dose, duration), simulates expected biomarker deltas with uncertainty, scores alternatives, selects a plan, and reports a simple retrospective validation metric.

- End‑to‑end multi‑agent orchestration (data → hypothesis → simulation → critic → validation)
- Human‑readable summaries for both planning and full pipeline runs
- Rich CLI: top‑K ranking, JSON output, CSV export, and API launcher
- FastAPI service with clean JSON responses and an explanatory `summary` field
- Synthetic data bootstrap for quick demos, plus tests and a notebook

## Quickstart

Prereqs: Python 3.11+

1) Create env and install deps

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

2) Run the API (direct)

```
uvicorn tsa.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open the interactive docs at `http://localhost:8000/docs` (the root `/` redirects there). A quick health check is available at `GET /status`:

```
curl http://localhost:8000/status
```

3) Bootstrap synthetic data (optional – API will auto‑ensure if missing)

```
curl -X POST http://localhost:8000/bootstrap_synth_data
```

4) Run the full pipeline

```
curl -X POST http://localhost:8000/run_pipeline -H 'Content-Type: application/json' -d '{"n": 6}'
```

### CLI

Run the pipeline directly from the terminal with a human‑readable summary (no server required):

```
export PYTHONPATH=src
python -m tsa.cli run -n 6
```

Preview proposed candidates:

```
python -m tsa.cli plan -n 6

Export proposals to CSV:

```
python -m tsa.cli plan -n 6 --csv outputs/proposals.csv
```

### Installable CLI (pipx)

Install the CLI with pipx to get a global `tsa` command:

```
pipx install .
```

Then run:

```
tsa run -n 6
tsa plan -n 6 --json

### Run the API via CLI

Start the FastAPI server (uvicorn) from the CLI:

```
# Local dev
export PYTHONPATH=src
python -m tsa.cli api --host 0.0.0.0 --port 8000 --reload --open-docs

# If installed (pipx/pip)
tsa api --host 0.0.0.0 --port 8000 --reload --open-docs
```

Open http://localhost:8000/docs to explore the endpoints.

For production‑style runs you can use multiple workers (not with `--reload`):

```
tsa api --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```
```
```

Tips:
- Add `--json` to include structured JSON after the summary.
- Use `--top N` with `run` to show a Top‑N ranking table.
- Use `--no-color` to disable ANSI styling in terminals that don’t support it.

Sample `run` output:

```
Pipeline Results
Selected: MAPT ↓  dose=1.0  duration=14d  score=1.899
Validation RMSE: 0.189 (lower is better)
Biomarker effects (first 3 of 3): -0.089±0.059, 0.003±0.050, 0.041±0.054

Top 5 Candidates
rank  target  dir  dose  duration  score
----------------------------------------
1     MAPT    ↓    1.0   14        1.899
2     APP     ↓    1.0   7         1.512
3     MAPT    ↓    0.5   7         0.950
4     MAPT    ↓    0.25  28        0.817
5     APP     ↓    0.25  14        0.654

Evaluated 6 candidate interventions and selected target MAPT (downregulation) at dose 1.0 for 14 days. Selection score: 1.899 (higher is better). Predictions include 3 biomarker effect values with uncertainties. Retrospective validation RMSE: 0.189.
```

## API

- `GET /status` – Health check `{ "ok": true }`
- `POST /bootstrap_synth_data` – Generates CSVs under `TSA_DATA_DIR` (see Config)
- `POST /plan` – Body: `{ "n": int }` → returns `proposals` plus a human `summary`
- `POST /run_pipeline` – Body: `{ "n": int }` → returns `proposals`, `simulations`, `selected`, `validation`, and a human `summary`

Example request/response shape for `run_pipeline`:

```
{
  "proposals": [{"target":"MAPT","direction":-1.0,"dose":0.5,"duration":14}, ...],
  "simulations": [{"biomarker_delta":[...],"uncertainty":[...]}, ...],
  "selected": {"choice": { ... }, "score": 1.23},
  "validation": {"rmse": 0.42},
  "summary": "Evaluated 6 candidates and selected target MAPT (downregulation) at dose 0.5 for 14 days. Selection score: 1.230 (higher is better). Predictions include 3 biomarker effect values with uncertainties. Retrospective validation RMSE: 0.420."
}
```

The `summary` field explains results in plain English and mirrors the CLI’s summary lines.

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
