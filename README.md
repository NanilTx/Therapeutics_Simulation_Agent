# Therapeutic Simulation Agent (TSA)

A compact, multi‑agent pipeline for therapeutic hypothesis generation and simulation — available as a CLI and a FastAPI service. TSA proposes candidate perturbations (target, dose, duration), simulates expected biomarker deltas with uncertainty, scores alternatives, selects a plan, and reports a simple retrospective validation metric.

- End‑to‑end multi‑agent orchestration (data → hypothesis → simulation → critic → validation)
- Human‑readable summaries for both planning and full pipeline runs
- Rich CLI: top‑K ranking, JSON output, CSV export, and API launcher
- FastAPI service with clean JSON responses and an explanatory `summary` field
- Synthetic data bootstrap for quick demos, plus tests and a notebook

## Quickstart

Prereqs: Python 3.9–3.12 (3.11–3.12 recommended)

1) Create env and install deps

```
python -m venv .venv
source .venv/bin/activate
# CLI only
pip install -r requirements-cli.txt
# Or API + CLI
# pip install -r requirements-api.txt
export PYTHONPATH=src
```

2) Run the CLI pipeline (no server)

Run the end-to-end pipeline from your terminal. Streaming is enabled by default so you’ll see progress as it happens.

```
export PYTHONPATH=src
python -m tsa.cli run -n 6 --top 5 --narrative
```

Example output (abridged):

```
[12:34:56] Starting end-to-end pipeline
[12:34:56] [1/5] Ensuring data availability…
[12:34:56] [1/5] Data ready at ./data
[12:34:56] [2/5] Generating 6 candidate proposals…
[12:34:56] [2/5] Generated 6 proposals across targets ['APP', 'MAPT']
[12:34:56] [3/5] Simulating biomarker effects for 6 candidates…
[12:34:56]   • [1/6] MAPT dose=0.25 dur=28d  effect_sum=0.045  uncertainty_sum=0.163
…
[12:34:56] [4/5] Selected MAPT (score=1.899)
[12:34:56] [5/5] Validation complete (RMSE=0.189)
[12:34:56] Done in 0.01s

Pipeline Results
Selected Candidate
target  dir  dose  duration  score
----------------------------------
MAPT    ↓    1.0   14        1.899
Validation RMSE: 0.189 (lower is better)

Top 5 Candidates
rank  target  dir  dose  duration  score
----------------------------------------
*1    MAPT    ↓    1.0   14        1.899
 2    APP     ↓    1.0   7         1.512
 3    MAPT    ↓    0.5   7         0.950
 4    MAPT    ↓    0.25  28        0.817
 5    APP     ↓    0.25  14        0.654

In plain terms:
We proposed 6 interventions and simulated expected biomarker effects with uncertainties. We prefer
candidates with strong overall effect and lower total uncertainty. Based on this, we selected MAPT at
dose 1.0 for 14 days. A quick retrospective check suggests the model’s predictions are reasonably
calibrated (RMSE shown above).
```

Optionally save a Markdown report and artifacts:

```
python -m tsa.cli run -n 6 --narrative --md outputs/pipeline_report.md --save outputs/
```

3) Run the API (direct)

```
uvicorn tsa.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open the interactive docs at `http://localhost:8000/docs` (the root `/` redirects there). A quick health check is available at `GET /status`:

```
curl http://localhost:8000/status
```

4) Bootstrap synthetic data (optional – API will auto‑ensure if missing)

```
curl -X POST http://localhost:8000/bootstrap_synth_data
```

5) Run the full pipeline

```
curl -X POST http://localhost:8000/run_pipeline -H 'Content-Type: application/json' -d '{"n": 6}'
```

### CLI

Note: The CLI no longer imports FastAPI at startup. You can use all CLI commands without installing `fastapi`/`uvicorn`. Only the `api` subcommand and the web service require them.

Run the pipeline directly from the terminal with a human‑readable summary (no server required):

```
export PYTHONPATH=src
python -m tsa.cli run -n 6
```

Preview proposed candidates:

```
python -m tsa.cli plan -n 6
```

Example output (abridged):

```
Plan Summary
Proposed 6 candidates across targets ['APP', 'MAPT'], doses [0.25, 0.5, 1.0], durations [7, 14, 28].
Examples (3):
  - MAPT ↓  dose=0.25  duration=28d
  - APP ↓  dose=0.25  duration=14d
  - MAPT ↓  dose=0.25  duration=7d

Proposals (first 6):
target  dir  dose  duration
---------------------------
MAPT    ↓    0.25  28      
APP     ↓    0.25  14      
MAPT    ↓    0.25  7       
APP     ↓    1.0   7       
MAPT    ↓    1.0   14      
MAPT    ↓    0.5   7       

Distribution:
  target APP    ████████████             2
  target MAPT   ████████████████████████ 4
  duration 7d     ████████████████████████ 3
  duration 14d    ████████████████         2
  duration 28d    ████████                 1
  dose 0.25      ████████████████████████ 3
  dose 0.5       ████████                 1
  dose 1.0       ████████████████         2

Proposed 6 intervention candidates across targets ['APP', 'MAPT'], doses [0.25, 0.5, 1.0], and
durations [7, 14, 28]. Example: target MAPT (downregulation) at dose 0.25 for 28 days.

Legend: dir ↑ upregulation, ↓ downregulation.
```

Export proposals to CSV:

```
python -m tsa.cli plan -n 6 --csv outputs/proposals.csv
```

Common flags

- `--top N` (run): show a Top‑N ranking table
- `--bio K` (run): show first K biomarker rows
- `--no-stream` (run): disable streaming (streaming is on by default)
- `--narrative` (run): add a plain‑English explanation
- `--csv PATH` (plan): write proposals to CSV
- `--show K` (plan): show the first K proposals in a table
- `--md PATH` (run/plan): export a Markdown report
- `--json`: append structured JSON after the summary
- `--no-color`: disable ANSI coloring
- `--save DIR`: auto-save outputs to DIR with timestamped filenames
- `--width N`: override terminal width for wrapping/tables
- `--version`: print CLI version

#### CLI Overview

| Subcommand | Purpose | Key options |
|---|---|---|
| `plan` | Generate and summarize candidate proposals | `-n`, `--show`, `--csv`, `--md`, `--save`, `--width`, `--json`, `--no-color`, `--no-legend` |
| `run` | Run end-to-end pipeline and report results | `-n`, `--top`, `--bio`, `--no-stream`, `--narrative`, `--md`, `--save`, `--width`, `--json`, `--no-color`, `--no-legend` |
| `api` | Launch FastAPI server (uvicorn) | `--host`, `--port`, `--reload`, `--workers`, `--log-level`, `--open-docs` |
| `init` | Ensure data directory is populated | `--show` |
| `info` | Show environment and configuration details | `--json` |
| (global) | Global flags available to all | `--version` |

Example with streaming (default) and Markdown report:

```
python -m tsa.cli run -n 6 --top 5 --bio 5 --narrative --md outputs/pipeline_report.md
```

Auto-save reports with a single flag:

```
python -m tsa.cli run -n 6 --narrative --save outputs/
python -m tsa.cli plan -n 6 --save outputs/
```

Utility commands:

```
# Show environment/config and optional library availability
tsa info --json

# Ensure data is present and list files
tsa init --show

# Print installed version
tsa --version
```

### Installable CLI (pipx)

Install the CLI with pipx to get a global `tsa` command. If your default Python is 3.13, specify Python 3.12 (or 3.11) explicitly, since the pinned dependencies target <3.13:

```
# CLI only
pipx install --python python3.12 .
# Or include API extras
# pipx install --python python3.12 ".[api]"
```

If you don’t have Python 3.12 available, on macOS with Homebrew:

```
brew install python@3.12
pipx install --python /opt/homebrew/bin/python3.12 .
```

Then run:

```
tsa run -n 6
tsa plan -n 6 --json
tsa run -n 6 --narrative --md outputs/report.md
tsa info --json
tsa init --show
```

### Run the API via CLI

Start the FastAPI server (uvicorn) from the CLI:

```
# Local dev (ensure API deps installed, e.g. `pip install -r requirements-api.txt`)
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

Tips:
- Add `--json` to include structured JSON after the summary.
- Use `--top N` with `run` to show a Top‑N ranking table.
- Use `--no-color` to disable ANSI styling in terminals that don’t support it.
- Run `tsa info` to see environment/config details (version, data dir, etc.).
- Run `tsa init` to ensure the local data directory is populated.

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
POST /run_pipeline  {"n": 6}

{
  "proposals": [
    {"target": "MAPT", "direction": -1.0, "dose": 0.25, "duration": 28},
    {"target": "APP",  "direction": -1.0, "dose": 0.25, "duration": 14},
    {"target": "MAPT", "direction": -1.0, "dose": 0.25, "duration": 7},
    …
  ],
  "simulations": [
    {"biomarker_delta": [-0.089, 0.003, 0.041], "uncertainty": [0.059, 0.050, 0.054]},
    …
  ],
  "selected": {
    "choice": {"target": "MAPT", "direction": -1.0, "dose": 1.0, "duration": 14},
    "score": 1.899
  },
  "validation": {"rmse": 0.189},
  "summary": "Evaluated 6 candidate interventions and selected target MAPT (downregulation) at dose 1.0 for 14 days. Selection score: 1.899 (higher is better). Predictions include 3 biomarker effect values with uncertainties. Retrospective validation RMSE: 0.189."
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

Install dev extras for testing. For the full suite (including API tests), include API extras as well:

```
# Editable install with dev tools only (CLI tests)
pip install -e .[dev]

# Or include API deps to run all tests
pip install -e .[api,dev]

# Run the full test suite
pytest -q

# If you installed only [dev] and want to skip API tests
pytest -q tests/test_loop.py
# or
pytest -q -k "not test_status and not test_pipeline"
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
