import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from ..data.synthetic import write_all
from ..service import orch, _summarize_plan, _summarize_pipeline

app = FastAPI(title="Therapeutic Simulation Agent", version="0.2.0")
logger = logging.getLogger("tsa.api")

@app.get("/status")
def status():
    return {"ok": True}

@app.get("/")
def root():
    # Redirect to the interactive API docs for convenience
    return RedirectResponse(url="/docs")

@app.post("/bootstrap_synth_data")
def bootstrap_synth_data():
    return write_all()

class PlanRequest(BaseModel):
    n: int = 6

@app.post("/plan")
def plan(req: PlanRequest):
    proposals = orch.hypo.propose(n=req.n)
    summary = _summarize_plan(proposals)
    try:
        logger.info("PLAN summary: %s", summary)
    except Exception:
        pass
    return {"proposals": proposals, "summary": summary}

@app.post("/run_pipeline")
def run_pipeline(req: PlanRequest):
    out = orch.run_pipeline(n=req.n)
    out["summary"] = _summarize_pipeline(out)
    try:
        logger.info("PIPELINE summary: %s", out["summary"])
    except Exception:
        pass
    return out
