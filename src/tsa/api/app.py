import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from ..agents.orchestrator import Orchestrator
from ..agents.data_agent import DataAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.simulation_agent import SimulationAgent
from ..agents.critic_agent import CriticAgent
from ..agents.validation_agent import ValidationAgent
from ..data.synthetic import write_all

app = FastAPI(title="Therapeutic Simulation Agent", version="0.2.0")
logger = logging.getLogger("tsa.api")

orch = Orchestrator(
    data=DataAgent(),
    hypo=HypothesisAgent(candidate_targets=["MAPT","APP","LRP1","BIN1"], doses=[0.25,0.5,1.0], durations=[7,14,28]),
    sim=SimulationAgent(latent_dim=16),
    critic=CriticAgent(),
    val=ValidationAgent(),
)

def _summarize_plan(proposals: list[dict]) -> str:
    if not proposals:
        return "No proposals were generated."
    n = len(proposals)
    targets = sorted({p.get("target") for p in proposals})
    doses = sorted({p.get("dose") for p in proposals})
    durations = sorted({p.get("duration") for p in proposals})
    example = proposals[0]
    dir_txt = "downregulation" if float(example.get("direction", -1.0)) < 0 else "upregulation"
    return (
        f"Proposed {n} intervention candidates across targets {targets}, "
        f"doses {doses}, and durations {durations}. Example: target {example.get('target')} "
        f"({dir_txt}) at dose {example.get('dose')} for {example.get('duration')} days."
    )

def _summarize_pipeline(out: dict) -> str:
    try:
        n = len(out.get("proposals", []))
        sel = out.get("selected", {}).get("choice", {})
        score = out.get("selected", {}).get("score")
        sims = out.get("simulations", [])
        val = out.get("validation", {})
        rmse = val.get("rmse")
        dim = len(sims[0].get("biomarker_delta", [])) if sims else 0
        dir_txt = "downregulation" if float(sel.get("direction", -1.0)) < 0 else "upregulation"
        return (
            f"Evaluated {n} candidate interventions and selected target {sel.get('target')} "
            f"({dir_txt}) at dose {sel.get('dose')} for {sel.get('duration')} days. "
            f"Selection score: {score:.3f} (higher is better). "
            f"Predictions include {dim} biomarker effect values with uncertainties. "
            f"Retrospective validation RMSE: {rmse:.3f}."
        )
    except Exception:
        return "Completed pipeline run and returned proposals, simulations, selected choice, and validation."

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
