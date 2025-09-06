from fastapi import FastAPI
from pydantic import BaseModel
from ..agents.orchestrator import Orchestrator
from ..agents.data_agent import DataAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.simulation_agent import SimulationAgent
from ..agents.critic_agent import CriticAgent
from ..agents.validation_agent import ValidationAgent
from ..data.synthetic import write_all

app = FastAPI(title="Therapeutic Simulation Agent", version="0.2.0")

orch = Orchestrator(
    data=DataAgent(),
    hypo=HypothesisAgent(candidate_targets=["MAPT","APP","LRP1","BIN1"], doses=[0.25,0.5,1.0], durations=[7,14,28]),
    sim=SimulationAgent(latent_dim=16),
    critic=CriticAgent(),
    val=ValidationAgent(),
)

@app.get("/status")
def status():
    return {"ok": True}

@app.post("/bootstrap_synth_data")
def bootstrap_synth_data():
    return write_all()

class PlanRequest(BaseModel):
    n: int = 6

@app.post("/plan")
def plan(req: PlanRequest):
    return {"proposals": orch.hypo.propose(n=req.n)}

@app.post("/run_pipeline")
def run_pipeline(req: PlanRequest):
    return orch.run_pipeline(n=req.n)
