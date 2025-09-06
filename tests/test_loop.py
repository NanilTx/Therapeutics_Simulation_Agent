from tsa.agents.orchestrator import Orchestrator
from tsa.agents.data_agent import DataAgent
from tsa.agents.hypothesis_agent import HypothesisAgent
from tsa.agents.simulation_agent import SimulationAgent
from tsa.agents.critic_agent import CriticAgent
from tsa.agents.validation_agent import ValidationAgent

def test_orchestrator():
    orch = Orchestrator(
        data=DataAgent(),
        hypo=HypothesisAgent(candidate_targets=["MAPT","APP"], doses=[0.5], durations=[7,14]),
        sim=SimulationAgent(latent_dim=16),
        critic=CriticAgent(),
        val=ValidationAgent(),
    )
    out = orch.run_pipeline(n=3)
    assert "proposals" in out and "selected" in out
