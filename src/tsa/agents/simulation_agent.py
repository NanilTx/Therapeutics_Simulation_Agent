from dataclasses import dataclass
from typing import Dict
from ..agent.simulator import Perturbation, PerturbationSimulator

@dataclass
class SimulationAgent:
    latent_dim: int = 16
    seed: int = 1337
    def run(self, plan: Dict) -> Dict:
        sim = PerturbationSimulator(self.latent_dim, self.seed)
        p = Perturbation(**plan)
        delta = sim.simulate_delta(p)
        pred, sigma = sim.predict_biomarker_delta(delta)
        return {"biomarker_delta": pred.tolist(), "uncertainty": sigma.tolist()}
