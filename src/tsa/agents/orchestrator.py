from dataclasses import dataclass
from .data_agent import DataAgent
from .hypothesis_agent import HypothesisAgent
from .simulation_agent import SimulationAgent
from .critic_agent import CriticAgent
from .validation_agent import ValidationAgent

@dataclass
class Orchestrator:
    data: DataAgent
    hypo: HypothesisAgent
    sim: SimulationAgent
    critic: CriticAgent
    val: ValidationAgent
    def run_pipeline(self, n=6):
        self.data.ensure()
        props = self.hypo.propose(n)
        sims = [self.sim.run(p) for p in props]
        pick = self.critic.select_best(props, sims)
        val = self.val.validate()
        return {"proposals": props, "simulations": sims, "selected": pick, "validation": val}
