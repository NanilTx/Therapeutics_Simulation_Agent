from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class HypothesisAgent:
    candidate_targets: List[str]
    doses: List[float]
    durations: List[int]
    def propose(self, n: int = 6) -> List[Dict]:
        rng = np.random.default_rng(123)
        space = [(t,d,k) for t in self.candidate_targets for d in self.doses for k in self.durations]
        rng.shuffle(space)
        return [dict(target=t, direction=-1.0, dose=float(d), duration=int(k)) for t,d,k in space[:n]]
