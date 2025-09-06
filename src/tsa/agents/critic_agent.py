from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class CriticAgent:
    def score(self, sims: List[Dict]) -> np.ndarray:
        scores = []
        for s in sims:
            eff = float(abs(np.array(s["biomarker_delta"]).astype(float)).sum())
            unc = float(np.array(s["uncertainty"]).astype(float).sum()) + 1e-6
            scores.append(eff / unc)
        return np.array(scores)
    def select_best(self, proposals: List[Dict], sims: List[Dict]) -> Dict:
        scores = self.score(sims)
        i = int(np.argmax(scores))
        return {"choice": proposals[i], "score": float(scores[i])}
