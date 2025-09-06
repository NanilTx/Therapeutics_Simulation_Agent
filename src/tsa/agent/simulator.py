from dataclasses import dataclass
import numpy as np

@dataclass
class Perturbation:
    target: str
    direction: float
    dose: float
    duration: int

class PerturbationSimulator:
    def __init__(self, latent_dim: int = 16, seed: int = 1337):
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        self.target_embed = {}
    def _embed_target(self, target: str):
        if target not in self.target_embed:
            self.target_embed[target] = self.rng.normal(0, 1, self.latent_dim)
        return self.target_embed[target]
    def simulate_delta(self, p: Perturbation):
        base = self._embed_target(p.target)
        scale = np.tanh(p.dose) * np.log1p(p.duration) * np.sign(p.direction)
        noise = self.rng.normal(0, 0.05, self.latent_dim)
        return base * 0.1 * scale + noise
    def predict_biomarker_delta(self, latent_delta):
        w = self.rng.normal(0, 0.2, (self.latent_dim, 3))
        pred = latent_delta @ w
        sigma = np.abs(pred) * 0.1 + 0.05
        return pred, sigma
