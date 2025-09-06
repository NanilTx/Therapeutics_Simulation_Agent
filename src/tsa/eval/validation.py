import numpy as np

def retrospective_validation(n=50, seed=1337):
    rng = np.random.default_rng(seed)
    observed = rng.normal(0, 1, (n, 3))
    predicted = observed + rng.normal(0, 0.2, (n, 3))
    rmse = float(np.sqrt(((observed - predicted) ** 2).mean()))
    return {"rmse": rmse}
