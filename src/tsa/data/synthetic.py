from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from ..config import DATA_DIR, RANDOM_SEED
RNG = np.random.default_rng(RANDOM_SEED)

def _mk_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def gen_expression(n_cells=800, n_genes=200, n_groups=4):
    base = RNG.normal(0, 1, (n_cells, n_genes))
    groups = RNG.integers(0, n_groups, size=n_cells)
    for g in range(n_groups):
        sl = slice(g*(n_genes//n_groups),(g+1)*(n_genes//n_groups))
        base[groups==g, sl] += RNG.normal(1.2, 0.25)
    idx = [f"cell_{i}" for i in range(n_cells)]
    cols = [f"g{i}" for i in range(n_genes)]
    return pd.DataFrame(base, index=idx, columns=cols), pd.Series(groups, index=idx, name="group")

def gen_proteomics(n_samples=300, n_prots=50):
    X = RNG.normal(0, 1, (n_samples, n_prots)) + RNG.normal(0.4, 0.3, (n_samples, 1))
    idx = [f"sample_{i}" for i in range(n_samples)]
    cols = [f"p{i}" for i in range(n_prots)]
    return pd.DataFrame(X, index=idx, columns=cols)

def gen_imaging_proxy(n_samples=300):
    return pd.DataFrame({
        "id": [f"sample_{i}" for i in range(n_samples)],
        "global_suvr": RNG.normal(1.2, 0.15, n_samples),
        "hippocampal_vol": RNG.normal(3.2, 0.4, n_samples)
    }).set_index("id")

def gen_clinical_traj(n_samples=300, tpoints=6):
    traj = RNG.normal(0, 0.2, (n_samples, tpoints)).cumsum(axis=1)
    return pd.DataFrame(traj, index=[f"sample_{i}" for i in range(n_samples)],
                        columns=[f"m{j}" for j in range(tpoints)])

def write_all():
    _mk_dir()
    expr, groups = gen_expression()
    expr.to_csv(DATA_DIR / 'expression.csv')
    groups.to_frame().to_csv(DATA_DIR / 'labels.csv')
    gen_proteomics().to_csv(DATA_DIR / 'proteomics.csv')
    gen_imaging_proxy().to_csv(DATA_DIR / 'imaging.csv')
    gen_clinical_traj().to_csv(DATA_DIR / 'clinical.csv')
    return {
        "expression": str(DATA_DIR / 'expression.csv'),
        "labels": str(DATA_DIR / 'labels.csv'),
        "proteomics": str(DATA_DIR / 'proteomics.csv'),
        "imaging": str(DATA_DIR / 'imaging.csv'),
        "clinical": str(DATA_DIR / 'clinical.csv')
    }
