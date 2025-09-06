from pathlib import Path
import pandas as pd
from ..config import DATA_DIR

def load_matrix(name: str = "expression.csv"):
    path = Path(DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Place CSV in {DATA_DIR}")
    return pd.read_csv(path, index_col=0)
