from dataclasses import dataclass
from ..config import DATA_DIR
from ..data.synthetic import write_all

@dataclass
class DataAgent:
    ensure_synth: bool = True
    def ensure(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        expr = DATA_DIR / "expression.csv"
        if not expr.exists() and self.ensure_synth:
            return write_all()
        return {"expression": str(expr)}
