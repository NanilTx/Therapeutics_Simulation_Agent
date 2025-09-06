from dataclasses import dataclass
from ..eval.validation import retrospective_validation

@dataclass
class ValidationAgent:
    def validate(self):
        return retrospective_validation()
