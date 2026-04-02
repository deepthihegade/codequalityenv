from dataclasses import dataclass
from typing import Optional

@dataclass
class Action:
    action_type: str        # "identify_bug" or "suggest_fix"
    line_number: Optional[int] = None
    code_patch: Optional[str] = None

@dataclass
class Observation:
    buggy_code: str
    task_level: str         # "easy", "medium", "hard"
    error_hint: str         # "syntax", "logic", "performance"

@dataclass
class StepResult:
    observation: Observation
    reward: float           # 0.0 to 1.0
    done: bool
    feedback: str

@dataclass
class State:
    episode_id: str
    step_count: int
    current_task: str