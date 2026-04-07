from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    action_type: str
    line_number: Optional[int] = None
    code_patch: Optional[str] = None

class Observation(BaseModel):
    buggy_code: str
    task_level: str
    error_hint: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    feedback: str

class State(BaseModel):
    episode_id: str
    step_count: int
    current_task: str