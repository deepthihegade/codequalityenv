from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

tasks = [
    {"level": "easy", "buggy_code": "def greet(name)\n    print('Hello, ' + name)", "error_hint": "syntax"},
    {"level": "medium", "buggy_code": "def find_max(nums):\n    max_val = 0\n    for n in nums:\n        if n > max_val:\n            max_val = n\n    return max_val", "error_hint": "logic"},
    {"level": "hard", "buggy_code": "def has_duplicate(nums):\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] == nums[j]:\n                return True\n    return False", "error_hint": "performance"},
]

state = {"index": 0, "step_count": 0, "episode_id": "ep-1"}

class Action(BaseModel):
    action_type: str
    line_number: Optional[int] = None
    code_patch: Optional[str] = None

@app.post("/reset")
def reset():
    state["index"] = 0
    state["step_count"] = 0
    t = tasks[0]
    return {"observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]}, "reward": 0.0, "done": False, "feedback": "New episode started!"}

@app.post("/step")
def step(action: Action):
    t = tasks[state["index"]]
    state["step_count"] += 1
    return {"observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]}, "reward": 0.5, "done": False, "feedback": "OK"}

@app.get("/state")
def get_state():
    return {"episode_id": state["episode_id"], "step_count": state["step_count"], "current_task": tasks[state["index"]]["level"]}
