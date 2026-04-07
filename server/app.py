from fastapi import FastAPI
from environment import Environment
from models import Action
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()
env = Environment()

@app.post("/reset")
def reset():
    result = env.reset()
    return {
        "observation": {
            "buggy_code": result.observation.buggy_code,
            "task_level": result.observation.task_level,
            "error_hint": result.observation.error_hint
        },
        "reward": result.reward,
        "done": result.done,
        "feedback": result.feedback
    }

@app.post("/step")
def step(action: dict):
    a = Action(
        action_type=action["action_type"],
        line_number=action.get("line_number"),
        code_patch=action.get("code_patch")
    )
    result = env.step(a)
    return {
        "observation": {
            "buggy_code": result.observation.buggy_code,
            "task_level": result.observation.task_level,
            "error_hint": result.observation.error_hint
        },
        "reward": result.reward,
        "done": result.done,
        "feedback": result.feedback
    }

@app.get("/state")
def state():
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "current_task": s.current_task
    }