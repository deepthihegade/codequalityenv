from fastapi import FastAPI
from environment import Environment
from models import Action
import json

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
def step(action: Action):
    result = env.step(action)
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