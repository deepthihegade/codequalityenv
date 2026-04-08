from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess
import tempfile
import os
from openai import OpenAI
import signal
import sys

app = FastAPI()

tasks = [
    {
        "level": "easy",
        "buggy_code": "def greet(name)\n    print('Hello, ' + name)",
        "error_hint": "syntax",
        "test": "greet('World')\nprint('PASS')"
    },
    {
        "level": "medium",
        "buggy_code": "def find_max(nums):\n    max_val = 0\n    for n in nums:\n        if n > max_val:\n            max_val = n\n    return max_val",
        "error_hint": "logic",
        "test": "result = find_max([-5, -1, -3])\nassert result == -1, f'Expected -1 got {result}'\nprint('PASS')"
    },
    {
        "level": "hard",
        "buggy_code": "def has_duplicate(nums):\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] == nums[j]:\n                return True\n    return False",
        "error_hint": "performance",
        "test": "assert has_duplicate([1,2,3,1]) == True\nassert has_duplicate([1,2,3]) == False\nprint('PASS')"
    },
]

state = {"index": 0, "step_count": 0, "episode_id": "ep-1"}


class Action(BaseModel):
    action_type: str
    line_number: Optional[int] = None
    code_patch: Optional[str] = None


def evaluate_fix(code_patch: str, task: dict):
    if not code_patch:
        return 0.01, "No fix provided."
    full_code = code_patch + "\n" + task["test"]
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            fname = f.name
        result = subprocess.run(["python3", fname], capture_output=True, text=True, timeout=5)
        os.unlink(fname)
        if "PASS" in result.stdout:
            return 0.99, "All tests passed! Perfect fix 🎉"
        elif result.returncode == 0:
            return 0.5, "Code runs but tests didn't pass."
        else:
            if "SyntaxError" in result.stderr:
                return 0.01, f"Syntax error: {result.stderr[:120]}"
            return 0.2, f"Fix has errors: {result.stderr[:120]}"
    except subprocess.TimeoutExpired:
        return 0.01, "Code timed out — possible infinite loop."
    except Exception as e:
        return 0.01, f"Error: {str(e)}"


@app.post("/reset")
def reset():
    state["index"] = 0
    state["step_count"] = 0
    state["episode_id"] = "ep-1"
    t = tasks[0]
    return {
        "observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]},
        "reward": 0.01,
        "done": False,
        "feedback": "New episode started! Fix the bug."
    }


@app.post("/step")
def step(action: Action):
    t = tasks[state["index"]]
    state["step_count"] += 1

    if action.action_type == "identify_bug":
        return {
            "observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]},
            "reward": 0.3,
            "done": False,
            "feedback": "Bug identified! Now try fixing it."
        }

    elif action.action_type == "suggest_fix":
        reward, feedback = evaluate_fix(action.code_patch, t)
        done = False

        if reward >= 0.8:
            if state["index"] < len(tasks) - 1:
                state["index"] += 1
                t = tasks[state["index"]]
                feedback += f" Moving to next task: {t['level']}"
            else:
                done = True
                feedback += " All tasks complete! 🏆"

        return {
            "observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]},
            "reward": reward,
            "done": done,
            "feedback": feedback
        }

    return {
        "observation": {"buggy_code": t["buggy_code"], "task_level": t["level"], "error_hint": t["error_hint"]},
        "reward": 0.0,
        "done": False,
        "feedback": "Unknown action. Use 'identify_bug' or 'suggest_fix'."
    }


@app.get("/state")
def get_state():
    return {
        "episode_id": state["episode_id"],
        "step_count": state["step_count"],
        "current_task": tasks[state["index"]]["level"]
    }


@app.get("/")
def home():
    return {
        "message": "🚀 Code Quality OpenEnv Running",
        "endpoints": {
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state (GET)"
        }
    }

