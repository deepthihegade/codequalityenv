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
        "buggy_code": (
            "def calculate_average(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        total += n\n"
            "    average = total / len(numbers)\n"
        ),
        "error_hint": "The function computes the average but never returns it. Add a return statement.",
        "test": (
            "result = calculate_average([10, 20, 30])\n"
            "assert result == 20.0, f'Expected 20.0 got {result}'\n"
            "result2 = calculate_average([1, 2, 3, 4])\n"
            "assert result2 == 2.5, f'Expected 2.5 got {result2}'\n"
            "print('PASS')\n"
        )
    },
    {
        "level": "medium",
        "buggy_code": (
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr)\n"  
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n"
        ),
        "error_hint": "Off-by-one error in the initial right boundary. Array indexing is zero-based.",
        "test": (
            "arr = [1, 3, 5, 7, 9, 11]\n"
            "assert binary_search(arr, 7) == 3, f'Expected 3 got {binary_search(arr, 7)}'\n"
            "assert binary_search(arr, 1) == 0, f'Expected 0 got {binary_search(arr, 1)}'\n"
            "assert binary_search(arr, 11) == 5, f'Expected 5 got {binary_search(arr, 11)}'\n"
            "assert binary_search(arr, 4) == -1, f'Expected -1 got {binary_search(arr, 4)}'\n"
            "print('PASS')\n"
        )
    },
    {
        "level": "hard",
        "buggy_code": (
            "def make_multiplier(factors):\n"
            "    funcs = []\n"
            "    for f in factors:\n"
            "        funcs.append(lambda x: x * f)\n"  
            "    return funcs\n"
        ),
        "error_hint": "Python closures capture variables by reference not by value. All lambdas end up using the last value of f.",
        "test": (
            "multipliers = make_multiplier([2, 3, 5])\n"
            "assert multipliers[0](10) == 20, f'Expected 20 got {multipliers[0](10)}'\n"
            "assert multipliers[1](10) == 30, f'Expected 30 got {multipliers[1](10)}'\n"
            "assert multipliers[2](10) == 50, f'Expected 50 got {multipliers[2](10)}'\n"
            "print('PASS')\n"
        )
    },
]

state = {"index": 0, "step_count": 0, "episode_id": "ep-1"}


class Action(BaseModel):
    action_type: str
    line_number: Optional[int] = None
    code_patch: Optional[str] = None


def evaluate_fix(code_patch: str, task: dict):
    
    if not code_patch or not code_patch.strip():
        return 0.01, "No fix provided."

    full_code = code_patch + "\n" + task["test"]
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            fname = f.name

        result = subprocess.run(
            ["python3", fname],
            capture_output=True, text=True, timeout=5
        )
        os.unlink(fname)

        if "PASS" in result.stdout:
            return 0.99, "All tests passed! Perfect fix 🎉"
        elif result.returncode == 0:
            return 0.50, "Code runs but tests didn't pass."
        else:
            if "SyntaxError" in result.stderr:
                return 0.01, f"Syntax error: {result.stderr[:120]}"
            if "IndexError" in result.stderr or "TypeError" in result.stderr:
                return 0.15, f"Runtime error: {result.stderr[:120]}"
            return 0.20, f"Fix has errors: {result.stderr[:120]}"

    except subprocess.TimeoutExpired:
        return 0.01, "Code timed out — possible infinite loop."
    except Exception as e:
        return 0.01, f"Error: {str(e)}"


@app.get("/")
def home():
    return {
        "message": "🚀 Code Quality OpenEnv — Neurobytes",
        "tasks": len(tasks),
        "reward_range": "(0, 1) exclusive",
        "endpoints": {
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state (GET)"
        }
    }


@app.post("/reset")
def reset():
    state["index"] = 0
    state["step_count"] = 0
    state["episode_id"] = "ep-1"
    t = tasks[0]
    return {
        "observation": {
            "buggy_code": t["buggy_code"],
            "task_level": t["level"],
            "error_hint": t["error_hint"]
        },
        "reward": 0.01,
        "done": False,
        "feedback": "New episode started! Review and fix the bug."
    }


@app.post("/step")
def step(action: Action):
    t = tasks[state["index"]]
    state["step_count"] += 1

    if action.action_type == "identify_bug":
        return {
            "observation": {
                "buggy_code": t["buggy_code"],
                "task_level": t["level"],
                "error_hint": t["error_hint"]
            },
            "reward": 0.30,
            "done": False,
            "feedback": "Bug identified! Now submit a fix using suggest_fix."
        }

    elif action.action_type == "suggest_fix":
        reward, feedback = evaluate_fix(action.code_patch, t)
        reward = round(max(0.01, min(0.99, reward)), 2)
        done = False

        if reward >= 0.80:
            if state["index"] < len(tasks) - 1:
                state["index"] += 1
                t = tasks[state["index"]]
                feedback += f" Moving to next task: {t['level']}"
            else:
                done = True
                feedback += " All tasks complete! 🏆"

        return {
            "observation": {
                "buggy_code": t["buggy_code"],
                "task_level": t["level"],
                "error_hint": t["error_hint"]
            },
            "reward": reward,
            "done": done,
            "feedback": feedback
        }

    return {
        "observation": {
            "buggy_code": t["buggy_code"],
            "task_level": t["level"],
            "error_hint": t["error_hint"]
        },
        "reward": 0.01,
        "done": False,
        "feedback": "Unknown action. Use 'identify_bug' or 'suggest_fix'."
    }


@app.get("/state")
def get_state():
    return {
        "episode_id": state["episode_id"],
        "step_count": state["step_count"],
        "current_task": tasks[state["index"]]["level"],
        "total_tasks": len(tasks)
    }


def run_inference(num_episodes: int = 1, steps_per_episode: int = 5):
    
    global state

    
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
    )
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    for episode in range(num_episodes):
        state["episode_id"] = f"ep-{episode + 1}"

        
        for task_idx in range(len(tasks)):
            state["index"] = task_idx
            state["step_count"] = 0
            task = tasks[task_idx]
            task_name = f"{task['level']}-task-{task_idx}"

            print(f"[START] task={task_name}", flush=True)

            total_reward = 0.0
            steps_taken = 0

            for step_num in range(1, steps_per_episode + 1):
                state["step_count"] += 1
                steps_taken += 1

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert Python code reviewer and bug fixer. "
                                    "Return ONLY the fixed Python code. "
                                    "No explanation, no markdown, no backticks. "
                                    "Just raw working Python code."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Fix this buggy Python code:\n\n"
                                    f"{task['buggy_code']}\n\n"
                                    f"Hint: {task['error_hint']}"
                                )
                            }
                        ],
                        timeout=60,
                        max_tokens=300
                    )
                    code_patch = response.choices[0].message.content.strip()
                    
                    if code_patch.startswith("```"):
                        lines = code_patch.split("\n")
                        code_patch = "\n".join(lines[1:-1])

                except Exception as e:
                    print(f"LLM call failed: {e}", flush=True)
                    code_patch = task["buggy_code"]  

                reward, feedback = evaluate_fix(code_patch, task)
                reward = round(max(0.01, min(0.99, reward)), 2)
                total_reward += reward

                print(f"[STEP] step={step_num} reward={reward:.2f}", flush=True)

                if reward >= 0.80:
                    break

            final_score = round(min(total_reward / steps_taken, 0.99), 2)
            final_score = max(0.01, final_score)
            print(f"[END] task={task_name} score={final_score:.2f} steps={steps_taken}", flush=True)


if __name__ == "__main__":

    def _hard_timeout(signum, frame):
        print("[END] task=timeout score=0.01 steps=0", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGALRM, _hard_timeout)
    signal.alarm(18 * 60)  

    run_inference(num_episodes=1, steps_per_episode=5)
