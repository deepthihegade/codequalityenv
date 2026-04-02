import ast
import subprocess
import tempfile
import os
from models import Action, Observation, StepResult, State
import uuid

# Our 3 tasks: easy → medium → hard
TASKS = [
    {
        "level": "easy",
        "error_hint": "syntax",
        "buggy_code": """def greet(name)
    print("Hello, " + name)""",
        "fixed_code": """def greet(name):
    print("Hello, " + name)""",
        "test": """
greet("World")
print("PASS")
"""
    },
    {
        "level": "medium",
        "error_hint": "logic",
        "buggy_code": """def find_max(nums):
    max_val = 0
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val""",
        "fixed_code": """def find_max(nums):
    max_val = nums[0]
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val""",
        "test": """
result = find_max([-5, -1, -3])
assert result == -1, f"Expected -1 got {result}"
print("PASS")
"""
    },
    {
        "level": "hard",
        "error_hint": "performance",
        "buggy_code": """def has_duplicate(nums):
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j and nums[i] == nums[j]:
                return True
    return False""",
        "fixed_code": """def has_duplicate(nums):
    return len(nums) != len(set(nums))""",
        "test": """
assert has_duplicate([1,2,3,1]) == True
assert has_duplicate([1,2,3]) == False
print("PASS")
"""
    }
]

class Environment:
    def __init__(self):
        self.current_task_index = 0
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())

    def reset(self) -> StepResult:
        self.current_task_index = 0
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        task = TASKS[0]
        return StepResult(
            observation=Observation(
                buggy_code=task["buggy_code"],
                task_level=task["level"],
                error_hint=task["error_hint"]
            ),
            reward=0.0,
            done=False,
            feedback="New episode started. Fix the bug!"
        )

    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        task = TASKS[self.current_task_index]

        if action.action_type == "identify_bug":
            # Partial reward for identifying correctly
            reward = 0.3
            feedback = "Bug identified! Now try fixing it."
            done = False
            obs = Observation(
                buggy_code=task["buggy_code"],
                task_level=task["level"],
                error_hint=task["error_hint"]
            )
            return StepResult(observation=obs, reward=reward, done=done, feedback=feedback)

        elif action.action_type == "suggest_fix":
            reward, feedback = self._evaluate_fix(action.code_patch, task)
            # Move to next task if fix is good
            if reward >= 0.8 and self.current_task_index < len(TASKS) - 1:
                self.current_task_index += 1
                next_task = TASKS[self.current_task_index]
                obs = Observation(
                    buggy_code=next_task["buggy_code"],
                    task_level=next_task["level"],
                    error_hint=next_task["error_hint"]
                )
                done = False
            else:
                obs = Observation(
                    buggy_code=task["buggy_code"],
                    task_level=task["level"],
                    error_hint=task["error_hint"]
                )
                done = self.current_task_index == len(TASKS) - 1 and reward >= 0.8

            return StepResult(observation=obs, reward=reward, done=done, feedback=feedback)

        return StepResult(
            observation=Observation(
                buggy_code=task["buggy_code"],
                task_level=task["level"],
                error_hint=task["error_hint"]
            ),
            reward=0.0,
            done=False,
            feedback="Unknown action."
        )

    def _evaluate_fix(self, code_patch: str, task: dict):
        if not code_patch:
            return 0.0, "No fix provided."

        # Run the fix + test together
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
                return 1.0, "All tests passed! Perfect fix 🎉"
            elif result.returncode == 0:
                return 0.5, "Code runs but tests didn't pass."
            else:
                if "SyntaxError" in result.stderr:
                    return 0.0, f"Syntax error in fix: {result.stderr[:100]}"
                return 0.2, f"Fix has errors: {result.stderr[:100]}"

        except subprocess.TimeoutExpired:
            return 0.0, "Code timed out — possible infinite loop."
        except Exception as e:
            return 0.0, f"Error running fix: {str(e)}"

    @property
    def state(self) -> State:
        return State(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_task=TASKS[self.current_task_index]["level"]
        )