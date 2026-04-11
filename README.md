CodeReview OpenEnv — Neurobytes

 -A real-world RL environment where an AI agent acts as a **senior Python code reviewer**.  
 -Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, Round 1.  
Team: Neurobytes


 Real-World Task

The agent simulates a **code review workflow** used daily by engineers at companies like Meta, Google, and Microsoft.

Given a buggy Python function, the agent must:
1. **Inspect** the code carefully before acting
2. **Identify** the exact line containing the bug
3. **Explain** the root cause of the bug
4. **Fix** the code so all tests pass
5. **Verify** the fix by running tests

This mirrors how real engineers review PRs — not just patching blindly, but reasoning through the problem step by step.





 Action Space

The agent has 5 actions available — designed to reward thoughtful, systematic review:

| Action | Required Fields | Reward |
|--------|----------|-----------|
| `inspect_code` | — | +0.15 first time, +0.05 repeat |
| `identify_bug` | `line_number: int` | +0.40 correct, +0.10 wrong |
| `explain_bug` | `explanation: str` | +0.30 to +0.40 |
| `suggest_fix` | `code_patch: str` | +0.05 to +0.99 based on tests |
| `run_tests` | — | 0.5× of last fix reward |

Workflow bonus: Agent gets +0.03 extra on `suggest_fix` for each completed prior step (inspect → identify → explain). Maximum bonus: +0.09.


 Observation Space

Each observation contains:

{
  "buggy_code": "def binary_search(arr, target):\n    left, right = 0, len(arr)...",
  "task_level": "medium",
  "task_title": "Off-by-One in Binary Search",
  "error_hint": "right boundary is len(arr) but arrays are zero-indexed...",
  "error_type": "off-by-one",
  "step_count": 2,
  "inspected": true,
  "identified": false,
  "explained": false
}




Reward Function

Rewards are dense at every step and strictly between (0, 1) exclusive:

| Outcome | Reward |
|--------|-----|
| All tests pass + full workflow | 0.99 + up to 0.09 bonus → clamped to 0.99 |
| All tests pass | 0.99 |
| Code runs, tests fail | 0.50 |
| Index/Type error in fix | 0.10 |
| Syntax error in fix | 0.05 |
| Correct bug line identified | 0.40 |
| Wrong line identified | 0.10 |
| Good explanation | 0.30–0.40 |
| First inspect | 0.15 |
| No fix / unknown action | 0.01 |

All rewards clamped strictly to (0.01, 0.99)— never 0.0 or 1.0.



 API Endpoints

| Method | Endpoint | Description |
| GET | `/` | Environment info |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit action |
| GET | `/state` | Current episode state |



Setup & Run Locally

cd codequalityenv
pip install fastapi uvicorn openai pydantic

Start the server:
uvicorn inference:app --host 0.0.0.0 --port 8000


Run inference (requires judges' env vars):

export API_KEY=your_key
export API_BASE_URL=https://your-proxy/v1
export MODEL_NAME=gpt-4o-mini
python inference.py

 Docker


docker build -t codereview-env .
docker run -p 8000:8000 \
  -e API_KEY=your_key \
  -e API_BASE_URL=https://your-proxy/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  codereview-env


HF Space

[https://huggingface.co/spaces/MeghanaK4/NEUROBYTRES]


Team Neurobytes
