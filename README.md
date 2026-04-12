CodeQualityEnv — PyTorch Bug Review RL Environment

An RL environment where an AI agent acts as a **senior PyTorch engineer**, 
reviewing and fixing real ML training bugs. Built for the 
**Meta PyTorch OpenEnv Hackathon** by team **Neurobytes**.



What It Does

The agent is given real, subtle PyTorch bugs and must follow a structured 
code review workflow:

1. `inspect_code` — Read and understand the buggy code
2. `identify_bug` — Pinpoint the exact line with the bug
3. `explain_bug` — Explain technically WHY it's a bug
4. `suggest_fix` — Submit corrected Python code
5. `run_tests` — Verify the fix passes PyTorch test cases


 The 5 PyTorch Tasks

| Level | Bug Type | Description |
|-------|----------|-------------|
| Easy | Wrong Reduction | `.sum()` instead of `.mean()` in MSE loss |
| Medium | Missing zero_grad | Gradient accumulation across steps |
| Hard | dtype Mismatch | float32 input vs float64 model crash |
| Expert | detach() abuse | Blocks gradient flow to weight network |
| Master | Wrong Norm Axis | `dim=1` instead of `dim=0` in BatchNorm |



 Reward Function

| Action | Reward |
|--------|--------|
| inspect_code (first time) | 0.15 |
| identify_bug (correct line) | 0.40 |
| explain_bug (detailed) | 0.30–0.40 |
| suggest_fix (tests pass) | 0.99 + workflow bonus |
| suggest_fix (partial) | 0.05–0.50 |
| run_tests | up to 0.50 |

> All rewards are strictly **(0.0, 1.0) exclusive** — never exactly 0 or 1.


🔁 Observation Space


{
  "buggy_code": "...",
  "task_level": "easy/medium/hard/expert/master",
  "task_title": "...",
  "task_description": "...",
  "error_hint": "...",
  "error_type": "...",
  "step_count": 3,
  "inspected": true,
  "identified": false,
  "explained": false
}


 Action Space

{ "action_type": "inspect_code" }
{ "action_type": "identify_bug", "line_number": 6 }
{ "action_type": "explain_bug", "explanation": "..." }
{ "action_type": "suggest_fix", "code_patch": "..." }
{ "action_type": "run_tests" }



 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment info |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take an action |
| GET | `/state` | Current state |



 Inference (LLM Agent)

API_BASE_URL=<proxy_url> API_KEY=<key> python inference.py


The agent uses the LLM to explain bugs and generate fixes across all 5 tasks.


Team Neurobytes
