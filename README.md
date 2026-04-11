CodeQualityEnv

An RL environment where an AI agent acts as a Python code reviewer, identifying and fixing real-world bugs.

Built with OpenEnv by team **Neurobytes**.

Environment Overview

| Field | Value |
|---|---|
| Observation | Buggy Python code + error hint + task level |
| Actions | `identify_bug`, `suggest_fix` |
| Reward Range | (0, 1) exclusive |
| Tasks | 3 (easy → medium → hard) |

Tasks

| Level | Bug Type | Description |
|---|---|---|
| Easy | Missing return | Function computes but never returns value |
| Medium | Off-by-one | Binary search with wrong initial boundary |
| Hard | Closure bug | Lambda captures loop variable by reference |

Reward Function

| Outcome | Reward |
|---|---|
| All tests pass | 0.99 |
| Code runs, tests fail | 0.50 |
| Runtime error | 0.15–0.20 |
| Syntax error / no fix | 0.01 |
| Bug identified (partial) | 0.30 |

API Endpoints

`POST /reset` — Start new episode
`POST /step` — Submit action
`GET /state` — Current episode state

Setup

```bash
git clone https://github.com/deepthihegade/codequalityenv
cd codequalityenv
pip install fastapi uvicorn openai pydantic
python inference.py
```

HF Space

https://huggingface.co/spaces/MeghanaK4/NEUROBYTRES
