CodeQualityEnv

An RL environment where an AI agent acts as a Python code reviewer, identifying and fixing real-world bugs.

Built with OpenEnv by team **Neurobytes**.

Environment Overview

| Field | Value |
|---|---|
| Observation | Buggy Python code + error hint + task level |
| Actions | `identify_bug`, `suggest_fix` |
| Reward Range | (0, 1) exclusive |
| Tasks | 3 (easy → medium → hard) 

What We Did?

1.Fixed Python runtime errors and exceptions

2.Resolved OpenEnv server entry point issues

3.Corrected project structure and file placement

4.Handled dependency conflicts and missing packages

5.Applied debugging techniques to real-world scenarios


Tech Stack
Python
OpenEnv
Flask / FastAPI
Git

Key Fixes
1. Entry Point Error
Fixed incorrect server reference (server.app:app → main:app)
2. File Structure Issue
Removed misplaced files and reorganized project layout
3. Dependency Issues
Installed and configured required packages properly

Tasks

| Level | Bug Type | Description |

| Easy | Missing return | Function computes but never returns value |
| Medium | Off-by-one | Binary search with wrong initial boundary |
| Hard | Closure bug | Lambda captures loop variable by reference |

Reward Function

| Outcome | Reward |

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

Future Scope:
Add automated debugging tools
Improve logging and monitoring
Support multiple environments
