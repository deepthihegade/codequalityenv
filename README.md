CodeQualityEnv 

An RL environment where an AI agent learns to review and fix buggy Python code.

Built with OpenEnv by team Neurobytes.

What It Does?
The agent is given buggy Python code and must:
1. Identify where the bug is
2. Suggest a corrected fix

It gets scored from 0.0 to 1.0 based on how good the fix will be.

3.Tasks
 Level 
 Type
 
Description:
Easy  Syntax  Fix a missing colon.
Medium  Logic  Fix wrong initial value in loop.
Hard  Performance  Optimize O(n²) to O(n).

Observation Space:
 Buggy Python code snippet
 Error hint (syntax / logic /            performance)
 Task level (easy / medium / hard)

Action Space:
 identify_bug(line_number) → partial     reward 0.3
 suggest_fix(code_patch) → full reward   up to 1.0

Reward Function:
 Outcome 
 Reward 
 All tests pass = 1.0 
 Code runs, tests fail = 0.5 
 Bug identified correctly = 0.3 
 Fix has errors = 0.2 
 Broken/no fix = 0.0 

How To Run:
 git clone:
https://github.com/deepthihegade/codequalityenv
 cd codequalityenv
 pip install fastapi uvicorn
 python3 baseline.py


Expected Output
 Task: easy → Reward:1.0 
 Task: medium → Reward:1.0 
 Task: hard → Reward:1.0 

Team Neurobyte
