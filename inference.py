import os
import sys
from openai import OpenAI
from server.environment import Environment
from models import Action

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI client pointing to HuggingFace
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def ask_llm(buggy_code: str, error_hint: str, task_level: str) -> str:
    prompt = f"""You are a Python code reviewer.
You are given buggy Python code with a {error_hint} error ({task_level} level).
Fix the bug and return ONLY the corrected Python code, nothing else.

Buggy code:
{buggy_code}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def run_task(env: Environment, task_name: str) -> float:
    result = env.reset()
    rewards = []
    done = False
    step = 0

    print(f"[START] task={task_name} env=code-quality-env model={MODEL_NAME}")

    while not done and step < 5:
        step += 1
        obs = result.observation

        # Ask LLM to fix the code
        fix = ask_llm(obs.buggy_code, obs.error_hint, obs.task_level)

        # Clean up LLM response (remove markdown code blocks if any)
        fix = fix.replace("```python", "").replace("```", "").strip()

        action = Action(action_type="suggest_fix", code_patch=fix)
        result = env.step(action)
        rewards.append(result.reward)
        done = result.done

        print(f"[STEP] step={step} action=suggest_fix reward={result.reward:.2f} done={str(done).lower()} error=null")

        if result.reward >= 1.0:
            break

    score = max(rewards) if rewards else 0.0
    success = score >= 1.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")
    return score

def main():
    env = Environment()
    tasks = ["easy", "medium", "hard"]
    total_score = 0.0

    for task in tasks:
        score = run_task(env, task)
        total_score += score

    avg = total_score / len(tasks)
    print(f"\nAverage score: {avg:.2f}")

if __name__ == "__main__":
    main()