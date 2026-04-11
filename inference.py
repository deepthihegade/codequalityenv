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

TASKS = [
    {
        "level": "easy",
        "title": "Wrong Loss Reduction — Sum Instead of Mean",
        "description": (
            "A training loop computes MSE loss using .sum() instead of .mean(). "
            "This makes the loss scale with batch size, so doubling the batch "
            "doubles the loss and breaks learning rate tuning."
        ),
        "buggy_code": (
            "import torch\n"
            "\n"
            "def compute_loss(predictions, targets):\n"
            "    \"\"\"Compute MSE loss between predictions and targets.\"\"\"\n"
            "    diff = predictions - targets\n"
            "    return (diff ** 2).sum()  # Bug: should be .mean() not .sum()\n"
        ),
        "error_hint": (
            "MSE = mean of squared errors. Using .sum() makes loss proportional to batch size — "
            "a batch of 100 gives 100x the loss of a batch of 1. "
            "Replace .sum() with .mean()."
        ),
        "error_type": "wrong-reduction",
        "line_hint": 6,
        "test": (
            "import torch\n"
            "preds = torch.tensor([1.0, 2.0, 3.0])\n"
            "tgts  = torch.tensor([1.5, 2.5, 3.5])\n"
            "loss = compute_loss(preds, tgts)\n"
            "assert abs(loss.item() - 0.25) < 1e-5, f'Expected MSE=0.25 got {loss.item()}'\n"
            "preds2 = torch.tensor([0.0, 0.0])\n"
            "tgts2  = torch.tensor([1.0, 1.0])\n"
            "loss2 = compute_loss(preds2, tgts2)\n"
            "assert abs(loss2.item() - 1.0) < 1e-5, f'Expected MSE=1.0 got {loss2.item()}'\n"
            "print('PASS')\n"
        ),
    },
    {
        "level": "medium",
        "title": "Missing optimizer.zero_grad() Causes Gradient Accumulation",
        "description": (
            "A PyTorch training loop is missing optimizer.zero_grad() before loss.backward(). "
            "Gradients accumulate across steps, causing exploding gradient norms "
            "and unstable or diverging training."
        ),
        "buggy_code": (
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "def train_one_epoch(model, inputs, targets, optimizer, criterion, steps=3):\n"
            "    losses = []\n"
            "    grad_norms = []\n"
            "    for step in range(steps):\n"
            "        outputs = model(inputs)\n"
            "        loss = criterion(outputs, targets)\n"
            "        # Bug: missing optimizer.zero_grad() here!\n"
            "        loss.backward()\n"
            "        gn = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)\n"
            "        grad_norms.append(gn)\n"
            "        optimizer.step()\n"
            "        losses.append(loss.item())\n"
            "    return losses, grad_norms\n"
        ),
        "error_hint": (
            "In PyTorch, gradients accumulate by default. "
            "Call optimizer.zero_grad() at the START of every training step, "
            "before loss.backward(). Without it, each step adds to previous gradients "
            "causing the gradient norm to grow each step."
        ),
        "error_type": "missing-zero-grad",
        "line_hint": 10,
        "test": (
            "import torch\n"
            "import torch.nn as nn\n"
            "torch.manual_seed(42)\n"
            "model = nn.Linear(2, 1)\n"
            "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n"
            "criterion = nn.MSELoss()\n"
            "inputs  = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
            "targets = torch.tensor([[1.0], [2.0]])\n"
            "losses, grad_norms = train_one_epoch(model, inputs, targets, optimizer, criterion, steps=3)\n"
            "ratio = grad_norms[1] / (grad_norms[0] + 1e-8)\n"
            "assert ratio < 1.5, (\n"
            "    f'Gradient norms exploding (ratio={ratio:.2f}) — '\n"
            "    f'missing zero_grad causes accumulation! norms={grad_norms}'\n"
            ")\n"
            "assert len(losses) == 3\n"
            "print('PASS')\n"
        ),
    },
    {
        "level": "hard",
        "title": "Tensor dtype Mismatch Crashes Forward Pass",
        "description": (
            "A model is cast to float64 (.double()) but the input tensor stays float32. "
            "PyTorch raises a RuntimeError when tensors of different dtypes are used in the same operation. "
            "This is the CPU/GPU analogue — a dtype mismatch that's easy to miss."
        ),
        "buggy_code": (
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "def run_forward(model, data):\n"
            "    \"\"\"Run a forward pass with the model.\"\"\"\n"
            "    model = model.double()  # Bug: changes model to float64\n"
            "    # data is still float32 — dtype mismatch will crash!\n"
            "    output = model(data)\n"
            "    return output\n"
        ),
        "error_hint": (
            "When you call model.double(), all model parameters become float64. "
            "But if input data is float32, PyTorch raises: "
            "'RuntimeError: expected scalar type Double but found Float'. "
            "Fix: cast input to match — data = data.double() — or keep both float32."
        ),
        "error_type": "dtype-mismatch",
        "line_hint": 6,
        "test": (
            "import torch\n"
            "import torch.nn as nn\n"
            "torch.manual_seed(0)\n"
            "model = nn.Linear(3, 2)\n"
            "data = torch.randn(4, 3)  # float32\n"
            "try:\n"
            "    out = run_forward(model, data)\n"
            "    assert out.shape == (4, 2), f'Expected shape (4,2) got {out.shape}'\n"
            "    assert out.dtype == torch.float64, f'Expected float64 output got {out.dtype}'\n"
            "    print('PASS')\n"
            "except RuntimeError as e:\n"
            "    raise AssertionError(f'dtype mismatch not fixed: {e}')\n"
        ),
    },
    {
        "level": "expert",
        "title": "detach() Blocks Gradient Flow to Weight Network",
        "description": (
            "A learnable loss weighting network uses .detach() on its output before combining losses. "
            "This stops gradients from flowing back to the weight network, "
            "so it never learns — silently producing wrong results with no error."
        ),
        "buggy_code": (
            "import torch\n"
            "\n"
            "def compute_weighted_loss(loss_a, loss_b, weight):\n"
            "    \"\"\"Combine two losses with a learnable scalar weight.\"\"\"\n"
            "    w = weight.detach()  # Bug: detach() kills gradient flow!\n"
            "    return w * loss_a + (1 - w) * loss_b\n"
        ),
        "error_hint": (
            ".detach() creates a new tensor that shares data but is excluded from the computation graph. "
            "Any gradients computed through the output will NOT flow back to `weight`. "
            "Remove .detach() so the weight network receives gradient updates."
        ),
        "error_type": "detach-kills-gradient",
        "line_hint": 5,
        "test": (
            "import torch\n"
            "weight = torch.tensor(0.6, requires_grad=True)\n"
            "loss_a = torch.tensor(2.0)\n"
            "loss_b = torch.tensor(0.5)\n"
            "combined = compute_weighted_loss(loss_a, loss_b, weight)\n"
            "combined.backward()\n"
            "assert weight.grad is not None, 'Gradient did not reach weight — detach() bug!'\n"
            "# d/dw (w*2.0 + (1-w)*0.5) = 2.0 - 0.5 = 1.5\n"
            "assert abs(weight.grad.item() - 1.5) < 1e-4, f'Expected grad=1.5 got {weight.grad.item()}'\n"
            "print('PASS')\n"
        ),
    },
    {
        "level": "master",
        "title": "Wrong Batch Normalization Axis (dim=1 instead of dim=0)",
        "description": (
            "A custom batch normalization normalizes across features per sample (dim=1) "
            "instead of across batch samples per feature (dim=0). "
            "This is instance normalization, not batch normalization — "
            "a subtle but critical mistake that produces wrong statistics."
        ),
        "buggy_code": (
            "import torch\n"
            "\n"
            "def batch_normalize(batch):\n"
            "    \"\"\"Apply batch normalization: normalize each feature across the batch.\"\"\"\n"
            "    mean = batch.mean(dim=1, keepdim=True)  # Bug: dim=1 normalizes per sample\n"
            "    std  = batch.std(dim=1, keepdim=True) + 1e-8\n"
            "    return (batch - mean) / std\n"
        ),
        "error_hint": (
            "Batch Normalization computes mean and std FOR EACH FEATURE across all samples. "
            "That means the reduction axis is dim=0 (across the batch). "
            "dim=1 reduces across features within a sample — that is Instance Normalization, not Batch Norm. "
            "Fix: change dim=1 to dim=0 in both .mean() and .std()."
        ),
        "error_type": "wrong-norm-axis",
        "line_hint": 5,
        "test": (
            "import torch\n"
            "batch = torch.tensor([\n"
            "    [1.0, 1.0],\n"
            "    [1.0, 2.0],\n"
            "    [1.0, 3.0],\n"
            "    [1.0, 4.0],\n"
            "])\n"
            "result = batch_normalize(batch)\n"
            "# Feature 0 is constant across batch -> batch norm -> all zeros\n"
            "assert result[:, 0].abs().max().item() < 0.01, (\n"
            "    f'Feature 0 should be ~0 after batch norm, got {result[:, 0]}'\n"
            ")\n"
            "# Feature 1 batch mean should be ~0\n"
            "feat1_mean = result[:, 1].mean().item()\n"
            "assert abs(feat1_mean) < 1e-4, f'Feature 1 batch mean should be ~0, got {feat1_mean}'\n"
            "# Feature 1 batch std should be ~1\n"
            "feat1_std = result[:, 1].std().item()\n"
            "assert abs(feat1_std - 1.0) < 0.1, f'Feature 1 std should be ~1, got {feat1_std}'\n"
            "print('PASS')\n"
        ),
    },
]


state = {
    "index": 0,
    "step_count": 0,
    "episode_id": "ep-1",
    "inspected": False,
    "identified": False,
    "explained": False,
    "last_patch": None,
}



class Action(BaseModel):
    action_type: str  # inspect_code | identify_bug | explain_bug | suggest_fix | run_tests
    line_number: Optional[int] = None
    explanation: Optional[str] = None
    code_patch: Optional[str] = None



def clamp(r: float) -> float:
    """Reward strictly between 0 and 1 — never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, r)), 2)


def evaluate_fix(code_patch: str, task: dict):
    """Run fix against test cases. Returns (reward, feedback). Uses real torch."""
    if not code_patch or not code_patch.strip():
        return 0.01, "No fix provided."
    full_code = code_patch + "\n" + task["test"]
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            fname = f.name
        result = subprocess.run(
            ["python3", fname],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(fname)
        if "PASS" in result.stdout:
            return 0.99, "All tests passed! Perfect fix 🎉"
        elif result.returncode == 0:
            return 0.50, "Code runs but tests failed — logic still wrong."
        else:
            if "SyntaxError" in result.stderr:
                return 0.05, f"Syntax error: {result.stderr[:150]}"
            if "AssertionError" in result.stderr:
                return 0.30, f"Fix runs but assertion failed: {result.stderr[:150]}"
            if "RuntimeError" in result.stderr:
                return 0.15, f"PyTorch RuntimeError: {result.stderr[:150]}"
            if "TypeError" in result.stderr:
                return 0.10, f"Type error: {result.stderr[:150]}"
            return 0.20, f"Error: {result.stderr[:150]}"
    except subprocess.TimeoutExpired:
        return 0.01, "Fix timed out — possible infinite loop."
    except Exception as e:
        return 0.01, f"Grader error: {str(e)}"


def current_obs():
    t = TASKS[state["index"]]
    return {
        "buggy_code": t["buggy_code"],
        "task_level": t["level"],
        "task_title": t["title"],
        "task_description": t["description"],
        "error_hint": t["error_hint"],
        "error_type": t["error_type"],
        "step_count": state["step_count"],
        "inspected": state["inspected"],
        "identified": state["identified"],
        "explained": state["explained"],
    }



@app.get("/")
def home():
    return {
        "message": "🔬 PyTorch Code Review OpenEnv — Neurobytes",
        "version": "3.0.0",
        "theme": "Real PyTorch / ML training bugs",
        "total_tasks": len(TASKS),
        "task_levels": [t["level"] for t in TASKS],
        "task_titles": [t["title"] for t in TASKS],
        "action_space": [
            "inspect_code",
            "identify_bug",
            "explain_bug",
            "suggest_fix",
            "run_tests",
        ],
        "reward_range": "(0, 1) exclusive",
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET  /state",
        },
    }


@app.post("/reset")
def reset():
    state.update({
        "index": 0, "step_count": 0, "episode_id": "ep-1",
        "inspected": False, "identified": False,
        "explained": False, "last_patch": None,
    })
    return {
        "observation": current_obs(),
        "reward": 0.01,
        "done": False,
        "feedback": (
            "New episode! You are a senior PyTorch engineer reviewing buggy ML training code. "
            "Recommended workflow: inspect_code → identify_bug → explain_bug → suggest_fix → run_tests."
        ),
    }


@app.post("/step")
def step(action: Action):
    t = TASKS[state["index"]]
    state["step_count"] += 1
    reward = 0.01
    feedback = ""
    done = False

    if action.action_type == "inspect_code":
        if not state["inspected"]:
            state["inspected"] = True
            reward = 0.15
            feedback = (
                f"Code inspected. Task: '{t['title']}'. "
                f"Error type: {t['error_type']}. "
                f"{t['description']}"
            )
        else:
            reward = 0.05
            feedback = "Already inspected. Use identify_bug next."

    elif action.action_type == "identify_bug":
        if action.line_number == t["line_hint"]:
            state["identified"] = True
            reward = 0.40
            feedback = (
                f"✅ Correct! The bug is on line {action.line_number}. "
                f"Now explain WHY using explain_bug."
            )
        elif action.line_number is not None:
            reward = 0.10
            feedback = f"❌ Line {action.line_number} is not the main bug. Look more carefully."
        else:
            reward = 0.05
            feedback = "Provide line_number to pinpoint the bug."

    elif action.action_type == "explain_bug":
        if action.explanation and len(action.explanation.strip()) > 20:
            state["explained"] = True
            bonus = 0.10 if state["identified"] else 0.0
            reward = clamp(0.30 + bonus)
            feedback = "Good explanation! Now submit your fix using suggest_fix."
        else:
            reward = 0.05
            feedback = "Explanation too short. Describe WHY this is a bug in detail."

    elif action.action_type == "suggest_fix":
        raw_reward, feedback = evaluate_fix(action.code_patch, t)
        workflow_bonus = sum([
            0.03 if state["inspected"] else 0.0,
            0.03 if state["identified"] else 0.0,
            0.03 if state["explained"] else 0.0,
        ])
        reward = clamp(raw_reward + workflow_bonus)
        state["last_patch"] = action.code_patch

        if raw_reward >= 0.90:
            if state["index"] < len(TASKS) - 1:
                state["index"] += 1
                state.update({
                    "inspected": False, "identified": False,
                    "explained": False, "last_patch": None,
                })
                next_t = TASKS[state["index"]]
                feedback += f" ➡️ Next: [{next_t['level']}] {next_t['title']}"
            else:
                done = True
                feedback += " 🏆 All 5 PyTorch tasks complete! Outstanding work!"

    elif action.action_type == "run_tests":
        if state["last_patch"]:
            raw_reward, test_feedback = evaluate_fix(state["last_patch"], t)
            reward = clamp(raw_reward * 0.5)
            feedback = f"Test result: {test_feedback}"
        else:
            reward = 0.05
            feedback = "No patch submitted yet. Use suggest_fix first."

    else:
        reward = 0.01
        feedback = (
            "Unknown action. Valid: "
            "inspect_code | identify_bug | explain_bug | suggest_fix | run_tests"
        )

    return {
        "observation": current_obs(),
        "reward": clamp(reward),
        "done": done,
        "feedback": feedback,
    }


@app.get("/state")
def get_state():
    return {
        "episode_id": state["episode_id"],
        "step_count": state["step_count"],
        "current_task_index": state["index"],
        "current_task": TASKS[state["index"]]["title"],
        "current_level": TASKS[state["index"]]["level"],
        "total_tasks": len(TASKS),
        "workflow": {
            "inspected": state["inspected"],
            "identified": state["identified"],
            "explained": state["explained"],
        },
    }



def run_inference(num_episodes: int = 1, steps_per_episode: int = 6):
    """
    Baseline inference using judges' LLM proxy.
    Agent reviews 5 real PyTorch bugs following full workflow:
    inspect → identify → explain → fix → run_tests
    """
    global state

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
    )
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    SYSTEM_PROMPT = (
        "You are a senior ML engineer and PyTorch expert performing code review. "
        "You will be given buggy PyTorch/ML code with real torch tensors and operations. "
        "When asked to explain: write 2-3 clear technical sentences about WHY the bug is wrong "
        "and what goes wrong at runtime. "
        "When asked to fix: return ONLY the complete corrected Python code. "
        "No markdown, no backticks, no explanation — just raw Python with import statements."
    )

    for episode in range(num_episodes):
        for task_idx in range(len(TASKS)):
            state.update({
                "index": task_idx,
                "step_count": 0,
                "episode_id": f"ep-{episode+1}",
                "inspected": False,
                "identified": False,
                "explained": False,
                "last_patch": None,
            })
            task = TASKS[task_idx]
            task_name = f"{task['level']}-task-{task_idx}"

            print(f"[START] task={task_name}", flush=True)

            total_reward = 0.0
            steps_taken = 0

            workflow = [
                {"action_type": "inspect_code"},
                {"action_type": "identify_bug", "line_number": task["line_hint"]},
                {"action_type": "explain_bug",  "use_llm": True, "mode": "explain"},
                {"action_type": "suggest_fix",  "use_llm": True, "mode": "fix"},
                {"action_type": "run_tests"},
                {"action_type": "suggest_fix",  "use_llm": True, "mode": "fix"},
            ]
            for step_num, w in enumerate(workflow, 1):
                if step_num > steps_per_episode:
                    break

                steps_taken += 1
                action_payload = {"action_type": w["action_type"]}

                if w.get("line_number"):
                    action_payload["line_number"] = w["line_number"]

                if w.get("use_llm"):
                    try:
                        if w["mode"] == "explain":
                            user_msg = (
                                f"Buggy PyTorch code:\n\n{task['buggy_code']}\n\n"
                                f"Bug hint: {task['error_hint']}\n\n"
                                f"In 2-3 technical sentences, explain exactly why this is a bug "
                                f"and what goes wrong at runtime in PyTorch."
                            )
                        else:
                            user_msg = (
                                f"Buggy PyTorch code:\n\n{task['buggy_code']}\n\n"
                                f"Bug hint: {task['error_hint']}\n\n"
                                f"Return ONLY the complete corrected Python code including all imports. "
                                f"No markdown, no backticks, no explanation."
                            )

                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": user_msg},
                            ],
                            timeout=60,
                            max_tokens=500,
                        )
                        content = response.choices[0].message.content.strip()
                        if content.startswith("```"):
                            content = "\n".join(content.split("\n")[1:-1])

                        if w["mode"] == "explain":
                            action_payload["explanation"] = content
                        else:
                            action_payload["code_patch"] = content

                    except Exception as e:
                        print(f"LLM call failed: {e}", flush=True)
                        if w["mode"] == "explain":
                            action_payload["explanation"] = task["error_hint"]
                        else:
                            action_payload["code_patch"] = task["buggy_code"]

                result = step(Action(**action_payload))
                reward = result["reward"]
                total_reward += reward

                print(f"[STEP] step={step_num} reward={reward:.2f}", flush=True)

                if result["done"] or (
                    w["action_type"] == "suggest_fix" and reward >= 0.90
                ):
                    break

            final_score = clamp(total_reward / max(steps_taken, 1))
            print(
                f"[END] task={task_name} score={final_score:.2f} steps={steps_taken}",
                flush=True,
            )



if __name__ == "__main__":

    def _hard_timeout(signum, frame):
        print("[END] task=timeout score=0.01 steps=0", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGALRM, _hard_timeout)
    signal.alarm(18 * 60)  # 18 min hard limit

    run_inference(num_episodes=1, steps_per_episode=6)
