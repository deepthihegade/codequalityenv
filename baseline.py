from server.environment import Environment
from models import Action


def run_baseline():
    env = Environment()

    # Start episode
    result = env.reset()
    print(f"\n🚀 Episode Started!")
    print(f"Task: {result.observation.task_level}")
    print(f"Error type: {result.observation.error_hint}")
    print(f"Buggy code:\n{result.observation.buggy_code}")
    print(f"Reward: {result.reward}")

    # Task 1 - Easy: identify then fix syntax bug
    print("\n--- TASK 1: EASY (Syntax) ---")
    result = env.step(Action(action_type="identify_bug", line_number=1))
    print(f"Identify bug → Reward: {result.reward} | {result.feedback}")

    result = env.step(Action(
        action_type="suggest_fix",
        code_patch="""def greet(name):
    print("Hello, " + name)"""
    ))
    print(f"Suggest fix → Reward: {result.reward} | {result.feedback}")

    # Task 2 - Medium: fix logic bug
    print("\n--- TASK 2: MEDIUM (Logic) ---")
    result = env.step(Action(
        action_type="suggest_fix",
        code_patch="""def find_max(nums):
    max_val = nums[0]
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val"""
    ))
    print(f"Suggest fix → Reward: {result.reward} | {result.feedback}")

    # Task 3 - Hard: optimize performance
    print("\n--- TASK 3: HARD (Performance) ---")
    result = env.step(Action(
        action_type="suggest_fix",
        code_patch="""def has_duplicate(nums):
    return len(nums) != len(set(nums))"""
    ))
    print(f"Suggest fix → Reward: {result.reward} | {result.feedback}")
    print(f"Done: {result.done}")

    # Print final state
    state = env.state
    print(f"\n📊 Final State:")
    print(f"Episode ID: {state.episode_id}")
    print(f"Total steps: {state.step_count}")
    print(f"Final task: {state.current_task}")


if __name__ == "__main__":
    run_baseline()