# =============================================================================
# run_benchmark.py
# Created:  [Lc-v0.0.3] 2026-02-12
# Updated:  [Lc-v0.0.3] 2026-02-12 Add human play mode
# =============================================================================

"""
Simplified human play script for Android World benchmarks.

Usage:
    python play_human.py --task=ContactsAddContact
    python play_human.py --task=SimpleCalculatorApp --console_port=5554
"""

import os
import sys
from absl import app
from absl import flags

# Add android_world to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'android_world'))

from android_world import registry
from android_world import suite_utils
from android_world.agents import human_agent
from android_world.env import env_launcher

# =============================================================================
# Flags
# =============================================================================

_TASK = flags.DEFINE_string(
    'task',
    None,
    'Task to play (e.g., ContactsAddContact). Use "all" to play all tasks sequentially, or leave empty for a random task.',
)

_N_TASKS = flags.DEFINE_integer(
    'n_tasks',
    None,
    'Number of tasks to play. Only used when --task is not specified. Default: play one random task.',
)

_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'Android emulator console port (usually 5554 for first emulator)',
)

_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    os.path.expandvars('%LOCALAPPDATA%\\Android\\Sdk\\platform-tools\\adb.exe'),
    'Path to ADB executable',
)

_TASK_SEED = flags.DEFINE_integer(
    'task_seed',
    42,
    'Random seed for task parameters',
)

# =============================================================================
# Main
# =============================================================================

def play_single_task(env, task, task_name):
    """Play a single task and return the reward."""
    print("\n" + "=" * 70)
    print(f"Task: {task_name}")
    print(f"GOAL: {task.goal}")
    print("=" * 70)
    print("\nNow interact with the Android emulator to complete the task.")
    print("When you're done, return here and press Enter to check your solution.")
    print("(Or type 'q' and press Enter to quit)\n")

    # Wait for user
    agent = human_agent.HumanAgent(env)
    result = agent.step(task.goal)

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating your solution...")
    print("=" * 70)

    try:
        reward = task.is_successful(env)
        
        if reward > 0.99:
            print(f"\n✅ SUCCESS! Reward: {reward:.2f}")
        elif reward > 0.5:
            print(f"\n⚠️  PARTIAL SUCCESS. Reward: {reward:.2f}")
        else:
            print(f"\n❌ FAILED. Reward: {reward:.2f}")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        reward = 0.0

    # Clean up
    task.tear_down(env)
    return reward


def main(argv):
    del argv

    print("=" * 70)
    print("Android World - Human Play Mode")
    print("=" * 70)

    # 1. Initialize environment
    print(f"\n[1/4] Connecting to Android emulator (port {_CONSOLE_PORT.value})...")
    env = env_launcher.load_and_setup_env(
        console_port=_CONSOLE_PORT.value,
        emulator_setup=False,  # Don't reset emulator
        adb_path=_ADB_PATH.value,
    )
    print("✓ Connected to emulator")

    # 2. Load tasks
    task_registry = registry.TaskRegistry()
    all_tasks = task_registry.get_registry(
        family=registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )

    # Determine which tasks to run
    if _TASK.value == "all":
        # Run all tasks
        task_names = sorted(all_tasks.keys())
        print(f"\n[2/4] Running ALL {len(task_names)} tasks...")
    elif _TASK.value:
        # Run specific task
        task_names = [_TASK.value]
        print(f"\n[2/4] Loading task: {_TASK.value}...")
    else:
        # Run random task(s)
        import random
        n = _N_TASKS.value or 1
        task_names = random.sample(sorted(all_tasks.keys()), min(n, len(all_tasks)))
        print(f"\n[2/4] Selected {len(task_names)} random task(s)...")

    # Validate tasks
    invalid_tasks = [t for t in task_names if t not in all_tasks]
    if invalid_tasks:
        print(f"\n❌ Error: Unknown task(s): {', '.join(invalid_tasks)}")
        print(f"\nAvailable tasks ({len(all_tasks)}):")
        for i, task_name in enumerate(sorted(all_tasks.keys())[:20], 1):
            print(f"  {i}. {task_name}")
        if len(all_tasks) > 20:
            print(f"  ... and {len(all_tasks) - 20} more")
        env.close()
        sys.exit(1)

    # 3. Run tasks
    rewards = []
    for idx, task_name in enumerate(task_names, 1):
        print(f"\n[{idx}/{len(task_names)}] Preparing task: {task_name}...")
        
        # Create suite for this task
        suite = suite_utils.create_suite(
            all_tasks,
            n_task_combinations=1,
            seed=_TASK_SEED.value,
            tasks=[task_name],
        )
        
        # Suite is a dict[str, list[TaskEval]], get the first instance
        task_instances = suite[task_name]
        task = task_instances[0]
        print(f"✓ Task loaded: {task.complexity}")

        # Initialize task
        print("Setting up task on emulator...")
        task.initialize_task(env)
        print("✓ Task initialized")

        # Play
        reward = play_single_task(env, task, task_name)
        rewards.append((task_name, reward))

    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    for task_name, reward in rewards:
        status = "✅" if reward > 0.99 else "⚠️" if reward > 0.5 else "❌"
        print(f"{status} {task_name}: {reward:.2f}")
    
    avg_reward = sum(r for _, r in rewards) / len(rewards) if rewards else 0
    print(f"\nAverage reward: {avg_reward:.2f}")
    print("=" * 70)

    env.close()


if __name__ == '__main__':
    app.run(main)
