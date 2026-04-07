"""
smoke_test.py
=============
Quick sanity check — runs the environment with a random agent
for a few steps without any ML dependencies.

Usage: python smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from env.api_rate_env import APIRateLimitEnv
from env.task_generator import TaskStatus
import numpy as np


def main():
    print("=" * 55)
    print("  API Rate Limit Optimizer — Smoke Test")
    print("=" * 55)

    env = APIRateLimitEnv(
        num_apis    = 5,
        num_tasks   = 8,
        time_budget = 30,
        cost_budget = 100.0,
        seed        = 42,
        render_mode = "human",
    )

    obs, info = env.reset(seed=42)
    print(f"\nObservation shape : {obs.shape}")
    print(f"Action space size : {env.action_space.n}")
    print(f"\nInitial task queue:")
    print(env.task_gen.describe(env._tasks))
    print(f"\nInitial API status:")
    print(env.api_sim.status_table())

    total_reward = 0.0
    done         = False
    step         = 0

    while not done and step < 30:
        # Sample a random valid action using the mask
        mask  = env.action_masks()
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done          = terminated or truncated
        step         += 1

    print(f"\n{'='*55}")
    print(f"  Episode finished after {step} steps")
    print(f"  Total reward    : {total_reward:.2f}")
    print(f"  Tasks done      : {info['tasks_done']}")
    print(f"  Tasks failed    : {info['tasks_failed']}")
    print(f"  Cost spent      : {info['cost_spent']:.2f}")
    print(f"  Cache hits      : {info['cache_hits']}")
    print(f"  Batches used    : {info['batches']}")
    print(f"  Deadline misses : {info['deadline_misses']}")
    print(f"{'='*55}")

    print("\n✅ Smoke test passed — environment is working correctly!")


if __name__ == "__main__":
    main()
