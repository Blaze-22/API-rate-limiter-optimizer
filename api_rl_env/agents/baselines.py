"""
Baseline Agents
===============
Simple rule-based baselines to compare against the RL agent.

Available baselines:
  RandomAgent        — picks a random valid action
  GreedyPriorityAgent — always serves highest-priority ready task on best API
  RuleBasedAgent     — hand-crafted heuristics with rate-limit awareness
"""

import numpy as np
from env.task_generator import TaskStatus


class RandomAgent:
    """Uniformly samples from valid (unmasked) actions."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        mask        = self.env.action_masks()
        valid       = np.where(mask)[0]
        if len(valid) == 0:
            return 0    # fallback: WAIT
        return int(np.random.choice(valid))


class GreedyPriorityAgent:
    """
    At each step:
      1. Find highest-priority PENDING task whose deps are met
      2. Pick the cheapest API with available quota
      3. If task result is cached → use cache
      4. Otherwise → CALL
    """

    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        env   = self.env
        tasks = env._tasks
        apis  = env.api_sim.apis

        # Sort tasks by priority desc
        candidates = [
            (i, t) for i, t in enumerate(tasks)
            if t.status == TaskStatus.PENDING and t.dependencies_met
        ]
        if not candidates:
            # Nothing to do: encode WAIT
            return env._encode_action(0, 0, 0, 0)

        candidates.sort(key=lambda x: -x[1].priority)
        task_i, task = candidates[0]

        # Check cache first
        if env.cache_mgr.has(task_i):
            return env._encode_action(4, task_i, 0, 0)   # ACTION_CACHE

        # Find cheapest available API
        available_apis = [
            (k, a) for k, a in enumerate(apis)
            if a.quota_remaining > 0 and not a.in_cooldown
        ]
        if not available_apis:
            return env._encode_action(0, 0, 0, 0)   # WAIT

        # Prefer preferred_apis, then cheapest
        preferred = [(k, a) for k, a in available_apis if k in task.preferred_apis]
        pool      = preferred if preferred else available_apis
        api_k     = min(pool, key=lambda x: x[1].cost_per_call)[0]

        return env._encode_action(1, task_i, 0, api_k)   # ACTION_CALL


class RuleBasedAgent:
    """
    More sophisticated heuristic:
      - Retry failed tasks if fail_count < 2
      - Batch two same-priority tasks when possible
      - Abandon very low-priority tasks near deadline pressure
      - Prefer high-reliability APIs for high-priority tasks
      - Fall back to cache when rate-limited
    """

    RETRY_THRESHOLD   = 2
    ABANDON_PRIORITY  = 2.0    # abandon tasks with priority ≤ this
    ABANDON_TIME_FRAC = 0.85   # only abandon when >85% of time is used

    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        env       = self.env
        tasks     = env._tasks
        apis      = env.api_sim.apis
        time_frac = env._timestep / env.time_budget

        available_apis = [
            (k, a) for k, a in enumerate(apis)
            if a.quota_remaining > 0 and not a.in_cooldown
        ]

        # ── 1. Retry recently failed tasks ───────────────────────────────
        for i, t in enumerate(tasks):
            if t.status == TaskStatus.FAILED and t.fail_count < self.RETRY_THRESHOLD:
                if t.last_api_used is not None:
                    api = apis[t.last_api_used]
                    if api.quota_remaining > 0 and not api.in_cooldown:
                        return env._encode_action(2, i, 0, 0)   # RETRY

        # ── 2. Abandon low-priority hopeless tasks late in episode ───────
        if time_frac > self.ABANDON_TIME_FRAC:
            for i, t in enumerate(tasks):
                if t.status == TaskStatus.PENDING and t.priority <= self.ABANDON_PRIORITY:
                    return env._encode_action(5, i, 0, 0)   # ABANDON

        # ── 3. Serve highest-priority ready tasks ────────────────────────
        ready = [
            (i, t) for i, t in enumerate(tasks)
            if t.status == TaskStatus.PENDING and t.dependencies_met
        ]
        ready.sort(key=lambda x: (-x[1].priority, x[1].deadline))

        if not ready:
            return env._encode_action(0, 0, 0, 0)   # WAIT

        task_i, task = ready[0]

        # ── 4. Cache hit ─────────────────────────────────────────────────
        if env.cache_mgr.has(task_i):
            return env._encode_action(4, task_i, 0, 0)

        if not available_apis:
            return env._encode_action(0, 0, 0, 0)   # WAIT for reset

        # ── 5. Try to batch if two tasks share the same preferred API ────
        if len(ready) >= 2:
            task_j, t2 = ready[1]
            shared = set(task.preferred_apis) & set(t2.preferred_apis)
            for ak in shared:
                a = apis[ak]
                if a.quota_remaining > 0 and not a.in_cooldown:
                    return env._encode_action(6, task_i, task_j, ak)   # BATCH

        # ── 6. Pick best API for task ────────────────────────────────────
        # High priority → pick highest reliability; else → cheapest
        preferred = [(k, a) for k, a in available_apis if k in task.preferred_apis]
        pool      = preferred if preferred else available_apis

        if task.priority >= 7:
            api_k = max(pool, key=lambda x: x[1].reliability)[0]
        else:
            api_k = min(pool, key=lambda x: x[1].cost_per_call)[0]

        return env._encode_action(1, task_i, 0, api_k)   # CALL
