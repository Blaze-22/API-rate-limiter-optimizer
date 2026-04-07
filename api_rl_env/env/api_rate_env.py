"""
API Rate Limit Optimizer - Core Gymnasium Environment
=====================================================
An RL environment where an agent learns to orchestrate API calls
across multiple services with rate limits, costs, and stochastic failures.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from env.api_simulator import APISimulator
from env.task_generator import TaskGenerator, Task, TaskStatus
from env.cache_manager import CacheManager


# ── Action constants ────────────────────────────────────────────────────────
ACTION_WAIT     = 0
ACTION_CALL     = 1   # (task_idx, api_idx)
ACTION_RETRY    = 2   # (task_idx,)
ACTION_REROUTE  = 3   # (task_idx, api_idx)
ACTION_CACHE    = 4   # (task_idx,)
ACTION_ABANDON  = 5   # (task_idx,)
ACTION_BATCH    = 6   # (task_idx, task_jdx, api_idx)


class APIRateLimitEnv(gym.Env):
    """
    Gymnasium environment: API Rate Limit Optimizer

    Observation space : flat vector of API states + task queue states
    Action space      : Discrete over (action_type, task_i, task_j, api_k)
                        encoded as a single integer; invalid actions are masked.

    Parameters
    ----------
    num_apis       : Number of simulated API services
    num_tasks      : Max tasks in a queue episode
    time_budget    : Max timesteps per episode
    cost_budget    : Max spend per episode (arbitrary units)
    seed           : Random seed for reproducibility
    render_mode    : 'human' prints a summary each step
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_apis: int = 5,
        num_tasks: int = 10,
        time_budget: int = 100,
        cost_budget: float = 200.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_apis      = num_apis
        self.num_tasks     = num_tasks
        self.time_budget   = time_budget
        self.cost_budget   = cost_budget
        self.render_mode   = render_mode

        # Sub-modules
        self.api_sim    = APISimulator(num_apis=num_apis, seed=seed)
        self.task_gen   = TaskGenerator(num_tasks=num_tasks, num_apis=num_apis, seed=seed)
        self.cache_mgr  = CacheManager(capacity=num_tasks * 2, ttl=20)

        # ── Action space ────────────────────────────────────────────────────
        # Encode as flat int:
        #   action_type (7) × task_i (num_tasks) × task_j (num_tasks) × api (num_apis)
        self._action_dims = (7, num_tasks, num_tasks, num_apis)
        total_actions = int(np.prod(self._action_dims))
        self.action_space = spaces.Discrete(total_actions)

        # ── Observation space ───────────────────────────────────────────────
        # Per-API: quota_remaining, window_reset, latency_est, reliability, cost_per_call  → 5*K
        # Per-task: priority, deadline_remaining, deps_met, status, cache_available        → 5*M
        # Global  : cost_spent_norm, time_remaining_norm                                   → 2
        obs_size = 5 * num_apis + 5 * num_tasks + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Internal state
        self._tasks:    list[Task] = []
        self._timestep: int        = 0
        self._cost_spent: float    = 0.0
        self._inflight: dict       = {}   # task_idx → api_idx
        self._episode_stats        = {}

    # ────────────────────────────────────────────────────────────────────────
    # Core Gymnasium API
    # ────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.api_sim    = APISimulator(num_apis=self.num_apis, seed=seed)
            self.task_gen   = TaskGenerator(num_tasks=self.num_tasks, num_apis=self.num_apis, seed=seed)

        self.api_sim.reset()
        self.cache_mgr.reset()

        self._tasks      = self.task_gen.generate()
        self._timestep   = 0
        self._cost_spent = 0.0
        self._inflight   = {}
        self._episode_stats = {
            "completed": 0, "failed": 0, "abandoned": 0,
            "cache_hits": 0, "batches": 0, "deadline_misses": 0,
        }

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        atype, task_i, task_j, api_k = self._decode_action(action)
        reward   = 0.0
        info     = {}

        # ── Advance time; process in-flight completions ──────────────────
        self._timestep += 1
        self._tick_inflight()
        self._check_deadlines()

        # ── Execute chosen action ────────────────────────────────────────
        if atype == ACTION_WAIT:
            reward += self._act_wait()

        elif atype == ACTION_CALL:
            reward += self._act_call(task_i, api_k)

        elif atype == ACTION_RETRY:
            reward += self._act_retry(task_i)

        elif atype == ACTION_REROUTE:
            reward += self._act_reroute(task_i, api_k)

        elif atype == ACTION_CACHE:
            reward += self._act_cache(task_i)

        elif atype == ACTION_ABANDON:
            reward += self._act_abandon(task_i)

        elif atype == ACTION_BATCH:
            reward += self._act_batch(task_i, task_j, api_k)

        # ── Penalty for budget overrun ───────────────────────────────────
        if self._cost_spent > self.cost_budget:
            reward -= 30.0

        # ── Step the API simulator (quota resets, flakiness updates) ─────
        self.api_sim.step()
        self.cache_mgr.tick()

        obs        = self._get_obs()
        terminated = self._is_done()
        truncated  = self._timestep >= self.time_budget
        info       = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (for MaskablePPO)."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        for atype in range(7):
            for ti in range(self.num_tasks):
                for tj in range(self.num_tasks):
                    for ak in range(self.num_apis):
                        idx = self._encode_action(atype, ti, tj, ak)
                        mask[idx] = self._is_valid(atype, ti, tj, ak)
        return mask

    def render(self):
        print(self._render_str())

    def _render_str(self) -> str:
        lines = [f"\n{'═'*55}"]
        lines.append(f"  Step {self._timestep:>3} | Cost {self._cost_spent:>6.1f}/{self.cost_budget}")
        lines.append(f"{'─'*55}")
        lines.append("  APIs:")
        for i, api in enumerate(self.api_sim.apis):
            bar = "█" * int(api.quota_remaining / api.quota_max * 10)
            bar = bar.ljust(10)
            lines.append(
                f"    [{i}] {api.name:<12} quota [{bar}] "
                f"{api.quota_remaining:>3}/{api.quota_max:<3} "
                f"rel={api.reliability:.2f} cost={api.cost_per_call:.1f}"
            )
        lines.append("  Tasks:")
        status_sym = {
            TaskStatus.PENDING:   "⬜",
            TaskStatus.INFLIGHT:  "🔄",
            TaskStatus.DONE:      "✅",
            TaskStatus.FAILED:    "❌",
            TaskStatus.ABANDONED: "🚫",
        }
        for i, t in enumerate(self._tasks):
            sym = status_sym.get(t.status, "?")
            lines.append(
                f"    [{i}] {sym} P={t.priority} DL={t.deadline - self._timestep:>3}t "
                f"deps={'✓' if t.dependencies_met else '✗'}"
            )
        lines.append(f"{'═'*55}")
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────────
    # Action Handlers
    # ────────────────────────────────────────────────────────────────────────

    def _act_wait(self) -> float:
        return -1.0   # small idle penalty encourages the agent to act

    def _act_call(self, task_i: int, api_k: int) -> float:
        task = self._tasks[task_i]
        api  = self.api_sim.apis[api_k]

        if not self._can_call(task, api):
            return -5.0   # invalid move penalty

        success, latency, cost = self.api_sim.call(api_k)
        self._cost_spent += cost

        if success:
            task.status = TaskStatus.DONE
            self._update_dependents(task_i)
            self._episode_stats["completed"] += 1
            bonus = max(0, task.deadline - self._timestep) * 0.5
            return task.priority * 10 + bonus - cost * 0.3
        else:
            task.status = TaskStatus.FAILED
            task.fail_count += 1
            self._episode_stats["failed"] += 1
            return -8.0

    def _act_retry(self, task_i: int) -> float:
        task = self._tasks[task_i]
        if task.status != TaskStatus.FAILED or task.fail_count >= 3:
            return -5.0

        last_api = task.last_api_used
        if last_api is None:
            return -5.0

        task.status = TaskStatus.PENDING
        return self._act_call(task_i, last_api)

    def _act_reroute(self, task_i: int, api_k: int) -> float:
        task = self._tasks[task_i]
        if task.status not in (TaskStatus.FAILED, TaskStatus.PENDING):
            return -5.0
        task.status = TaskStatus.PENDING
        return self._act_call(task_i, api_k)

    def _act_cache(self, task_i: int) -> float:
        task = self._tasks[task_i]
        if not self.cache_mgr.has(task_i):
            return -5.0
        task.status = TaskStatus.DONE
        self._update_dependents(task_i)
        self._episode_stats["cache_hits"] += 1
        self._episode_stats["completed"] += 1
        return task.priority * 8   # slightly less than real call (staleness)

    def _act_abandon(self, task_i: int) -> float:
        task = self._tasks[task_i]
        if task.status in (TaskStatus.DONE, TaskStatus.ABANDONED):
            return -5.0
        task.status = TaskStatus.ABANDONED
        self._episode_stats["abandoned"] += 1
        return -task.priority * 2   # proportional penalty

    def _act_batch(self, task_i: int, task_j: int, api_k: int) -> float:
        if task_i == task_j:
            return -5.0
        t1, t2 = self._tasks[task_i], self._tasks[task_j]
        api     = self.api_sim.apis[api_k]

        if not (self._can_call(t1, api) and self._can_call(t2, api)):
            return -5.0

        # Batch counts as one quota unit
        success, _, cost = self.api_sim.call(api_k)
        self._cost_spent += cost   # only one call cost!

        if success:
            for tidx, task in [(task_i, t1), (task_j, t2)]:
                task.status = TaskStatus.DONE
                self._update_dependents(tidx)
                self._episode_stats["completed"] += 1
            self._episode_stats["batches"] += 1
            reward = (t1.priority + t2.priority) * 10 - cost * 0.3 + 15
            return reward
        else:
            for task in (t1, t2):
                task.status = TaskStatus.FAILED
                task.fail_count += 1
            return -12.0

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────

    def _can_call(self, task, api) -> bool:
        return (
            task.status == TaskStatus.PENDING
            and task.dependencies_met
            and api.quota_remaining > 0
            and not api.in_cooldown
        )

    def _tick_inflight(self):
        """Resolve any async in-flight calls (placeholder for latency modeling)."""
        pass

    def _check_deadlines(self):
        for task in self._tasks:
            if task.status == TaskStatus.PENDING and self._timestep > task.deadline:
                task.status = TaskStatus.FAILED
                self._episode_stats["deadline_misses"] += 1

    def _update_dependents(self, completed_idx: int):
        """Mark dependent tasks as ready if all their deps are now done."""
        done_ids = {i for i, t in enumerate(self._tasks) if t.status == TaskStatus.DONE}
        for task in self._tasks:
            if task.status == TaskStatus.PENDING:
                task.dependencies_met = all(d in done_ids for d in task.depends_on)

    def _is_done(self) -> bool:
        all_resolved = all(
            t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABANDONED)
            for t in self._tasks
        )
        return all_resolved or self._cost_spent >= self.cost_budget * 1.5

    def _get_obs(self) -> np.ndarray:
        parts = []

        # API features (normalised)
        for api in self.api_sim.apis:
            parts += [
                api.quota_remaining / max(api.quota_max, 1),
                api.window_reset_in / 20.0,
                min(api.latency_est / 5.0, 1.0),
                api.reliability,
                min(api.cost_per_call / 10.0, 1.0),
            ]

        # Task features (normalised)
        done_ids = {i for i, t in enumerate(self._tasks) if t.status == TaskStatus.DONE}
        for i, task in enumerate(self._tasks):
            status_norm = {
                TaskStatus.PENDING:   0.2,
                TaskStatus.INFLIGHT:  0.4,
                TaskStatus.DONE:      1.0,
                TaskStatus.FAILED:    0.0,
                TaskStatus.ABANDONED: 0.1,
            }.get(task.status, 0.0)

            deadline_remaining = max(task.deadline - self._timestep, 0)
            parts += [
                task.priority / 10.0,
                min(deadline_remaining / self.time_budget, 1.0),
                float(task.dependencies_met),
                status_norm,
                float(self.cache_mgr.has(i)),
            ]

        # Global
        parts += [
            min(self._cost_spent / self.cost_budget, 1.5),
            1.0 - self._timestep / self.time_budget,
        ]

        return np.clip(np.array(parts, dtype=np.float32), 0.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "timestep":        self._timestep,
            "cost_spent":      self._cost_spent,
            "cost_remaining":  self.cost_budget - self._cost_spent,
            "tasks_done":      sum(1 for t in self._tasks if t.status == TaskStatus.DONE),
            "tasks_pending":   sum(1 for t in self._tasks if t.status == TaskStatus.PENDING),
            "tasks_failed":    sum(1 for t in self._tasks if t.status == TaskStatus.FAILED),
            **self._episode_stats,
        }

    def _encode_action(self, atype, ti, tj, ak) -> int:
        d = self._action_dims
        return int(atype * d[1]*d[2]*d[3] + ti * d[2]*d[3] + tj * d[3] + ak)

    def _decode_action(self, action: int):
        d  = self._action_dims
        ak = action % d[3];             action //= d[3]
        tj = action % d[2];             action //= d[2]
        ti = action % d[1];             action //= d[1]
        return action, ti, tj, ak

    def _is_valid(self, atype, ti, tj, ak) -> bool:
        if ti >= len(self._tasks):
            return False
        task_i = self._tasks[ti]
        api    = self.api_sim.apis[ak]

        if atype == ACTION_WAIT:
            return ti == 0 and tj == 0 and ak == 0   # only one WAIT action

        if atype == ACTION_CALL:
            return self._can_call(task_i, api)

        if atype == ACTION_RETRY:
            return (task_i.status == TaskStatus.FAILED
                    and task_i.fail_count < 3
                    and task_i.last_api_used is not None)

        if atype == ACTION_REROUTE:
            return (task_i.status in (TaskStatus.FAILED, TaskStatus.PENDING)
                    and api.quota_remaining > 0)

        if atype == ACTION_CACHE:
            return (task_i.status == TaskStatus.PENDING
                    and self.cache_mgr.has(ti))

        if atype == ACTION_ABANDON:
            return task_i.status not in (TaskStatus.DONE, TaskStatus.ABANDONED)

        if atype == ACTION_BATCH:
            if ti == tj or tj >= len(self._tasks):
                return False
            task_j = self._tasks[tj]
            return (self._can_call(task_i, api)
                    and self._can_call(task_j, api))

        return False
