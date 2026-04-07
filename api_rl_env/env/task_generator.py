"""
Task Generator
==============
Generates a queue of tasks with:
  - Random priorities (1–10)
  - Deadlines distributed across the episode
  - Dependency graphs (DAGs) — some tasks require others first
  - Required API type hints (some tasks need specific APIs)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TaskStatus(Enum):
    PENDING   = auto()
    INFLIGHT  = auto()
    DONE      = auto()
    FAILED    = auto()
    ABANDONED = auto()


@dataclass
class Task:
    id:               int
    priority:         float          # 1.0–10.0
    deadline:         int            # absolute timestep deadline
    depends_on:       list[int]      # list of task IDs that must be DONE first
    preferred_apis:   list[int]      # API indices that suit this task type
    status:           TaskStatus     = TaskStatus.PENDING
    dependencies_met: bool           = True   # True if no deps or all deps done
    fail_count:       int            = 0
    last_api_used:    Optional[int]  = None

    def __post_init__(self):
        # Tasks with no dependencies start as ready
        self.dependencies_met = len(self.depends_on) == 0


class TaskGenerator:
    """
    Generates M tasks with a realistic dependency DAG.

    Dependency structure:
      - ~40% of tasks have no dependencies   (leaf tasks — always ready)
      - ~40% depend on exactly one prior task
      - ~20% depend on two prior tasks       (fan-in)

    This ensures the agent must sequence tasks smartly.
    """

    def __init__(
        self,
        num_tasks:   int = 10,
        num_apis:    int = 5,
        time_budget: int = 100,
        seed:        Optional[int] = None,
    ):
        self.num_tasks   = num_tasks
        self.num_apis    = num_apis
        self.time_budget = time_budget
        self.rng         = np.random.default_rng(seed)

    def generate(self) -> list[Task]:
        tasks = []

        for i in range(self.num_tasks):
            priority = float(self.rng.integers(1, 11))

            # Deadline: tasks have varying urgency
            # High priority → tighter deadlines (more pressure)
            slack    = int(self.rng.integers(10, self.time_budget))
            deadline = min(self.time_budget, slack)

            # Dependency generation (only depend on earlier tasks → DAG)
            dep_roll = self.rng.random()
            if i == 0 or dep_roll < 0.40:
                depends_on = []
            elif dep_roll < 0.80:
                parent = int(self.rng.integers(0, i))
                depends_on = [parent]
            else:
                parents    = self.rng.choice(i, size=min(2, i), replace=False).tolist()
                depends_on = [int(p) for p in parents]

            # Preferred APIs (2–3 compatible services per task type)
            num_preferred = int(self.rng.integers(1, min(4, self.num_apis + 1)))
            preferred_apis = self.rng.choice(
                self.num_apis, size=num_preferred, replace=False
            ).tolist()

            tasks.append(Task(
                id            = i,
                priority      = priority,
                deadline      = deadline,
                depends_on    = depends_on,
                preferred_apis= preferred_apis,
            ))

        return tasks

    def describe(self, tasks: list[Task]) -> str:
        lines = ["Task Queue:"]
        lines.append(f"  {'ID':>2}  {'P':>3}  {'DL':>4}  {'Deps':<15}  {'APIs':<12}  Status")
        lines.append("  " + "-" * 55)
        for t in tasks:
            deps_str = str(t.depends_on) if t.depends_on else "none"
            apis_str = str(t.preferred_apis)
            lines.append(
                f"  {t.id:>2}  {t.priority:>3.0f}  {t.deadline:>4}  "
                f"{deps_str:<15}  {apis_str:<12}  {t.status.name}"
            )
        return "\n".join(lines)
