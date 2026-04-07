"""
API Simulator
=============
Simulates multiple third-party API services with:
  - Rate-limit quotas that reset on windows
  - Stochastic failures drawn from Beta distribution
  - Latency spikes (log-normal)
  - Dynamic pricing fluctuations
  - Cascading cooldown on repeated failures
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class APIService:
    name:             str
    quota_max:        int
    quota_remaining:  int
    window_size:      int    # timesteps between quota resets
    window_reset_in:  int    # timesteps until next reset
    base_reliability: float  # P(success) baseline
    reliability:      float  # current rolling estimate
    base_latency:     float  # mean latency in "units"
    latency_est:      float  # current rolling latency estimate
    base_cost:        float  # cost per call
    cost_per_call:    float  # current (possibly dynamic) cost
    fail_streak:      int    = 0
    in_cooldown:      bool   = False
    cooldown_timer:   int    = 0
    last_api_used:    Optional[int] = None


# Default API profiles — realistic diversity
API_PROFILES = [
    # name,           quota, window, reliability, latency, cost
    ("SearchAPI",        60,    10,    0.97,        0.3,   0.5),
    ("LLM_API",           5,    10,    0.92,        2.0,   5.0),
    ("DatabaseAPI",      30,    10,    0.99,        0.2,   0.2),
    ("WeatherAPI",      120,    10,    0.85,        0.5,   0.1),
    ("PaymentAPI",        3,    10,    0.995,       1.0,   8.0),
    ("EmailAPI",         20,    10,    0.90,        0.8,   1.0),
    ("StorageAPI",       50,    10,    0.96,        0.4,   0.3),
    ("AnalyticsAPI",     15,    10,    0.88,        1.5,   2.0),
    ("NotificationAPI",  40,    10,    0.91,        0.6,   0.7),
    ("TranslationAPI",   25,    10,    0.93,        1.2,   1.5),
]


class APISimulator:
    """
    Manages K API services with stochastic dynamics.

    On each `step()` call:
      - Quotas may reset if their window expires
      - Reliability drifts slightly (slow random walk)
      - Latency can spike (log-normal)
      - Dynamic pricing fluctuates ±10%
      - Cooldowns tick down
    """

    def __init__(self, num_apis: int = 5, seed: Optional[int] = None):
        self.num_apis = min(num_apis, len(API_PROFILES))
        self.rng      = np.random.default_rng(seed)
        self.apis: list[APIService] = []
        self._build_apis()

    def reset(self):
        self.apis = []
        self._build_apis()

    def _build_apis(self):
        for i in range(self.num_apis):
            name, quota, window, rel, lat, cost = API_PROFILES[i]
            # Add jitter so each episode is slightly different
            quota_r = max(1, int(quota * self.rng.uniform(0.8, 1.2)))
            rel_r   = float(np.clip(rel + self.rng.normal(0, 0.02), 0.5, 0.999))
            lat_r   = float(max(0.05, lat * self.rng.uniform(0.8, 1.2)))
            cost_r  = float(max(0.1, cost * self.rng.uniform(0.9, 1.1)))

            self.apis.append(APIService(
                name             = name,
                quota_max        = quota_r,
                quota_remaining  = quota_r,
                window_size      = window,
                window_reset_in  = window,
                base_reliability = rel_r,
                reliability      = rel_r,
                base_latency     = lat_r,
                latency_est      = lat_r,
                base_cost        = cost_r,
                cost_per_call    = cost_r,
            ))

    def call(self, api_idx: int) -> Tuple[bool, float, float]:
        """
        Execute one API call.

        Returns
        -------
        success  : bool
        latency  : float (simulated response time)
        cost     : float (charge for this call)
        """
        api = self.apis[api_idx]

        if api.quota_remaining <= 0 or api.in_cooldown:
            return False, 0.0, 0.0

        # Decrement quota
        api.quota_remaining -= 1

        # Stochastic outcome
        success  = self.rng.random() < api.reliability
        latency  = self._sample_latency(api)
        cost     = api.cost_per_call

        # Update fail streak & cooldown
        if success:
            api.fail_streak = 0
        else:
            api.fail_streak += 1
            if api.fail_streak >= 3:
                api.in_cooldown   = True
                api.cooldown_timer = int(self.rng.integers(3, 8))
                api.fail_streak   = 0

        # Update rolling reliability estimate (EMA)
        api.reliability = 0.9 * api.reliability + 0.1 * float(success)

        # Update rolling latency estimate
        api.latency_est = 0.85 * api.latency_est + 0.15 * latency

        return success, latency, cost

    def step(self):
        """Advance one timestep: reset windows, drift params, tick cooldowns."""
        for api in self.apis:
            # Quota window reset
            api.window_reset_in -= 1
            if api.window_reset_in <= 0:
                api.quota_remaining = api.quota_max
                api.window_reset_in = api.window_size
                # Slight quota jitter on reset
                jitter = int(self.rng.integers(-2, 3))
                api.quota_remaining = max(1, api.quota_remaining + jitter)

            # Cooldown tick
            if api.in_cooldown:
                api.cooldown_timer -= 1
                if api.cooldown_timer <= 0:
                    api.in_cooldown = False

            # Reliability slow random walk (mean-reverting)
            drift = self.rng.normal(0, 0.005)
            api.reliability = float(np.clip(
                api.reliability + drift * (api.base_reliability - api.reliability),
                0.5, 0.999
            ))

            # Dynamic pricing fluctuation ±10%
            price_shock    = self.rng.normal(1.0, 0.05)
            api.cost_per_call = float(
                np.clip(api.base_cost * price_shock, api.base_cost * 0.5, api.base_cost * 2.0)
            )

    # ── Private ──────────────────────────────────────────────────────────────

    def _sample_latency(self, api: APIService) -> float:
        """
        Latency is log-normally distributed with occasional spikes.
        ~5% chance of a 3–10× spike (simulates overloaded upstream).
        """
        base  = self.rng.lognormal(mean=np.log(api.base_latency), sigma=0.3)
        spike = self.rng.random() < 0.05
        if spike:
            base *= self.rng.uniform(3, 10)
        return float(base)

    def status_table(self) -> str:
        """Pretty-print current API status."""
        lines = [f"{'API':<16} {'Quota':>6} {'Reset':>6} {'Rel':>6} {'Lat':>6} {'Cost':>6} {'Cool':>5}"]
        lines.append("-" * 55)
        for api in self.apis:
            cool = f"{api.cooldown_timer}t" if api.in_cooldown else "  —"
            lines.append(
                f"{api.name:<16} {api.quota_remaining:>3}/{api.quota_max:<2} "
                f"{api.window_reset_in:>5}t "
                f"{api.reliability:>6.3f} "
                f"{api.latency_est:>6.2f} "
                f"{api.cost_per_call:>6.2f} "
                f"{cool:>5}"
            )
        return "\n".join(lines)
