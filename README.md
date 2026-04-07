# API-Rate-Limit-Optimizer-Environment
An RL environment where an agent learns to orchestrate API calls across multiple stochastic services with rate limits, costs, failures, and task dependencies.

📁 Project Structure
api_rl_env/
├── env/
│   ├── api_rate_env.py     ← Core Gymnasium environment
│   ├── api_simulator.py    ← Stochastic API service models
│   ├── task_generator.py   ← Task queue with dependency DAGs
│   └── cache_manager.py    ← LRU cache with TTL expiry
├── agents/
│   └── baselines.py        ← Random / Greedy / Rule-based agents
├── train.py                ← Train + evaluate + baseline comparison
├── smoke_test.py           ← Quick sanity check (no ML needed)
├── requirements.txt
└── README.md

🚀 Quick Start
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke test (no ML needed)
python smoke_test.py

# 3. Compare baselines
python train.py --baselines

# 4. Train RL agent (recommended: with action masking)
python train.py --train --masked --timesteps 300000

# 5. Evaluate trained agent
python train.py --evaluate --masked

# 6. Watch it play (render mode)
python train.py --evaluate --masked --render

🧭 Core Concept
A reinforcement learning agent acts as an intelligent API orchestration layer that must complete a queue of tasks by dispatching calls across several third-party APIs — each with different rate limits, costs, latencies, and failure probabilities. The agent learns to schedule, prioritize, retry, cache, and reroute requests optimally under real-world stochastic constraints.

🕹️ Action Space
ActionDescriptionWAITDo nothing this timestep (small idle penalty)CALLDispatch task i to API kRETRYRetry a recently failed task on the same APIREROUTESwitch a failed/slow task to a different APICACHE_USEServe task from cache — no quota consumedABANDONDrop a low-priority task to free budgetBATCHBundle two tasks into a single API call
Actions are masked when invalid (e.g. calling a rate-limited or cooled-down API), preventing the agent from wasting exploration on impossible moves.

📦 Observation Space
A flat normalised float vector [0, 1] containing:
Per API (×K): quota remaining, window reset countdown, rolling latency estimate, rolling reliability estimate, current cost per call
Per Task (×M): priority score, deadline remaining, dependencies met flag, current status encoding, cache availability flag
Global (×2): cost spent (normalised), time remaining (normalised)

🏆 Reward Function
EventRewardTask completed on timepriority × 10 + deadline_bonusCache hitpriority × 8Successful batchAbove + +15 bonusWAIT (idle)-1Invalid action-5API call failure-8Deadline missed-50Rate limit violation-20Budget overrun-30

⚙️ Environment Parameters
ParameterDefaultDescriptionnum_apis5Number of simulated API servicesnum_tasks10Tasks per episodetime_budget100Max timesteps per episodecost_budget200.0Max spend per episodeseedNoneReproducibility seed

🌩️ Stochastic Dynamics — What Makes It Hard
DynamicMechanismQuota window resetsQuota refills every N steps with ±jitterAPI reliability driftMean-reverting random walk each stepLatency spikesLog-normal base + 5% chance of 3–10× spikeDynamic pricingCost fluctuates ±10% per stepCascading cooldown3 consecutive failures → API locked for 3–8 stepsTask deadlinesMissed deadlines auto-fail pending tasks

🤖 API Profiles
APIQuota/windowReliabilityLatencyCostSearchAPI600.97low0.5LLM_API50.92high5.0DatabaseAPI300.99very low0.2WeatherAPI1200.85medium0.1PaymentAPI30.995medium8.0

📊 Baselines to Beat
AgentStrategyRandomAgentUniformly samples from valid actionsGreedyPriorityAgentHighest-priority task on cheapest available APIRuleBasedAgentRetry → batch → reroute heuristic with rate-limit awarenessRL Agent (PPO)Fully learned policy — should outperform all above under load

🔬 Research Extensions
ExtensionDescriptionMulti-agentMultiple orchestrators sharing the same API quotas — learn not to collideCurriculum learningStart 2 APIs / 5 tasks → gradually scale to 10 / 50LLM-as-judge rewardUse Claude API to score workflow quality as a dense reward signalAdversarial APIOne API randomly spikes pricing mid-episode to force reroutingLive dashboardMatplotlib animation showing quota bars, task queue, and agent decisions in real timeInterpretabilityVisualise the agent's learned when-to-wait-vs-retry policy as a heatmap

📦 Dependencies
gymnasium>=0.29.0
numpy>=1.24.0
stable-baselines3>=2.2.0
sb3-contrib>=2.2.0
matplotlib>=3.7.0