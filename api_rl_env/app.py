import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from env.api_rate_env import APIRateLimitEnv
from agents.baselines import RandomAgent, GreedyPriorityAgent, RuleBasedAgent

def run_episode(agent_name, num_apis, num_tasks, seed):
    env = APIRateLimitEnv(
        num_apis    = int(num_apis),
        num_tasks   = int(num_tasks),
        time_budget = 100,
        cost_budget = 200.0,
        seed        = int(seed),
    )

    agents = {
        "Random"        : RandomAgent,
        "GreedyPriority": GreedyPriorityAgent,
        "RuleBased"     : RuleBasedAgent,
    }

    obs, _  = env.reset(seed=int(seed))
    agent   = agents[agent_name](env)
    done    = False
    total_r = 0.0
    log     = []

    while not done:
        action              = agent.predict(obs)
        obs, r, term, trunc, info = env.step(action)
        total_r            += r
        done                = term or trunc

    log.append(f"✅ Tasks Completed  : {info['tasks_done']}")
    log.append(f"❌ Tasks Failed     : {info['tasks_failed']}")
    log.append(f"💰 Cost Spent       : {info['cost_spent']:.2f}")
    log.append(f"🏆 Total Reward     : {total_r:.2f}")
    log.append(f"⚡ Cache Hits       : {info['cache_hits']}")
    log.append(f"📦 Batches Used     : {info['batches']}")
    log.append(f"⏰ Deadline Misses  : {info['deadline_misses']}")

    return "\n".join(log)


demo = gr.Interface(
    fn      = run_episode,
    inputs  = [
        gr.Dropdown(["Random", "GreedyPriority", "RuleBased"], label="Agent", value="RuleBased"),
        gr.Slider(2, 10, value=5, step=1, label="Number of APIs"),
        gr.Slider(3, 15, value=10, step=1, label="Number of Tasks"),
        gr.Slider(0, 100, value=42, step=1, label="Random Seed"),
    ],
    outputs = gr.Textbox(label="Episode Results", lines=10),
    title   = "🌐 API Rate Limit Optimizer — RL Environment",
    description = (
        "Simulate an intelligent API orchestration agent managing rate limits, "
        "costs, retries, caching and task dependencies.\n\n"
        "Choose an agent, configure the environment, and run an episode!"
    ),
    examples=[
        ["RuleBased", 5, 10, 42],
        ["GreedyPriority", 7, 12, 7],
        ["Random", 3, 6, 0],
    ]
)

demo.launch()