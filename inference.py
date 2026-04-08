import os
from openai import OpenAI
from env.api_rate_env import APIRateLimitEnv
from agents.baselines import RuleBasedAgent
import numpy as np

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run():
    print("START")

    env   = APIRateLimitEnv(num_apis=5, num_tasks=10, time_budget=100, cost_budget=200.0, seed=42)
    obs, _= env.reset(seed=42)
    agent = RuleBasedAgent(env)
    done  = False
    total_reward = 0.0
    step  = 0

    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1
        print(f"STEP {step} | reward={reward:.2f} | tasks_done={info['tasks_done']} | cost={info['cost_spent']:.2f}")

    print(f"END | total_reward={total_reward:.2f} | tasks_done={info['tasks_done']} | tasks_failed={info['tasks_failed']}")

if __name__ == "__main__":
    run()
