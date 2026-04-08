from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from env.api_rate_env import APIRateLimitEnv
from agents.baselines import RuleBasedAgent

app = FastAPI()

env   = None
agent = None

class ResetRequest(BaseModel):
    seed: int = 42

class StepRequest(BaseModel):
    action: int

@app.post("/reset")
def reset(req: ResetRequest):
    global env, agent
    env    = APIRateLimitEnv(num_apis=5, num_tasks=10, time_budget=100, cost_budget=200.0, seed=req.seed)
    obs, _ = env.reset(seed=req.seed)
    agent  = RuleBasedAgent(env)
    return {"observation": obs.tolist(), "info": env._get_info()}

@app.post("/step")
def step(req: StepRequest):
    global env, agent
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

@app.get("/validate")
def validate():
    return {"status": "ok"}