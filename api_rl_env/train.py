"""
train.py
========
Train the API Rate Limit Optimizer using PPO (or MaskablePPO).

Usage
-----
  python train.py                         # default settings
  python train.py --timesteps 500000      # longer training
  python train.py --masked                # use MaskablePPO (recommended)
  python train.py --render                # watch last eval episode
"""

import argparse
import os
import numpy as np

from env.api_rate_env import APIRateLimitEnv


def make_env(seed=0, render_mode=None):
    env = APIRateLimitEnv(
        num_apis    = 5,
        num_tasks   = 10,
        time_budget = 100,
        cost_budget = 200.0,
        seed        = seed,
        render_mode = render_mode,
    )
    return env


def train(args):
    try:
        if args.masked:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            from stable_baselines3.common.env_util import make_vec_env

            def _make():
                env = make_env()
                return ActionMasker(env, lambda e: e.action_masks())

            vec_env = make_vec_env(_make, n_envs=args.n_envs)
            model   = MaskablePPO(
                "MlpPolicy",
                vec_env,
                verbose           = 1,
                learning_rate     = 3e-4,
                n_steps           = 512,
                batch_size        = 64,
                n_epochs          = 10,
                gamma             = 0.99,
                gae_lambda        = 0.95,
                clip_range        = 0.2,
                tensorboard_log   = "./logs/",
            )
            print("✅ Using MaskablePPO (invalid actions are masked)")
        else:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env

            vec_env = make_vec_env(make_env, n_envs=args.n_envs)
            model   = PPO(
                "MlpPolicy",
                vec_env,
                verbose         = 1,
                learning_rate   = 3e-4,
                n_steps         = 512,
                batch_size      = 64,
                n_epochs        = 10,
                gamma           = 0.99,
                gae_lambda      = 0.95,
                clip_range      = 0.2,
                tensorboard_log = "./logs/",
            )
            print("✅ Using standard PPO")

    except ImportError as e:
        print(f"\n⚠️  Missing package: {e}")
        print("Install with: pip install stable-baselines3 sb3-contrib")
        return

    print(f"\n🚀 Training for {args.timesteps:,} timesteps …\n")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    os.makedirs("./models", exist_ok=True)
    model.save("./models/api_rate_ppo")
    print("\n💾 Model saved to ./models/api_rate_ppo")


def evaluate(args):
    try:
        if args.masked:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            env   = make_env(render_mode="human" if args.render else None)
            env   = ActionMasker(env, lambda e: e.action_masks())
            model = MaskablePPO.load("./models/api_rate_ppo", env=env)
        else:
            from stable_baselines3 import PPO
            env   = make_env(render_mode="human" if args.render else None)
            model = PPO.load("./models/api_rate_ppo", env=env)
    except FileNotFoundError:
        print("❌ No saved model found. Run with --train first.")
        return
    except ImportError as e:
        print(f"⚠️  Missing package: {e}")
        return

    print(f"\n📊 Evaluating over {args.eval_episodes} episodes …\n")

    ep_rewards, ep_done, ep_costs = [], [], []

    for ep in range(args.eval_episodes):
        obs, _    = env.env.reset() if hasattr(env, "env") else env.reset()
        done      = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = (
                env.step(action) if hasattr(env, "step") else env.env.step(action)
            )
            ep_reward += reward
            done       = terminated or truncated

        ep_rewards.append(ep_reward)
        ep_done.append(info.get("tasks_done", 0))
        ep_costs.append(info.get("cost_spent", 0))

    print(f"  Mean reward      : {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"  Mean tasks done  : {np.mean(ep_done):.1f}")
    print(f"  Mean cost spent  : {np.mean(ep_costs):.1f}")


def run_baselines(args):
    from agents.baselines import RandomAgent, GreedyPriorityAgent, RuleBasedAgent

    agents = {
        "Random"        : RandomAgent,
        "GreedyPriority": GreedyPriorityAgent,
        "RuleBased"     : RuleBasedAgent,
    }

    print(f"\n📊 Baseline evaluation over {args.eval_episodes} episodes:\n")
    print(f"  {'Agent':<20} {'Reward':>10}  {'Tasks Done':>10}  {'Cost':>8}")
    print("  " + "-" * 55)

    for name, AgentClass in agents.items():
        rewards, dones, costs = [], [], []
        for ep in range(args.eval_episodes):
            env    = make_env(seed=ep)
            agent  = AgentClass(env)
            obs, _ = env.reset(seed=ep)
            done   = False
            total  = 0.0
            while not done:
                action          = agent.predict(obs)
                obs, r, term, trunc, info = env.step(action)
                total          += r
                done            = term or trunc
            rewards.append(total)
            dones.append(info.get("tasks_done", 0))
            costs.append(info.get("cost_spent", 0))
        print(
            f"  {name:<20} {np.mean(rewards):>10.2f}  "
            f"{np.mean(dones):>10.1f}  {np.mean(costs):>8.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Rate Limit Optimizer — RL Trainer")
    parser.add_argument("--train",          action="store_true", help="Train a new model")
    parser.add_argument("--evaluate",       action="store_true", help="Evaluate saved model")
    parser.add_argument("--baselines",      action="store_true", help="Run baseline comparison")
    parser.add_argument("--masked",         action="store_true", help="Use MaskablePPO")
    parser.add_argument("--render",         action="store_true", help="Render eval episode")
    parser.add_argument("--timesteps",      type=int, default=200_000)
    parser.add_argument("--n_envs",         type=int, default=4)
    parser.add_argument("--eval_episodes",  type=int, default=20)
    args = parser.parse_args()

    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.baselines:
        run_baselines(args)
    if not any([args.train, args.evaluate, args.baselines]):
        print("Specify --train, --evaluate, or --baselines. Use --help for options.")
