"""Deterministic evaluation of a trained policy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..envs import BenchEnv, Obs

__all__ = ["evaluate_policy"]


def evaluate_policy(
    agent,
    env: BenchEnv,
    episodes: int = 5,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``episodes`` deterministic episodes and summarise the returns.

    The return of an episode is the **mean per-agent reward**, so the numbers are
    comparable between the single-robot and the multi-robot environments.
    """
    was_training = getattr(agent, "_train_mode", True)
    if hasattr(agent, "eval"):
        agent.eval()

    obs: Obs = env.reset()
    returns = []
    running = np.zeros(env.num_envs, dtype=np.float64)
    budget = int(max_steps or (episodes * env.max_episode_steps + 10))
    for _ in range(budget):
        action = agent.act(obs.obs, obs.state, deterministic=True)
        step = env.step(action)
        running += step.reward.mean(axis=1)
        for i in np.nonzero(step.done)[0]:
            returns.append(float(running[i]))
            running[i] = 0.0
        obs = Obs(step.obs, step.state)
        if len(returns) >= episodes:
            break

    if was_training and hasattr(agent, "train"):
        agent.train()
    if not returns:
        return {"eval_return": float("nan"), "eval_return_std": float("nan"), "episodes": 0}
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "episodes": len(returns),
    }
