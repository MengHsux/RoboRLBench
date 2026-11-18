"""Learning sanity: a baseline must actually *learn*, not just run.

These are slow (a minute or two on a laptop CPU), so they are marked ``slow``:

    pytest -m "not slow"      # skip them
    pytest -m slow            # only them
"""

from __future__ import annotations

import numpy as np
import pytest

from roborlbench.envs import make_env
from roborlbench.runners import TrainConfig, Trainer
from roborlbench.runners.evaluate import evaluate_policy

pytestmark = pytest.mark.slow


def random_policy_return(env, episodes: int = 10, seed: int = 0) -> float:
    """Mean per-agent return of a uniform random policy, for reference."""
    rng = np.random.default_rng(seed)
    env.reset()
    running = np.zeros(env.num_envs, dtype=np.float64)
    returns = []
    for _ in range(episodes * env.max_episode_steps + 10):
        action = rng.uniform(
            -1.0, 1.0, (env.num_envs, env.num_agents, env.act_dim)
        ).astype(np.float32)
        step = env.step(action)
        running += step.reward.mean(axis=1)
        for i in np.nonzero(step.done)[0]:
            returns.append(float(running[i]))
            running[i] = 0.0
        if len(returns) >= episodes:
            break
    return float(np.mean(returns))


def _train(env_name: str, algo: str, steps: int, num_envs: int, **algo_kwargs) -> float:
    config = TrainConfig(
        env=env_name,
        algo=algo,
        num_envs=num_envs,
        total_steps=steps,
        seed=0,
        device="cpu",
        eval_interval=steps,
        eval_episodes=10,
        save_interval=0,
        log_dir="runs/_test_learning",
        run_name=f"{algo}-{env_name}",
        algo_kwargs=algo_kwargs,
    )
    return float(Trainer(config).train()["final_eval_return"])


def test_sac_learns_point_reach():
    """SAC must get much closer to the goal than a random policy."""
    baseline = random_policy_return(make_env("point-reach", num_envs=1, seed=0))
    trained = _train("point-reach", "sac", 20_000, 4, warmup=2_000)
    assert baseline < -100.0, f"unexpected random baseline {baseline}"
    assert trained > baseline + 80.0, (
        f"SAC did not learn: random={baseline:.1f} trained={trained:.1f}"
    )


def test_mappo_learns_multi_hunt():
    """MAPPO must improve the team return on the 3-agent encirclement task."""
    baseline = random_policy_return(make_env("multi-hunt-2d", num_envs=1, seed=0))
    trained = _train("multi-hunt-2d", "mappo", 20_000, 4, rollout_steps=24)
    assert trained > baseline + 200.0, (
        f"MAPPO did not learn: random={baseline:.1f} trained={trained:.1f}"
    )


def test_evaluate_policy_completes_the_requested_episodes():
    env = make_env("point-reach", num_envs=2, seed=0)
    from roborlbench.algos import build_agent

    agent = build_agent("sac", obs_dim=env.obs_dim, act_dim=env.act_dim,
                        num_agents=env.num_agents, state_dim=env.state_dim,
                        num_envs=env.num_envs, seed=0)
    result = evaluate_policy(agent, env, episodes=4)
    assert result["episodes"] >= 4
    assert np.isfinite(result["eval_return"])
