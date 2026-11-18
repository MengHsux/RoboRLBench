"""Every baseline must build, update and checkpoint on its reference task.

``point-reach`` is the single-robot reference task and ``multi-hunt-2d`` the
multi-robot one; both are pure NumPy so this test needs no simulator at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from roborlbench.algos import build_agent, get_algo_spec, list_algos
from roborlbench.envs import make_env
from roborlbench.runners import TrainConfig, Trainer
from roborlbench.runners.evaluate import evaluate_policy

ALGOS = list_algos()
STEPS = 192
NUM_ENVS = 2

#: metrics that exist whether or not a gradient step happened
BOOKKEEPING = {"train_return", "steps_per_s", "eval_return"}


def reference_env(name: str) -> str:
    return "point-reach" if get_algo_spec(name).family == "single" else "multi-hunt-2d"


def small_overrides(name: str) -> dict:
    if get_algo_spec(name).kind == "off_policy":
        return {"buffer_size": 8192, "warmup": 128, "batch_size": 32, "grad_steps": 1}
    return {"rollout_steps": 8, "num_epochs": 1, "num_minibatches": 2}


def make_config(name: str, log_dir, run_name: str) -> TrainConfig:
    return TrainConfig(
        env=reference_env(name),
        algo=name,
        num_envs=NUM_ENVS,
        total_steps=STEPS,
        seed=0,
        device="cpu",
        eval_interval=STEPS // 2,
        eval_episodes=2,
        save_interval=0,
        log_dir=str(log_dir),
        run_name=run_name,
        algo_kwargs=small_overrides(name),
    )


@pytest.mark.parametrize("name", ALGOS)
def test_baseline_trains(name, tmp_path):
    trainer = Trainer(make_config(name, tmp_path, f"fit-{name}"))
    summary = trainer.train()

    assert trainer.history, "the training loop produced no iterations"
    assert any(set(row) - BOOKKEEPING for row in trainer.history), (
        f"{name} never performed a gradient update"
    )
    for row in trainer.history:
        for key, value in row.items():
            assert np.isfinite(value), f"{name}: {key} is not finite"

    assert summary["steps"] >= STEPS
    assert summary["final_eval_return"] is not None
    assert np.isfinite(summary["final_eval_return"])


@pytest.mark.parametrize("name", ["sac", "ppo", "mappo", "hasac"])
def test_checkpoint_roundtrip(name, tmp_path):
    """A reloaded agent must reproduce the saved policy exactly."""
    trainer = Trainer(make_config(name, tmp_path, f"ckpt-{name}"))
    trainer.train()
    path = tmp_path / f"{name}.pt"
    trainer.agent.save(str(path))

    env = trainer.env
    reloaded = build_agent(
        name,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_agents=env.num_agents,
        state_dim=env.state_dim,
        num_envs=env.num_envs,
        device="cpu",
        seed=0,
        **small_overrides(name),
    )
    reloaded.load(str(path))
    reloaded.eval()

    obs = env.reset()
    first = trainer.agent.act(obs.obs, obs.state, deterministic=True)
    second = reloaded.act(obs.obs, obs.state, deterministic=True)
    np.testing.assert_allclose(first, second, atol=1e-5)


@pytest.mark.parametrize("name", ALGOS)
def test_evaluation_runs(name, tmp_path):
    trainer = Trainer(make_config(name, tmp_path, f"eval-{name}"))
    env = trainer.env
    agent = build_agent(
        name,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_agents=env.num_agents,
        state_dim=env.state_dim,
        num_envs=env.num_envs,
        seed=0,
        **small_overrides(name),
    )
    result = evaluate_policy(agent, env, episodes=2)
    assert result["episodes"] >= 1
    assert np.isfinite(result["eval_return"])


def test_multi_agent_baselines_work_on_a_single_robot_env():
    """Multi-agent baselines accept ``num_agents == 1`` (needed for the Go2 tasks)."""
    env = make_env("point-reach", num_envs=1, seed=0)
    agent = build_agent(
        "mappo",
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_agents=env.num_agents,
        state_dim=env.state_dim,
        num_envs=env.num_envs,
        rollout_steps=4,
        num_minibatches=1,
        num_epochs=1,
    )
    obs = env.reset()
    buf = agent.make_buffer()
    for _ in range(4):
        action, logp, value = agent.act_batch(
            _t(obs.obs), _t(obs.state)
        )
        step = env.step(action.numpy().astype(np.float32))
        buf.add(obs.obs, obs.state, action.numpy(), logp, value,
                step.reward, step.done, step.timeout)
        obs = step
    buf.compute_returns(agent.value_of(_t(obs.state), _t(obs.obs)), None)
    metrics = agent.update(buf)
    assert metrics


def _t(x):
    import torch

    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
