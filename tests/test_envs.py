"""The BenchEnv contract: shapes, dtypes, auto-reset and truncation."""

from __future__ import annotations

import numpy as np
import pytest

from roborlbench.envs import ENVS, Obs, make_env, probe_env

ANALYTIC = sorted(n for n, s in ENVS.items() if s.backend == "analytic")


@pytest.mark.parametrize("name", ANALYTIC)
@pytest.mark.parametrize("num_envs", [1, 4])
def test_probe(name, num_envs):
    env = make_env(name, num_envs=num_envs, seed=1)
    info = probe_env(env, steps=5, seed=0)
    assert info["num_envs"] == num_envs
    assert info["obs_dim"] == env.obs_dim
    assert info["state_dim"] == env.state_dim
    assert np.isfinite(info["mean_reward_per_step"])


@pytest.mark.parametrize("name", ANALYTIC)
def test_reset_is_deterministic_given_the_seed(name):
    a = make_env(name, num_envs=2, seed=7).reset()
    b = make_env(name, num_envs=2, seed=7).reset()
    np.testing.assert_allclose(a.obs, b.obs)
    np.testing.assert_allclose(a.state, b.state)


@pytest.mark.parametrize("name", ANALYTIC)
def test_auto_reset_and_timeout(name):
    env = make_env(name, num_envs=2, seed=3)
    env.reset()
    zeros = np.zeros((env.num_envs, env.num_agents, env.act_dim), dtype=np.float32)
    saw_done = False
    for _ in range(env.max_episode_steps + 5):
        step = env.step(zeros)
        assert step.done.shape == (env.num_envs,)
        if step.done.any():
            saw_done = True
            assert step.info.get("timeout") is not None
            # the returned observation is the *post-reset* one
            assert np.isfinite(step.obs).all()
            break
    assert saw_done, f"{name} never finished an episode"


def test_wrong_action_shape_is_rejected():
    env = make_env("point-reach", num_envs=2, seed=0)
    env.reset()
    with pytest.raises(ValueError, match="expected actions of shape"):
        env.step(np.zeros((2, 1), dtype=np.float32))


def test_unknown_environment_lists_the_alternatives():
    with pytest.raises(KeyError, match="unknown environment"):
        make_env("does-not-exist")


def test_obs_helper_squeezes_single_agent():
    obs = Obs(obs=np.zeros((3, 1, 4), np.float32), state=np.zeros((3, 9), np.float32))
    flat_obs, state = obs.squeeze_agents()
    assert flat_obs.shape == (3, 4) and state.shape == (3, 9)
    with pytest.raises(ValueError):
        Obs(obs=np.zeros((3, 2, 4), np.float32), state=np.zeros((3, 9), np.float32)).squeeze_agents()
