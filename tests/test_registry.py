"""The registries are the contract the rest of the project relies on."""

from __future__ import annotations

import pytest

from roborlbench.algos import ALGOS, AgentBase, build_agent, list_algos
from roborlbench.envs import ENVS, list_envs


def test_twenty_baselines_ten_per_family():
    assert len(ALGOS) == 20
    assert len(list_algos("single")) == 10
    assert len(list_algos("multi")) == 10
    # the user's own method must never be part of the benchmark
    assert "tsser" not in {name.lower() for name in ALGOS}


@pytest.mark.parametrize("name", sorted(ALGOS))
def test_spec_is_complete(name):
    spec = ALGOS[name]
    assert issubclass(spec.cls, AgentBase)
    assert spec.family in ("single", "multi")
    assert spec.kind in ("on_policy", "off_policy")
    assert spec.venue and spec.year >= 2017 and spec.description
    assert spec.multi_agent == (spec.family == "multi")


@pytest.mark.parametrize("name", sorted(ALGOS))
def test_build_agent_from_spec(name):
    spec = ALGOS[name]
    agent = build_agent(
        name,
        obs_dim=8,
        act_dim=2,
        num_agents=1 if spec.family == "single" else 3,
        state_dim=12,
        num_envs=2,
        seed=0,
        **_cheap_overrides(spec),
    )
    assert isinstance(agent, spec.cls)
    assert agent.multi_agent == spec.multi_agent
    # modules() and state_dict() must work before any training step
    assert agent.modules()
    assert agent.state_dict()["algo"] == name


def _cheap_overrides(spec) -> dict:
    """Shrink the buffers so constructing an agent stays instantaneous."""
    if spec.kind == "off_policy":
        return {"buffer_size": 4096, "batch_size": 32, "warmup": 64}
    return {"rollout_steps": 8, "num_minibatches": 2}


def test_unknown_hyper_parameter_is_rejected():
    with pytest.raises(TypeError, match="unknown hyper-parameter"):
        build_agent("sac", obs_dim=8, act_dim=2, not_a_real_option=1)


def test_single_agent_baseline_rejects_multi_agent_env():
    with pytest.raises(ValueError, match="single-agent baseline"):
        build_agent("sac", obs_dim=8, act_dim=2, num_agents=3)


def test_derived_baselines_accept_inherited_options():
    # ``batch_size`` is declared by OffPolicyAgent, not by TQC itself
    agent = build_agent("tqc", obs_dim=8, act_dim=2, batch_size=64, buffer_size=4096)
    assert agent.batch_size == 64


def test_environment_registry():
    assert len(ENVS) == 12
    assert set(list_envs("single")) | set(list_envs("multi")) == set(ENVS)
    for name, spec in ENVS.items():
        assert spec.entry_point.count(":") == 1, name
        assert spec.backend in ("analytic", "mujoco", "isaacgym", "isaaclab")
        if spec.kind == "single":
            assert spec.num_agents == 1
