"""Environment registry and factory.

Importing this module registers every benchmark environment.  Simulator
specific adapters are imported lazily, so on a CPU-only machine only the
``analytic`` and (if installed) ``mujoco`` environments can be instantiated,
while ``import roborlbench`` keeps working.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Union

import numpy as np

from .base import (
    ENVS,
    BenchEnv,
    EnvSpec,
    Obs,
    Step,
    get_env_spec,
    list_envs,
    register_env,
)

__all__ = [
    "ENVS",
    "BenchEnv",
    "EnvSpec",
    "Obs",
    "Step",
    "get_env_spec",
    "list_envs",
    "make_env",
    "register_env",
]

# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
# -- analytic (pure NumPy, CPU) --------------------------------------------- #
register_env(
    "point-reach",
    "roborlbench.envs.surrogate:make_point_reach",
    kind="single",
    backend="analytic",
    max_episode_steps=100,
    description="2-D point mass reaching a random goal (dense reward, CPU)",
)

register_env(
    "multi-hunt-2d",
    "roborlbench.envs.surrogate:make_multi_hunt",
    kind="multi",
    backend="analytic",
    num_agents=3,
    max_episode_steps=150,
    description="3-agent 2-D cooperative prey encirclement (CPU)",
)

# -- MuJoCo (CPU, requires the upstream repositories) ----------------------- #
register_env(
    "hunt-multi-mujoco",
    "roborlbench.envs.mujoco_backend:make_hunt_multi",
    kind="multi",
    backend="mujoco",
    task="wolf",
    num_agents=3,
    max_episode_steps=750,
    description="3x Go2 quadruped wolf-hunt in MuJoCo (legged_hunt)",
)

register_env(
    "so101-mujoco",
    "roborlbench.envs.mujoco_backend:make_so101",
    kind="single",
    backend="mujoco",
    task="reach",
    max_episode_steps=200,
    description="SO-101 arm manipulation in MuJoCo (reach/lift/pick/stack)",
)

# -- Isaac Gym (GPU, legged_gym) -------------------------------------------- #
register_env(
    "hunt-multi-isaacgym",
    "roborlbench.envs.isaacgym_backend:make_hunt_multi",
    kind="multi",
    backend="isaacgym",
    task="wolf",
    num_agents=3,
    max_episode_steps=750,
    description="3x Go2 wolf-hunt in Isaac Gym (legged_hunt-main)",
)

register_env(
    "hunt-single-isaacgym",
    "roborlbench.envs.isaacgym_backend:make_hunt_single",
    kind="single",
    backend="isaacgym",
    task="wolf",
    max_episode_steps=750,
    description="Single Go2 hunting in Isaac Gym (legged_hunt-main-single)",
)

register_env(
    "go2-flat-isaacgym",
    "roborlbench.envs.isaacgym_backend:make_go2_flat",
    kind="single",
    backend="isaacgym",
    task="go2",
    max_episode_steps=1000,
    description="Go2 flat-ground velocity tracking in Isaac Gym (legged_gym)",
)

# -- Isaac Lab (GPU) -------------------------------------------------------- #
register_env(
    "go2-flat",
    "roborlbench.envs.isaaclab_backend:make_go2_flat",
    kind="single",
    backend="isaaclab",
    task="go2-flat",
    max_episode_steps=1000,
    description="Go2 flat-ground velocity tracking (Isaac Lab 3.0, go2_rl_lab)",
)

register_env(
    "go2-mixed",
    "roborlbench.envs.isaaclab_backend:make_go2_mixed",
    kind="single",
    backend="isaaclab",
    task="go2-mixed",
    max_episode_steps=1000,
    description="Go2 mixed rough terrain velocity tracking (Isaac Lab 3.0)",
)

register_env(
    "so101-lift",
    "roborlbench.envs.isaaclab_backend:make_so101_lift",
    kind="single",
    backend="isaaclab",
    task="OpenSO101-Lift-v0",
    max_episode_steps=600,
    description="SO-101 lift task (Isaac Lab, OpenSO-101)",
)

register_env(
    "so101-pickplace",
    "roborlbench.envs.isaaclab_backend:make_so101_pickplace",
    kind="single",
    backend="isaaclab",
    task="OpenSO101-PickPlace-v0",
    max_episode_steps=600,
    description="SO-101 pick-and-place task (Isaac Lab, OpenSO-101)",
)

register_env(
    "so101-stack",
    "roborlbench.envs.isaaclab_backend:make_so101_stack",
    kind="single",
    backend="isaaclab",
    task="OpenSO101-Stack-v0",
    max_episode_steps=600,
    description="SO-101 cube stacking task (Isaac Lab, OpenSO-101)",
)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_env(
    name: Union[str, EnvSpec],
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    **kwargs: Any,
) -> BenchEnv:
    """Instantiate a registered environment.

    Parameters
    ----------
    name:
        Registry key (see :func:`list_envs`) or an :class:`EnvSpec`.
    num_envs:
        Number of task copies simulated in parallel.
    seed:
        Base seed; adapters are expected to derive per-copy seeds from it.
    device:
        ``"cpu"`` or ``"cuda"``; ignored by CPU-only backends.
    kwargs:
        Forwarded to the factory and override the defaults in ``EnvSpec.kwargs``.
    """
    spec = name if isinstance(name, EnvSpec) else get_env_spec(name)
    module_name, attr = spec.entry_point.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - simulator dependent
        raise ImportError(
            f"environment {spec.name!r} needs backend {spec.backend!r}, but "
            f"{module_name!r} could not be imported ({exc}).\n"
            f"See docs/ENVIRONMENTS.md for the installation instructions of "
            f"the {spec.backend!r} backend, or use one of the analytic "
            f"environments: {sorted(n for n, s in ENVS.items() if s.backend == 'analytic')}"
        ) from exc

    factory = getattr(module, attr)
    merged: Dict[str, Any] = dict(spec.kwargs)
    merged.update(kwargs)
    env = factory(num_envs=int(num_envs), seed=int(seed), device=device, **merged)
    if not isinstance(env, BenchEnv):
        raise TypeError(f"{spec.entry_point} returned {type(env)!r}, expected a BenchEnv")
    if env.num_agents != spec.num_agents:
        raise RuntimeError(
            f"{spec.name}: adapter reports num_agents={env.num_agents}, "
            f"registry says {spec.num_agents}"
        )
    return env


def probe_env(env: BenchEnv, steps: int = 3, seed: int = 0) -> Dict[str, Any]:
    """Run a few random steps and validate the shape contract.

    Used by ``roborlbench.stats`` / the test-suite; raises ``AssertionError``
    with a precise message if an adapter violates the :class:`BenchEnv` API.
    """
    rng = np.random.default_rng(seed)
    obs = env.reset()
    _check_obs(env, obs)
    total = np.zeros((env.num_envs, env.num_agents), dtype=np.float64)
    for _ in range(steps):
        actions = rng.uniform(-1.0, 1.0, size=(env.num_envs, env.num_agents, env.act_dim))
        step = env.step(actions.astype(np.float32))
        _check_obs(env, Obs(step.obs, step.state))
        assert step.reward.shape == (env.num_envs, env.num_agents), (
            f"reward shape {step.reward.shape} != {(env.num_envs, env.num_agents)}"
        )
        assert step.done.shape == (env.num_envs,), f"done shape {step.done.shape}"
        total += step.reward
    return {
        "obs_dim": env.obs_dim,
        "state_dim": env.state_dim,
        "act_dim": env.act_dim,
        "num_envs": env.num_envs,
        "num_agents": env.num_agents,
        "mean_reward_per_step": float(total.mean() / max(steps, 1)),
    }


def _check_obs(env: BenchEnv, obs: Obs) -> None:
    assert obs.obs.shape == (env.num_envs, env.num_agents, env.obs_dim), (
        f"{type(env).__name__}: obs shape {obs.obs.shape} != "
        f"{(env.num_envs, env.num_agents, env.obs_dim)}"
    )
    assert obs.state.shape == (env.num_envs, env.state_dim), (
        f"{type(env).__name__}: state shape {obs.state.shape} != "
        f"{(env.num_envs, env.state_dim)}"
    )
    assert np.isfinite(obs.obs).all(), f"{type(env).__name__}: non-finite observation"
