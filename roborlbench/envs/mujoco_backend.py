"""MuJoCo adapters (CPU).

Two upstream projects expose a MuJoCo replica of their Isaac task, which makes
a full reinforcement-learning benchmark possible on a machine without a GPU:

* ``legged_hunt-main/mujoco_tools/wolf_env.py`` - :class:`WolfMuJoCoEnv`,
  ``num_fields`` parallel arenas x ``num_wolfs`` Go2 quadrupeds per arena.
* ``OpenSO-101-main/src/openso101/offpolicy/env.py`` - :class:`VecSO101Env`,
  the SO-101 arm with four manipulation tasks.

Both are imported lazily; nothing in this module is loaded unless one of the
``*-mujoco`` environments is instantiated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ._simpaths import import_module
from .base import BenchEnv, Obs, Step

__all__ = ["make_hunt_multi", "make_so101", "WolfMuJoCoAdapter", "SO101MuJoCoAdapter"]


# --------------------------------------------------------------------------- #
# Multi-robot wolf hunt
# --------------------------------------------------------------------------- #
class WolfMuJoCoAdapter(BenchEnv):
    """Wrap ``mujoco_tools.wolf_env.WolfMuJoCoEnv`` behind :class:`BenchEnv`.

    The upstream environment exposes

    ``obs (F, W, 52)``, ``privileged_obs (F, W, 55)``, ``rewards (F*W,)`` and
    ``dones (F*W,)`` and already performs per-robot auto-reset.  The adapter
    promotes this to the benchmark contract: rewards become ``(F, W)``, the done
    mask becomes per-arena (``any`` over the wolves) and the centralised state
    is the concatenation of the privileged observations of all wolves.
    """

    backend = "mujoco"

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cpu",
        num_agents: int = 3,
        prey_speed: float = 0.0,
        terminate_on_contact: bool = True,
        episode_length_s: Optional[float] = None,
        collect_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        mod = import_module("legged_hunt", "mujoco_tools.wolf_env")
        self._inner = mod.WolfMuJoCoEnv(
            num_wolfs=int(num_agents),
            num_fields=int(num_envs),
            seed=int(seed),
            prey_speed=float(prey_speed),
            terminate_on_contact=bool(terminate_on_contact),
            episode_length_s=episode_length_s,
            **kwargs,
        )
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(self._inner.num_obs)
        self.act_dim = int(self._inner.num_actions)
        self.state_dim = self.num_agents * int(self._inner.num_privileged_obs)
        self.max_episode_steps = int(self._inner.max_episode_length)
        self.collect_metrics = bool(collect_metrics)

    # ------------------------------------------------------------------ #
    def reset(self) -> Obs:
        obs, _ = self._inner.reset()
        priv = self._inner.get_privileged_observations()
        return self._make_obs(_as_np(obs), _as_np(priv).reshape(self.num_envs, -1))

    def step(self, actions: np.ndarray) -> Step:
        actions = self.check_actions(actions)
        import torch  # local import keeps the module import-light

        flat = torch.as_tensor(
            actions.reshape(self.num_envs * self.num_agents, self.act_dim),
            dtype=torch.float32,
        )
        obs, priv, rew, done, extras = self._inner.step(flat)
        rew = _as_np(rew).reshape(self.num_envs, self.num_agents)
        done_agents = _as_np(done).reshape(self.num_envs, self.num_agents)
        done_env = done_agents.any(axis=1)

        info: Dict[str, Any] = {}
        timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
        if timeouts is not None:
            info["timeout"] = _as_np(timeouts).reshape(self.num_envs, -1).any(axis=1)
        if self.collect_metrics:
            try:
                info["metrics"] = {
                    k: float(v)
                    for k, v in self._inner.task_metrics().items()
                    if isinstance(v, (int, float, np.floating))
                }
            except Exception:  # pragma: no cover - metrics are best effort
                pass

        return Step(
            obs=_as_np(obs),
            state=_as_np(priv).reshape(self.num_envs, -1),
            reward=rew,
            done=done_env,
            info=info,
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        self._inner = None


def make_hunt_multi(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    num_agents: int = 3,
    **kwargs: Any,
) -> WolfMuJoCoAdapter:
    """``hunt-multi-mujoco`` factory."""
    return WolfMuJoCoAdapter(
        num_envs=num_envs, seed=seed, device=device, num_agents=num_agents, **kwargs
    )


# --------------------------------------------------------------------------- #
# SO-101 manipulation
# --------------------------------------------------------------------------- #
class SO101MuJoCoAdapter(BenchEnv):
    """Wrap ``openso101.offpolicy.env.VecSO101Env`` behind :class:`BenchEnv`."""

    backend = "mujoco"
    num_agents = 1

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cpu",
        task: str = "reach",
        randomize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        mod = import_module("openso101", "openso101.offpolicy.env")
        self._inner = mod.VecSO101Env(
            task=str(task), num_envs=int(num_envs), seed=int(seed),
            randomize=bool(randomize),
        )
        self.task_name = str(task)
        self.num_envs = int(num_envs)
        self.obs_dim = int(mod.OBS_DIM)
        self.act_dim = int(mod.ACTION_DIM)
        self.state_dim = self.obs_dim
        self.max_episode_steps = int(getattr(self._inner.envs[0], "max_episode_steps", 200))

    def reset(self) -> Obs:
        obs = _as_np(self._inner.reset())
        return self._make_obs(obs[:, None, :], obs)

    def step(self, actions: np.ndarray) -> Step:
        actions = self.check_actions(actions)
        obs, rew, done, terminated, success, _infos = self._inner.step(
            actions[:, 0, :].astype(np.float32)
        )
        obs = _as_np(obs)
        rew = _as_np(rew).astype(np.float32)[:, None]
        done = _as_np(done).astype(bool)
        terminated = _as_np(terminated).astype(bool)
        info: Dict[str, Any] = {
            "timeout": done & ~terminated,
            "metrics": {"success_rate": float(np.mean(_as_np(success)))},
        }
        return Step(
            obs=obs[:, None, :],
            state=obs,
            reward=rew,
            done=done,
            info=info,
        )

    def close(self) -> None:  # pragma: no cover
        self._inner.close()


def make_so101(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    task: str = "reach",
    **kwargs: Any,
) -> SO101MuJoCoAdapter:
    """``so101-mujoco`` factory."""
    return SO101MuJoCoAdapter(num_envs=num_envs, seed=seed, device=device, task=task, **kwargs)


# --------------------------------------------------------------------------- #
def _as_np(x: Any) -> np.ndarray:
    """Convert torch tensors / arrays to ``float32`` numpy without a hard dep."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
