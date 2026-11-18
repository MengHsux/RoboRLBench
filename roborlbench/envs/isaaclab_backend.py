"""Isaac Lab adapters (GPU).

Wraps the Isaac Lab tasks shipped by ``go2_rl_lab`` (``Go2_control``) and
``openso101`` (``OpenSO-101``):

===================== ==========================================
registry key          upstream gym id
===================== ==========================================
``go2-flat``          ``go2-flat``
``go2-mixed``         ``go2-mixed``
``so101-lift``        ``OpenSO101-Lift-v0``
``so101-pickplace``   ``OpenSO101-PickPlace-v0``
``so101-stack``       ``OpenSO101-Stack-v0``
===================== ==========================================

Isaac Lab environments are ``gymnasium`` environments that return
``(obs, reward, terminated, truncated, extras)`` from ``step`` and reset
terminated copies internally.  Observations are dictionaries; the benchmark
uses the ``"policy"`` group as the actor input and every other group
concatenated as the centralised critic state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ._simpaths import ensure_importable, import_module
from .base import BenchEnv, Obs, Step

__all__ = [
    "IsaacLabAdapter",
    "make_go2_flat",
    "make_go2_mixed",
    "make_so101_lift",
    "make_so101_pickplace",
    "make_so101_stack",
]

_APP: Any = None


def _launch_app(headless: bool = True, enable_cameras: bool = False) -> Any:
    """Start Isaac Sim exactly once per process and return the app handle."""
    global _APP
    if _APP is not None:
        return _APP
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:  # pragma: no cover - GPU only
        raise ImportError(
            "Isaac Lab is required for the ``isaaclab`` backend but could not "
            "be imported. Install it following docs/ENVIRONMENTS.md "
            f"(original error: {exc})."
        ) from exc
    launcher = AppLauncher({"headless": bool(headless), "enable_cameras": bool(enable_cameras)})
    _APP = launcher.app
    return _APP


def _register_task(repo: str, task: str) -> None:
    """Import the package that registers ``task`` in the gym registry."""
    import gymnasium as gym

    if task in gym.registry:
        return
    if repo == "go2_rl_lab":
        ensure_importable("go2_rl_lab")
        import_module("go2_rl_lab", "go2_rl_lab.tasks")          # noqa: F401
    elif repo == "openso101":
        ensure_importable("openso101")
        import_module("openso101", "openso101.tasks")            # noqa: F401
    else:  # pragma: no cover - defensive
        raise KeyError(repo)

    if task not in gym.registry:
        raise RuntimeError(
            f"gym id {task!r} is not registered after importing {repo}.tasks; "
            f"check that the upstream package is installed in this environment."
        )


class IsaacLabAdapter(BenchEnv):
    """Adapter from an Isaac Lab ``ManagerBasedRLEnv`` to :class:`BenchEnv`."""

    backend = "isaaclab"
    num_agents = 1

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cuda:0",
        repo: str = "go2_rl_lab",
        task: str = "go2-flat",
        headless: bool = True,
        policy_group: str = "policy",
        **make_kwargs: Any,
    ) -> None:
        super().__init__()
        _launch_app(headless=headless)
        _register_task(repo, task)

        import gymnasium as gym

        self._env = gym.make(
            task,
            num_envs=int(num_envs),
            disable_env_checker=True,
            **make_kwargs,
        )
        try:
            self._env.unwrapped.seed(int(seed))
        except AttributeError:  # pragma: no cover - Isaac Lab seeds via cfg
            pass
        self.task_name = task
        self.num_envs = int(self._env.unwrapped.num_envs)
        self.num_agents = 1
        self.policy_group = policy_group

        act_space = self._env.action_space
        self.act_dim = int(np.prod(act_space.shape[-1:]))
        self.act_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)[: self.act_dim]
        self.act_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)[: self.act_dim]

        # infer the observation layout from a real reset
        obs_dict, _ = self._env.reset()
        obs_vec, state_vec = self._split_obs(obs_dict)
        self.obs_dim = int(obs_vec.shape[-1])
        self.state_dim = int(state_vec.shape[-1])

        cfg = self._env.unwrapped.cfg
        sim_cfg = getattr(cfg, "sim", None)
        sim_dt = float(getattr(sim_cfg, "dt", 0.005)) if sim_cfg is not None else 0.005
        step_dt = max(sim_dt * float(getattr(cfg, "decimation", 1)), 1e-6)
        horizon = float(getattr(cfg, "episode_length_s", 20.0))
        self.max_episode_steps = int(round(horizon / step_dt))
        self._episode_step = np.zeros(self.num_envs, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def _split_obs(self, obs_dict: Any) -> "tuple[np.ndarray, np.ndarray]":
        if not isinstance(obs_dict, dict):
            vec = _np(obs_dict).reshape(self.num_envs, -1)
            return vec, vec
        if self.policy_group in obs_dict:
            policy = _np(obs_dict[self.policy_group]).reshape(self.num_envs, -1)
        else:
            key = next(iter(obs_dict))
            policy = _np(obs_dict[key]).reshape(self.num_envs, -1)
        groups: List[np.ndarray] = [
            _np(v).reshape(self.num_envs, -1) for v in obs_dict.values()
        ]
        return policy, np.concatenate(groups, axis=-1)

    def reset(self) -> Obs:
        obs_dict, _ = self._env.reset()
        self._episode_step[:] = 0
        vec, state = self._split_obs(obs_dict)
        return self._make_obs(vec[:, None, :], state)

    def step(self, actions: np.ndarray) -> Step:
        actions = self.check_actions(actions)
        obs_dict, rew, terminated, truncated, extras = self._env.step(actions[:, 0, :])
        self._episode_step += 1
        vec, state = self._split_obs(obs_dict)
        rew = _np(rew).reshape(self.num_envs, 1)
        terminated = _np(terminated).astype(bool).reshape(self.num_envs)
        truncated = _np(truncated).astype(bool).reshape(self.num_envs)
        self._episode_step[terminated | truncated] = 0

        info: Dict[str, Any] = {"timeout": truncated & ~terminated}
        metrics = _extra_metrics(extras)
        if metrics:
            info["metrics"] = metrics
        return Step(
            obs=vec[:, None, :],
            state=state,
            reward=rew,
            done=terminated | truncated,
            info=info,
        )

    def close(self) -> None:  # pragma: no cover
        self._env.close()


def _extra_metrics(extras: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(extras, dict):
        return out
    log = extras.get("log", {})
    if isinstance(log, dict):
        for key, value in log.items():
            try:
                arr = _np(value)
                if arr.size:
                    out[str(key)] = float(arr.mean())
            except Exception:  # pragma: no cover
                continue
    for key in ("success", "is_success"):
        if key in extras:
            try:
                out["success_rate"] = float(_np(extras[key]).mean())
            except Exception:  # pragma: no cover
                pass
    return out


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_go2_flat(num_envs: int = 1, seed: int = 0, device: str = "cuda:0", **kw: Any) -> IsaacLabAdapter:
    return IsaacLabAdapter(num_envs=num_envs, seed=seed, device=device,
                           repo="go2_rl_lab", task="go2-flat", **kw)


def make_go2_mixed(num_envs: int = 1, seed: int = 0, device: str = "cuda:0", **kw: Any) -> IsaacLabAdapter:
    return IsaacLabAdapter(num_envs=num_envs, seed=seed, device=device,
                           repo="go2_rl_lab", task="go2-mixed", **kw)


def make_so101_lift(num_envs: int = 1, seed: int = 0, device: str = "cuda:0", **kw: Any) -> IsaacLabAdapter:
    return IsaacLabAdapter(num_envs=num_envs, seed=seed, device=device,
                           repo="openso101", task="OpenSO101-Lift-v0", **kw)


def make_so101_pickplace(num_envs: int = 1, seed: int = 0, device: str = "cuda:0", **kw: Any) -> IsaacLabAdapter:
    return IsaacLabAdapter(num_envs=num_envs, seed=seed, device=device,
                           repo="openso101", task="OpenSO101-PickPlace-v0", **kw)


def make_so101_stack(num_envs: int = 1, seed: int = 0, device: str = "cuda:0", **kw: Any) -> IsaacLabAdapter:
    return IsaacLabAdapter(num_envs=num_envs, seed=seed, device=device,
                           repo="openso101", task="OpenSO101-Stack-v0", **kw)


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
