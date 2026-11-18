"""Isaac Gym adapters (GPU).

Wraps the ``legged_gym`` tasks that ship with ``legged_hunt-main`` and
``legged_hunt-main-single``:

=============== ==========================================================
registry key    upstream task
=============== ==========================================================
``hunt-multi-isaacgym``   ``wolf`` (3 Go2 / arena, MARL)
``hunt-single-isaacgym``  ``wolf`` (1 Go2 / arena, single agent)
``go2-flat-isaacgym``     ``go2``  (flat-ground velocity tracking)
=============== ==========================================================

The ``legged_gym`` environments already auto-reset, expose
``(obs, privileged_obs, rew, dones, extras)`` from ``step`` and keep the
observation buffer on the env object.  The adapter only has to re-shape the
per-robot tensors into the benchmark's ``(num_envs, num_agents, dim)`` layout.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from ._simpaths import import_module
from .base import BenchEnv, Obs, Step

__all__ = [
    "LeggedGymAdapter",
    "make_hunt_multi",
    "make_hunt_single",
    "make_go2_flat",
]


def _make_args(task: str, num_envs: int, seed: int, headless: bool, sim_device: str) -> SimpleNamespace:
    """Minimal replacement for ``legged_gym.utils.get_args()``.

    ``get_args()`` parses ``sys.argv``, which would swallow RoboRLBench's own
    flags.  We build the namespace by hand instead; every attribute read by
    ``parse_sim_params`` / ``update_cfg_from_args`` is present.
    """
    use_gpu = sim_device.startswith("cuda")
    return SimpleNamespace(
        task=task,
        resume=False,
        experiment_name="roborlbench",
        run_name="tmp",
        load_run="-1",
        checkpoint=-1,
        num_envs=None,          # do not clobber the cfg we pass explicitly
        seed=int(seed),
        max_iterations=None,
        rl_device=sim_device,
        sim_device=sim_device,
        physics_engine=None,    # filled below from gymapi
        device=sim_device,
        use_gpu=use_gpu,
        use_gpu_pipeline=use_gpu,
        pipeline="gpu" if use_gpu else "cpu",
        flex=False,
        num_threads=0,
        subscenes=0,
        headless=bool(headless),
        compute_device_id=0,
        graphics_device_id=0,
        sim_device_id=0,
        sim_device_type="cuda" if use_gpu else "cpu",
    )


def _patch_single_agent_neighbours(wolf_robot_module) -> None:
    """Make the wolf env's neighbour features well defined for a single wolf.

    ``square_hunting.compute_edges_vec`` gathers the two nearest neighbours and
    therefore assumes ``num_wolfs >= 3``.  When the arena holds exactly one
    robot there *are* no neighbours, so the correct value of every neighbour
    feature is zero.  We patch the module-level name that ``wolf_robot`` calls
    instead of editing the upstream repository.
    """
    import torch

    original = wolf_robot_module.compute_edges_vec

    def safe_edges(tensor, *args, **kwargs):
        if tensor.shape[1] >= 3:
            return original(tensor, *args, **kwargs)
        n, w = tensor.shape[0], tensor.shape[1]
        zeros = lambda *shape: torch.zeros(*shape, device=tensor.device)
        return zeros(n, w, 3), zeros(n, w, 2), zeros(n, w), zeros(n, w, 1)

    wolf_robot_module.compute_edges_vec = safe_edges


class LeggedGymAdapter(BenchEnv):
    """Adapter for any ``legged_gym`` task.

    Parameters
    ----------
    repo:
        ``"legged_hunt"`` or ``"legged_hunt_single"`` - which checkout to import.
    task:
        Registered legged_gym task name (``"wolf"`` or ``"go2"``).
    num_agents:
        Wolves per arena.  Ignored by single-robot tasks.
    """

    backend = "isaacgym"

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cuda:0",
        repo: str = "legged_hunt",
        task: str = "wolf",
        num_agents: int = 1,
        headless: bool = True,
        max_episode_s: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        gym = import_module(repo, "isaacgym")
        _ = gym  # import has the side effect of registering the gymapi backends

        envs_pkg = import_module(repo, "legged_gym.envs")          # noqa: F401 (registers tasks)
        task_registry = import_module(repo, "legged_gym.utils").task_registry

        args = _make_args(task, num_envs, seed, headless, device)
        import isaacgym.gymapi as gymapi  # noqa: F401  (imported after sys.path update)
        args.physics_engine = gymapi.SIM_PHYSX

        env_cfg = copy.deepcopy(task_registry.get_cfgs(task)[0])
        env_cfg.seed = int(seed)
        if hasattr(env_cfg.env, "num_fields") and hasattr(env_cfg.env, "num_wolfs_per_field"):
            # wolf task: num_envs is the number of *actors*
            env_cfg.env.num_fields = int(num_envs)
            env_cfg.env.num_wolfs_per_field = int(num_agents)
            env_cfg.env.num_envs = int(num_envs) * int(num_agents)
            if hasattr(env_cfg.env, "num_sheeps_per_field"):
                n_sheep = env_cfg.env.num_sheeps_per_field if getattr(env_cfg.env, "actor_prey", True) else 0
                env_cfg.env.num_alls = env_cfg.env.num_fields * (
                    env_cfg.env.num_wolfs_per_field + n_sheep
                )
            if max_episode_s is not None:
                env_cfg.env.episode_length_s = float(max_episode_s)
        else:
            env_cfg.env.num_envs = int(num_envs)
            if max_episode_s is not None:
                env_cfg.env.episode_length_s = float(max_episode_s)

        if task == "wolf":
            wolf_robot = import_module(repo, "legged_gym.envs.wolf.wolf_robot")
            _patch_single_agent_neighbours(wolf_robot)

        env, _ = task_registry.make_env(
            name=task,
            args=args,
            env_cfg=env_cfg,
        )
        self._env = env
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(env.num_obs)
        self.act_dim = int(env.num_actions)
        priv = getattr(env, "num_privileged_obs", None)
        self.state_dim = (int(priv) if priv else self.obs_dim) * self.num_agents
        self.max_episode_steps = int(env.max_episode_length)
        self.device = device

    # ------------------------------------------------------------------ #
    def _state_from_priv(self, priv: np.ndarray) -> np.ndarray:
        return priv.reshape(self.num_envs, -1)

    def reset(self) -> Obs:
        self._env.reset()
        obs = _np(self._env.obs_buf).reshape(self.num_envs, self.num_agents, self.obs_dim)
        priv = getattr(self._env, "privileged_obs_buf", None)
        priv = obs.reshape(self.num_envs, -1) if priv is None else _np(priv)
        state = self._state_from_priv(priv.reshape(self.num_envs, self.num_agents, -1))
        return self._make_obs(obs, state)

    def step(self, actions: np.ndarray) -> Step:
        actions = self.check_actions(actions)
        import torch

        flat = torch.as_tensor(
            actions.reshape(self.num_envs * self.num_agents, self.act_dim),
            dtype=torch.float32,
            device=self.device,
        )
        obs, priv, rew, done, _extras = self._env.step(flat)
        obs = _np(obs).reshape(self.num_envs, self.num_agents, self.obs_dim)
        priv = _np(priv) if priv is not None else obs
        priv = priv.reshape(self.num_envs, self.num_agents, -1)
        rew = _np(rew).reshape(self.num_envs, self.num_agents)
        done_agents = _np(done).reshape(self.num_envs, self.num_agents)
        timeout_agents = _np(self._env.time_out_buf).reshape(self.num_envs, self.num_agents)

        info: Dict[str, Any] = {"timeout": timeout_agents.any(axis=1)}
        metrics = self._metrics()
        if metrics:
            info["metrics"] = metrics
        return Step(
            obs=obs,
            state=self._state_from_priv(priv.reshape(self.num_envs, -1)),
            reward=rew,
            done=done_agents.any(axis=1),
            info=info,
        )

    def _metrics(self) -> Dict[str, float]:
        """Task metrics when the upstream env exposes them."""
        out: Dict[str, float] = {}
        env = self._env
        try:
            if hasattr(env, "task_metrics"):          # MuJoCo-style helper
                out.update({k: float(v) for k, v in env.task_metrics().items()
                            if isinstance(v, (int, float, np.floating))})
            elif hasattr(env, "encirclement_ratio"):
                out["encirclement_ratio"] = float(_np(env.encirclement_ratio).mean())
        except Exception:  # pragma: no cover - metrics are best effort
            pass
        return out

    def close(self) -> None:  # pragma: no cover
        self._env = None


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_hunt_multi(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cuda:0",
    num_agents: int = 3,
    **kwargs: Any,
) -> LeggedGymAdapter:
    """``hunt-multi-isaacgym`` factory (legged_hunt-main, 3 wolves)."""
    return LeggedGymAdapter(
        num_envs=num_envs, seed=seed, device=device, repo="legged_hunt",
        task="wolf", num_agents=num_agents, **kwargs,
    )


def make_hunt_single(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cuda:0",
    **kwargs: Any,
) -> LeggedGymAdapter:
    """``hunt-single-isaacgym`` factory (legged_hunt-main-single, 1 wolf)."""
    return LeggedGymAdapter(
        num_envs=num_envs, seed=seed, device=device, repo="legged_hunt_single",
        task="wolf", num_agents=1, **kwargs,
    )


def make_go2_flat(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cuda:0",
    **kwargs: Any,
) -> LeggedGymAdapter:
    """``go2-flat-isaacgym`` factory (legged_gym ``go2`` flat task)."""
    return LeggedGymAdapter(
        num_envs=num_envs, seed=seed, device=device, repo="legged_hunt",
        task="go2", num_agents=1, **kwargs,
    )


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)
