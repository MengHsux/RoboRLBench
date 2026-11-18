"""The unified environment interface used by every algorithm in the benchmark.

The benchmark exposes *one* API no matter which simulator is underneath::

    spec  = get_env_spec("hunt-multi-mujoco")
    env   = make_env(spec, num_envs=8, seed=0)
    obs   = env.reset()                      # Obs(obs=(8, 3, 52), state=(8, 158))
    step  = env.step(actions)                # actions: (8, 3, 12) in [-1, 1]

Conventions
-----------
* ``num_envs`` independent copies of the task are simulated in parallel.
* ``num_agents`` is the number of co-operating robots inside one task copy.
  Single-robot tasks use ``num_agents == 1``; the runners then squeeze the agent
  axis so the 10 single-agent baselines see plain ``(N, obs_dim)`` tensors.
* Actions are **always** normalised to ``[-1, 1]`` per actuator.  Adapters are
  responsible for mapping them onto the simulator's own action space.
* ``step`` performs *auto-reset*: environments that terminate inside a step are
  reset before the observation is returned, exactly like ``rsl_rl`` and
  Isaac Lab do.  The returned ``done`` mask tells the algorithm which
  transitions must not bootstrap from ``next_obs``.
* ``step.done`` marked ``info["timeout"]`` are truncations (time limit); the
  off-policy agents bootstrap from those transitions, which avoids the
  "time-limit-is-terminal" bias.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #
@dataclass
class EnvSpec:
    """Static description of a benchmark environment.

    Attributes
    ----------
    name:
        Registry key, e.g. ``"go2-flat"``.
    kind:
        ``"single"`` (one robot per task copy, 10 single-agent baselines) or
        ``"multi"`` (several co-operating robots, 10 multi-agent baselines).
    backend:
        ``"analytic"`` (built in, CPU), ``"mujoco"`` (CPU), ``"isaacgym"`` or
        ``"isaaclab"`` (GPU).  Only used for diagnostics and for choosing the
        default number of parallel environments.
    task:
        Backend specific task identifier, e.g. ``"wolf"`` or
        ``"OpenSO101-PickPlace-v0"``.
    num_agents:
        Number of robots inside one task copy.
    max_episode_steps:
        Default horizon in control steps (used for logging, not enforced here).
    entry_point:
        ``"module:attribute"`` of the factory function.  The factory has the
        signature ``factory(num_envs: int, seed: int, device: str, **kwargs)``
        and returns a :class:`BenchEnv`.
    description:
        One line human readable summary.
    kwargs:
        Default keyword arguments handed to the factory.
    """

    name: str
    kind: str
    backend: str
    task: str
    entry_point: str
    num_agents: int = 1
    max_episode_steps: int = 1000
    description: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("single", "multi"):
            raise ValueError(f"EnvSpec.kind must be 'single' or 'multi', got {self.kind!r}")
        if self.num_agents < 1:
            raise ValueError("EnvSpec.num_agents must be >= 1")
        if self.kind == "single" and self.num_agents != 1:
            raise ValueError(f"{self.name}: a 'single' task must have num_agents == 1")

    @property
    def is_multi_agent(self) -> bool:
        return self.num_agents > 1


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    """Micro-batch of observations.

    ``obs``   : ``(num_envs, num_agents, obs_dim)`` float32
    ``state`` : ``(num_envs, state_dim)`` float32 - the *global* (centralised)
                state used by the CTDE critics.  For single-robot tasks it is
                simply a copy of the observation.
    """

    obs: np.ndarray
    state: np.ndarray

    def squeeze_agents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(num_envs, obs_dim)`` / ``(num_envs, state_dim)`` views."""
        if self.obs.shape[1] != 1:
            raise ValueError("squeeze_agents() called on a multi-agent observation")
        return self.obs[:, 0], self.state


@dataclass
class Step:
    """One transition for all parallel environments.

    ``obs``/``state`` are the *next* observations **after auto-reset**.
    ``reward`` is ``(num_envs, num_agents)``; ``done`` is ``(num_envs,)``.
    ``info`` may contain:

    * ``"timeout"``: ``(num_envs,)`` bool, True for truncations (time limit).
    * ``"metrics"``: ``dict[str, float]`` task specific scalars for logging.
    * any other simulator specific payload (ignored by the algorithms).
    """

    obs: np.ndarray
    state: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def timeout(self) -> np.ndarray:
        t = self.info.get("timeout", None)
        if t is None:
            return np.zeros_like(self.done, dtype=bool)
        return np.asarray(t, dtype=bool)

    def squeeze_agents(self):
        """``(next_obs, next_state, reward, done)`` with the agent axis removed."""
        if self.obs.shape[1] != 1:
            raise ValueError("squeeze_agents() called on a multi-agent step")
        return self.obs[:, 0], self.state, self.reward[:, 0], self.done


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BenchEnv(abc.ABC):
    """Vectorised, auto-resetting, multi-agent aware environment."""

    #: filled in by subclasses
    num_envs: int = 1
    num_agents: int = 1
    obs_dim: int = 0
    act_dim: int = 0
    max_episode_steps: int = 1000
    #: normalised action bounds, shape ``(act_dim,)``.  The benchmark always
    #: uses symmetric ``[-1, 1]`` boxes; the attributes exist so that wrappers
    #: can rescale if a backend needs it.
    act_low: np.ndarray = np.array([-1.0], dtype=np.float32)
    act_high: np.ndarray = np.array([1.0], dtype=np.float32)
    #: size of the centralised state vector
    state_dim: int = 0
    #: name of the backend, mostly for logging
    backend: str = "unknown"

    def __init__(self) -> None:
        if self.state_dim == 0:
            self.state_dim = self.obs_dim
        self.act_low = np.full(self.act_dim, -1.0, dtype=np.float32)
        self.act_high = np.full(self.act_dim, 1.0, dtype=np.float32)

    # -- mandatory API ------------------------------------------------------ #
    @abc.abstractmethod
    def reset(self) -> Obs:
        """Reset **all** parallel environments and return the first observation."""

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Step:
        """Advance the simulation by one control step.

        Parameters
        ----------
        actions:
            ``(num_envs, num_agents, act_dim)`` float array in ``[-1, 1]``.
        """

    def close(self) -> None:  # pragma: no cover - simulator specific
        """Release simulator resources.  Optional."""

    # -- helpers ------------------------------------------------------------ #
    @property
    def is_multi_agent(self) -> bool:
        return self.num_agents > 1

    @property
    def total_agents(self) -> int:
        return self.num_envs * self.num_agents

    def check_actions(self, actions: np.ndarray) -> np.ndarray:
        expected = (self.num_envs, self.num_agents, self.act_dim)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != expected:
            raise ValueError(
                f"{type(self).__name__}.step expected actions of shape {expected}, "
                f"got {actions.shape}"
            )
        return actions

    def _make_obs(self, obs: np.ndarray, state: Optional[np.ndarray] = None) -> Obs:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if state is None:
            state = obs.reshape(self.num_envs, -1)
        return Obs(obs=obs, state=np.ascontiguousarray(state, dtype=np.float32))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(num_envs={self.num_envs}, "
            f"num_agents={self.num_agents}, obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim})"
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ENVS: Dict[str, EnvSpec] = {}


def register_env(
    name: str,
    entry_point: str,
    *,
    kind: str = "single",
    backend: str = "analytic",
    task: Optional[str] = None,
    num_agents: int = 1,
    max_episode_steps: int = 1000,
    description: str = "",
    **kwargs: Any,
) -> EnvSpec:
    """Register an environment factory under ``name``."""
    if name in ENVS:
        raise KeyError(f"environment {name!r} is already registered")
    spec = EnvSpec(
        name=name,
        kind=kind,
        backend=backend,
        task=task or name,
        entry_point=entry_point,
        num_agents=num_agents,
        max_episode_steps=max_episode_steps,
        description=description,
        kwargs=dict(kwargs),
    )
    ENVS[name] = spec
    return spec


def get_env_spec(name: str) -> EnvSpec:
    if name not in ENVS:
        raise KeyError(
            f"unknown environment {name!r}. Available: {sorted(ENVS)}"
        )
    return ENVS[name]


def list_envs(kind: Optional[str] = None) -> Dict[str, str]:
    """Return ``{name: description}`` optionally filtered by ``kind``."""
    return {
        n: s.description
        for n, s in sorted(ENVS.items())
        if kind is None or s.kind == kind
    }
