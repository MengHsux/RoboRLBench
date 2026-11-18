"""Base classes shared by all twenty baselines.

Two families
------------
``OnPolicyAgent``
    Collects a rollout, then performs several epochs of minibatch updates
    (PPO family, MAPPO family).  The rollout is produced by the runner through
    :meth:`act_batch`; :meth:`update` consumes a :class:`RolloutBuffer`.

``OffPolicyAgent``
    Pushes every transition into its own :class:`ReplayBuffer` and performs
    ``grad_steps`` gradient steps per vectorised environment step (SAC/TD3
    family and their MARL counterparts).

Both families share the same :meth:`act` signature, the same observation
normaliser and the same checkpoint format, so the runners are algorithm
agnostic.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .buffers import ReplayBuffer, RolloutBuffer
from .normalizer import RunningMeanStd

__all__ = ["AgentBase", "OnPolicyAgent", "OffPolicyAgent"]


class AgentBase(abc.ABC):
    """Common bookkeeping: device, normalisers, checkpointing."""

    #: ``"on_policy"`` or ``"off_policy"``
    kind: str = "off_policy"
    #: short identifier used by the CLI and in log directories
    name: str = "base"
    #: set to True by the MARL family; the CLI uses it to check env compatibility
    multi_agent: bool = False

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        normalize_obs: bool = True,
        normalize_state: bool = False,
        **_: Any,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim or obs_dim)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.seed = int(seed)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

        self.obs_rms: Optional[RunningMeanStd] = (
            RunningMeanStd(self.obs_dim) if normalize_obs else None
        )
        self.state_rms: Optional[RunningMeanStd] = (
            RunningMeanStd(self.state_dim) if normalize_state else None
        )
        self._train_mode = True

    # ------------------------------------------------------------------ #
    # Observation handling
    # ------------------------------------------------------------------ #
    def _update_stats_np(self, obs: np.ndarray, state: np.ndarray) -> None:
        if self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(-1, self.obs_dim))
        if self.state_rms is not None:
            self.state_rms.update(state.reshape(-1, self.state_dim))

    def norm_obs_np(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs) if self.obs_rms is not None else obs

    def norm_state_np(self, state: np.ndarray) -> np.ndarray:
        return self.state_rms.normalize(state) if self.state_rms is not None else state

    def norm_obs_t(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs) if self.obs_rms is not None else obs

    def norm_state_t(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_rms.normalize(state) if self.state_rms is not None else state

    def _to_tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def act(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return an action of shape ``(num_envs, num_agents, act_dim)``."""
        obs = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            o = self.norm_obs_t(self._to_tensor(obs))
            a = self._act_impl(o, deterministic)
        return a.detach().cpu().numpy().astype(np.float32)

    @abc.abstractmethod
    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Torch-only action selection on normalised observations."""

    def observe(self, obs: np.ndarray, state: Optional[np.ndarray] = None) -> None:
        """Feed the normaliser with freshly collected observations."""
        self._update_stats_np(np.asarray(obs, dtype=np.float32),
                              np.asarray(state if state is not None else obs, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, torch.nn.Module]:
        """Trainable / target networks that must be checkpointed."""
        return {}

    def extra_state(self) -> Dict[str, Any]:
        """Algorithm specific buffers (optimisers, temperature, ...)."""
        return {}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        """Inverse of :meth:`extra_state`."""

    def state_dict(self) -> Dict[str, Any]:
        return {
            "algo": self.name,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "num_envs": self.num_envs,
            "modules": {k: m.state_dict() for k, m in self.modules().items()},
            "extra": self.extra_state(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "state_rms": self.state_rms.state_dict() if self.state_rms else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        modules = self.modules()
        for key, value in state["modules"].items():
            if key not in modules:
                raise KeyError(f"{self.name}: checkpoint contains unknown module {key!r}")
            modules[key].load_state_dict(value)
        self.load_extra_state(state.get("extra", {}))
        if self.obs_rms is not None and state.get("obs_rms"):
            self.obs_rms.load_state_dict(state["obs_rms"])
        if self.state_rms is not None and state.get("state_rms"):
            self.state_rms.load_state_dict(state["state_rms"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.load_state_dict(state)

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        self._train_mode = True
        for m in self.modules().values():
            m.train()

    def eval(self) -> None:
        self._train_mode = False
        for m in self.modules().values():
            m.eval()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{type(self).__name__}(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
                f"num_agents={self.num_agents}, device={self.device})")


class OnPolicyAgent(AgentBase):
    """Agent that trains from a :class:`RolloutBuffer`."""

    kind = "on_policy"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        rollout_steps: int = 24,
        gamma: float = 0.99,
        lam: float = 0.95,
        num_minibatches: int = 4,
        num_epochs: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs)
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.num_minibatches = int(num_minibatches)
        self.num_epochs = int(num_epochs)

    # ------------------------------------------------------------------ #
    def make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            steps=self.rollout_steps,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            gamma=self.gamma,
            lam=self.lam,
            device=str(self.device),
        )

    @abc.abstractmethod
    def act_batch(self, obs: torch.Tensor, state: torch.Tensor
                  ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Return ``(action, log_prob, value)`` for a batch of observations."""

    @abc.abstractmethod
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Consume a filled rollout buffer and return the training metrics."""

    @property
    def minibatch_size(self) -> int:
        total = self.rollout_steps * self.num_envs
        return max(1, total // max(1, self.num_minibatches))


class OffPolicyAgent(AgentBase):
    """Agent with its own replay buffer and gradient steps per env step."""

    kind = "off_policy"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        warmup: int = 2_000,
        grad_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.warmup = int(warmup)
        self.grad_steps = int(grad_steps)
        rows = max(1, int(buffer_size) // max(1, self.num_envs))
        self.buffer = ReplayBuffer(
            capacity=rows,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            device=str(self.device),
        )

    # ------------------------------------------------------------------ #
    def store(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        timeout: Optional[np.ndarray] = None,
    ) -> None:
        self.observe(obs, state)
        self.buffer.add(obs, state, action, reward, next_obs, next_state, done, timeout)

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.warmup, self.batch_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        return self.buffer.sample(self.batch_size, generator=self._gen)

    @abc.abstractmethod
    def update(self) -> Dict[str, float]:
        """Perform one gradient step and return the losses."""

    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_batch(
        batch: Dict[str, torch.Tensor],
        obs_rms: Optional[RunningMeanStd],
        state_rms: Optional[RunningMeanStd],
    ) -> Dict[str, torch.Tensor]:
        """Apply the running normalisers to a sampled minibatch."""
        if obs_rms is not None:
            batch["obs"] = obs_rms.normalize(batch["obs"])
            batch["next_obs"] = obs_rms.normalize(batch["next_obs"])
        if state_rms is not None:
            batch["state"] = state_rms.normalize(batch["state"])
            batch["next_state"] = state_rms.normalize(batch["next_state"])
        return batch
