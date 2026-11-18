"""Replay and rollout buffers.

Every buffer keeps the **agent axis** explicitly, so the exact same code path
serves a 1-agent point mass and a 3-agent wolf pack:

``obs``      ``(capacity, num_envs, num_agents, obs_dim)``
``state``    ``(capacity, num_envs, state_dim)``
``action``   ``(capacity, num_envs, num_agents, act_dim)``
``reward``   ``(capacity, num_envs, num_agents)``
``done``     ``(capacity, num_envs)``          (terminal, no bootstrap)
``timeout``  ``(capacity, num_envs)``          (truncation, bootstrap allowed)
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

__all__ = ["ReplayBuffer", "RolloutBuffer"]


def _t(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class ReplayBuffer:
    """Circular, vectorised replay buffer for off-policy algorithms."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.state_dim = int(state_dim or obs_dim)
        self.device = torch.device(device)

        shape = (self.capacity, self.num_envs)
        self.obs = torch.zeros(*shape, self.num_agents, self.obs_dim, device=self.device)
        self.next_obs = torch.zeros_like(self.obs)
        self.state = torch.zeros(*shape, self.state_dim, device=self.device)
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(*shape, self.num_agents, self.act_dim, device=self.device)
        self.reward = torch.zeros(*shape, self.num_agents, device=self.device)
        self.done = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self.timeout = torch.zeros(*shape, dtype=torch.bool, device=self.device)

        #: sampled with probability proportional to ``priority`` when enabled
        #: (TD7's LAP).  ``priority`` stores a per-transition *exploration*
        #: weight; uniform sampling is the default for every other baseline.
        self.priority = torch.ones(*shape, device=self.device)
        self.use_priority = False

        self.ptr = 0
        self.size = 0
        self._total = 0

    # ------------------------------------------------------------------ #
    def add(
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
        self.obs[self.ptr] = _t(obs, self.device)
        self.state[self.ptr] = _t(state, self.device)
        self.action[self.ptr] = _t(action, self.device)
        self.reward[self.ptr] = _t(reward, self.device)
        self.next_obs[self.ptr] = _t(next_obs, self.device)
        self.next_state[self.ptr] = _t(next_state, self.device)
        self.done[self.ptr] = _t(done, self.device, torch.bool).reshape(self.num_envs)
        if timeout is None:
            self.timeout[self.ptr] = False
        else:
            self.timeout[self.ptr] = _t(timeout, self.device, torch.bool).reshape(self.num_envs)
        self.priority[self.ptr] = 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._total += self.num_envs

    # ------------------------------------------------------------------ #
    def set_priority(self, row: torch.Tensor, env: torch.Tensor, value: torch.Tensor) -> None:
        """Overwrite the sampling weight of the sampled transitions (TD7/LAP)."""
        with torch.no_grad():
            self.priority[row, env] = value.detach().to(self.device).reshape(-1).clamp_min(1e-6)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Number of stored *environment steps* (not rows)."""
        return self._total

    def ready(self, min_steps: int) -> bool:
        return self.size > 0 and self._total >= min_steps

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        return_index: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample ``batch_size`` transitions from the flattened ``(row, env)`` space."""
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        n = self.size * self.num_envs
        if self.use_priority:
            probs = self.priority[: self.size].reshape(-1).to(torch.float32)
            idx = torch.multinomial(probs, int(batch_size), replacement=True, generator=generator)
        else:
            idx = torch.randint(0, n, (batch_size,), device=self.device, generator=generator)
        row = idx // self.num_envs
        env = idx % self.num_envs
        batch = {
            "obs": self.obs[row, env],
            "state": self.state[row, env],
            "action": self.action[row, env],
            "reward": self.reward[row, env],
            "next_obs": self.next_obs[row, env],
            "next_state": self.next_state[row, env],
            # bootstrapping mask: 0 for true terminals, 1 otherwise
            "discount": (~self.done[row, env]).float(),
        }
        if return_index:
            batch["row"] = row
            batch["env"] = env
        return batch

    def reset(self) -> None:
        self.ptr = self.size = self._total = 0


class RolloutBuffer:
    """On-policy storage with generalised advantage estimation (per agent)."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: Optional[int] = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.steps = int(steps)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.state_dim = int(state_dim or obs_dim)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.device = torch.device(device)

        shape = (self.steps, self.num_envs)
        self.obs = torch.zeros(*shape, self.num_agents, self.obs_dim, device=self.device)
        self.state = torch.zeros(*shape, self.state_dim, device=self.device)
        self.action = torch.zeros(*shape, self.num_agents, self.act_dim, device=self.device)
        self.logp = torch.zeros(*shape, self.num_agents, device=self.device)
        self.value = torch.zeros(*shape, self.num_agents, device=self.device)
        self.reward = torch.zeros(*shape, self.num_agents, device=self.device)
        self.done = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self.timeout = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        # filled by compute_returns
        self.adv = torch.zeros(*shape, self.num_agents, device=self.device)
        self.ret = torch.zeros(*shape, self.num_agents, device=self.device)
        self.ptr = 0

    # ------------------------------------------------------------------ #
    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        logp: torch.Tensor,
        value: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: Optional[np.ndarray] = None,
    ) -> None:
        i = self.ptr
        self.obs[i] = _t(obs, self.device)
        self.state[i] = _t(state, self.device)
        self.action[i] = _t(action, self.device)
        self.logp[i] = logp.detach().to(self.device)
        self.value[i] = value.detach().to(self.device)
        self.reward[i] = _t(reward, self.device)
        self.done[i] = _t(done, self.device, torch.bool).reshape(self.num_envs)
        self.timeout[i] = False if timeout is None else _t(timeout, self.device, torch.bool)
        self.ptr += 1

    # ------------------------------------------------------------------ #
    def compute_returns(
        self,
        last_value: torch.Tensor,
        last_done: Optional[torch.Tensor] = None,
    ) -> None:
        """GAE-λ per agent.

        ``last_value`` is ``(num_envs, num_agents)``; the bootstrapping mask uses
        the *terminal* flag only, so time-limit truncations are bootstrapped.
        """
        last_value = last_value.detach().to(self.device)
        if last_done is None:
            last_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminal = self.done & ~self.timeout
        adv = torch.zeros(self.num_agents, device=self.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_value = last_value
                next_nonterm = (~last_done).float().unsqueeze(-1)
            else:
                next_value = self.value[t + 1]
                next_nonterm = (~terminal[t + 1]).float().unsqueeze(-1)
            not_done = (~terminal[t]).float().unsqueeze(-1)
            delta = self.reward[t] + self.gamma * next_value * next_nonterm - self.value[t]
            adv = delta + self.gamma * self.lam * not_done * adv
            self.adv[t] = adv
        self.ret[:] = self.adv + self.value

    # ------------------------------------------------------------------ #
    def flatten(self) -> Dict[str, torch.Tensor]:
        """``(steps*num_envs, ...)`` view used by the update loops."""
        return {
            "obs": self.obs.reshape(-1, self.num_agents, self.obs_dim),
            "state": self.state.reshape(-1, self.state_dim),
            "action": self.action.reshape(-1, self.num_agents, self.act_dim),
            "logp": self.logp.reshape(-1, self.num_agents),
            "value": self.value.reshape(-1, self.num_agents),
            "adv": self.adv.reshape(-1, self.num_agents),
            "ret": self.ret.reshape(-1, self.num_agents),
        }

    def minibatches(
        self,
        batch_size: int,
        num_epochs: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        flat = self.flatten()
        total = flat["obs"].shape[0]
        for _ in range(num_epochs):
            perm = torch.randperm(total, device=self.device, generator=generator)
            for start in range(0, total, batch_size):
                idx = perm[start: start + batch_size]
                if idx.numel() == 0:
                    continue
                yield {k: v[idx] for k, v in flat.items()}

    def reset(self) -> None:
        self.ptr = 0
