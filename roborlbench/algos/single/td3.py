"""Twin Delayed DDPG (Fujimoto et al., 2018).

Deterministic off-policy actor-critic with clipped double-Q, target policy
smoothing and delayed policy updates.  ``TD3`` is the parent of TD7, DroQ,
REDQ and FastTD3 in this benchmark.

Reference: https://arxiv.org/abs/1802.09477
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OffPolicyAgent
from ..nets import DeterministicActor, TwinCritic, soft_update

__all__ = ["TD3"]


class TD3(OffPolicyAgent):
    name = "td3"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        hidden=(256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        warmup: int = 2_000,
        grad_steps: int = 1,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau,
            warmup=warmup, grad_steps=grad_steps, **kwargs,
        )
        assert self.num_agents == 1, "TD3 is a single-agent algorithm"
        self.hidden = tuple(hidden)
        self.policy_delay = int(policy_delay)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self._update_count = 0

        self.actor = self.build_actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = self.build_critic()
        self.critic_target = copy.deepcopy(self.critic)
        for net in (self.actor_target, self.critic_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.lr = float(lr)

    # ------------------------------------------------------------------ #
    def build_actor(self) -> nn.Module:
        return DeterministicActor(self.obs_dim, self.act_dim, self.hidden).to(self.device)

    def build_critic(self) -> nn.Module:
        return TwinCritic(self.state_dim + self.act_dim, self.hidden).to(self.device)

    # ------------------------------------------------------------------ #
    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b = obs.shape[0]
        a = self.actor(obs.reshape(b, -1))
        return a.reshape(b, self.num_agents, self.act_dim)

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        batch = self.sample()
        batch = self.normalize_batch(batch, self.obs_rms, self.state_rms)
        metrics = self._update_critic(batch)
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            metrics.update(self._update_actor(batch))
            self._soft_update_targets()
        return metrics

    # ------------------------------------------------------------------ #
    def _target_action(self, next_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            noise = (torch.randn_like(self.actor_target(next_obs)) * self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            return (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

    def _target_q(self, next_state: torch.Tensor, next_action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q1, q2 = self.critic_target(next_state, next_action)
            return torch.min(q1, q2)

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        next_action = self._target_action(batch["next_obs"].reshape(b, -1))
        q_target = self._target_q(batch["next_state"], next_action)
        target = batch["reward"].reshape(-1) + self.gamma * batch["discount"].reshape(-1) * q_target

        q1, q2 = self.critic(batch["state"], batch["action"].reshape(b, -1))
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action = self.actor(batch["obs"].reshape(b, -1))
        q1, _ = self.critic(batch["state"], action)
        actor_loss = -q1.mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()
        return {"actor_loss": float(actor_loss.detach())}

    def _soft_update_targets(self) -> None:
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
        }

    def extra_state(self) -> Dict:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_count": self._update_count,
        }

    def load_extra_state(self, state: Dict) -> None:
        if "actor_optim" in state:
            self.actor_optim.load_state_dict(state["actor_optim"])
        if "critic_optim" in state:
            self.critic_optim.load_state_dict(state["critic_optim"])
        self._update_count = int(state.get("update_count", 0))
