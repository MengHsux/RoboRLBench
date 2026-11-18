"""Soft Actor-Critic (Haarnoja et al., 2018).

Maximum-entropy off-policy actor-critic with a squashed Gaussian policy, twin
critics and automatic temperature tuning.  ``SAC`` is the parent class of four
other baselines in this benchmark (SimBa, CrossQ, TQC, DroQ); each of those
overrides one specific component and documents the difference in its module
docstring.

Reference: https://arxiv.org/abs/1801.01290
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OffPolicyAgent
from ..nets import SquashedGaussianActor, TwinCritic, soft_update

__all__ = ["SAC"]


class SAC(OffPolicyAgent):
    name = "sac"

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
        critic_lr: Optional[float] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        warmup: int = 2_000,
        grad_steps: int = 1,
        autotune: bool = True,
        init_alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau,
            warmup=warmup, grad_steps=grad_steps, **kwargs,
        )
        assert self.num_agents == 1, "SAC is a single-agent algorithm"
        self.hidden = tuple(hidden)
        self.autotune = bool(autotune)
        self.target_entropy = float(target_entropy if target_entropy is not None
                                    else -float(self.act_dim))

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr or lr
        )
        self.log_alpha = torch.tensor(
            [math.log(float(init_alpha))], requires_grad=True, device=self.device
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    # ------------------------------------------------------------------ #
    # Building blocks (overridden by derived baselines)
    # ------------------------------------------------------------------ #
    def build_actor(self) -> nn.Module:
        return SquashedGaussianActor(self.obs_dim, self.act_dim, self.hidden).to(self.device)

    def build_critic(self) -> nn.Module:
        return TwinCritic(self.state_dim + self.act_dim, self.hidden).to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b = obs.shape[0]
        a = self.actor.act(obs.reshape(b, -1), deterministic)
        return a.reshape(b, self.num_agents, self.act_dim)

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        batch = self.sample()
        batch = self.normalize_batch(batch, self.obs_rms, self.state_rms)
        metrics = self._update_critic(batch)
        metrics.update(self._update_actor(batch))
        return metrics

    def _q(self, critic: nn.Module, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return critic.q(state, action)

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.rsample(
                batch["next_obs"].reshape(b, -1)
            )
            q_next = self._q(self.critic_target, batch["next_state"], next_action)
            target = (
                batch["reward"].reshape(-1)
                + self.gamma * batch["discount"].reshape(-1)
                * (q_next - self.alpha.detach() * next_logp)
            )
        q1, q2 = self.critic(batch["state"], batch["action"].reshape(b, -1))
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()
        self._soft_update_targets()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action, logp, _ = self.actor.rsample(batch["obs"].reshape(b, -1))
        q = self._q(self.critic, batch["state"], action)
        actor_loss = (self.alpha.detach() * logp - q).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        out = {"actor_loss": float(actor_loss.detach()),
               "alpha": float(self.alpha.detach())}
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            out["alpha_loss"] = float(alpha_loss.detach())
        return out

    def _soft_update_targets(self) -> None:
        soft_update(self.critic_target, self.critic, self.tau)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic, "critic_target": self.critic_target}

    def extra_state(self) -> Dict:
        return {
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_extra_state(self, state: Dict) -> None:
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        for key, opt in (("actor_optim", self.actor_optim),
                         ("critic_optim", self.critic_optim),
                         ("alpha_optim", self.alpha_optim)):
            if key in state:
                opt.load_state_dict(state[key])
