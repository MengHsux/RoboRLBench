"""MASAC - Multi-Agent Soft Actor-Critic (Pu et al., 2021).

MASAC brings the maximum-entropy objective of SAC to the CTDE setting:

* each agent has a **squashed Gaussian** actor,
* the **centralised twin critic** ``Q_i(s, a_1..a_N)`` is trained with the
  entropy-augmented Bellman target
  ``y_i = r_i + gamma * (min Q'_i(s', a') - alpha * log pi_i(a_i'|o_i))``,
* the temperature ``alpha`` is tuned automatically.

Unlike HAPPO/HASAC the actors are updated **simultaneously** (all agents at
once), which makes MASAC the natural multi-agent counterpart of SAC.

Reference: https://arxiv.org/abs/2103.15019
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import SquashedGaussianActor, soft_update
from .base import MARLOffPolicy, joint_action
from .nets_marl import TwinCentralQ

__all__ = ["MASAC"]


class MASAC(MARLOffPolicy):
    name = "masac"

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
        self.hidden = tuple(hidden)
        self.autotune = bool(autotune)
        # one temperature for the whole team; target entropy is per agent
        self.target_entropy = float(
            target_entropy if target_entropy is not None else -float(act_dim)
        )

        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, self.hidden).to(self.device)
        self.critic = self.build_critic()
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.tensor(
            [math.log(float(init_alpha))], requires_grad=True, device=self.device
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return TwinCentralQ(
            self.state_dim, self.num_agents, self.act_dim, hidden=self.hidden
        ).to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _agent_act(self, flat_obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return self.actor.act(flat_obs, deterministic)

    @torch.no_grad()
    def _logp_all(self, obs: torch.Tensor) -> torch.Tensor:
        """Log-probability of actions sampled from the *current* policy, ``(B, A)``."""
        b, n = obs.shape[0], obs.shape[1]
        _, logp, _ = self.actor.rsample(obs.reshape(b * n, -1))
        return logp.reshape(b, n)

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        batch = self.sample()
        batch = self.normalize_batch(batch, self.obs_rms, self.state_rms)
        metrics = self._update_critic(batch)
        metrics.update(self._update_actor(batch))
        self._soft_update_targets()
        return metrics

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b, n = batch["state"].shape[0], self.num_agents
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.rsample(batch["next_obs"].reshape(b * n, -1))
            next_action = next_action.reshape(b, n, self.act_dim)
            next_logp = next_logp.reshape(b, n)
            q_next = self.critic_target.q(batch["next_state"], joint_action(next_action))
            target = batch["reward"] + self.gamma * batch["discount"].unsqueeze(-1) * (
                q_next - self.alpha.detach() * next_logp
            )

        q1, q2 = self.critic(batch["state"], joint_action(batch["action"]))
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b, n = batch["state"].shape[0], self.num_agents
        actor_loss = torch.zeros((), device=self.device)
        logps = []
        for i in range(n):
            own, logp_i, _ = self.actor.rsample(batch["obs"][:, i])
            joint = torch.cat(
                [batch["action"][:, :i], own.unsqueeze(1), batch["action"][:, i + 1:]], dim=1
            )
            q = self.critic.q(batch["state"], joint_action(joint))[:, i]
            actor_loss = actor_loss + (self.alpha.detach() * logp_i - q).mean()
            logps.append(logp_i)
        actor_loss = actor_loss / float(n)

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        out = {"actor_loss": float(actor_loss.detach()), "alpha": float(self.alpha.detach())}
        if self.autotune:
            logp_mean = torch.stack(logps, dim=-1).mean()
            alpha_loss = -(self.log_alpha * (logp_mean.detach() + self.target_entropy))
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
