"""CrossQ - Batch Normalisation in Deep RL, and why it matters for sample
efficiency (Bhatt et al., ICLR 2024).

CrossQ is a deliberately minimal SAC variant that reaches the sample efficiency
of REDQ / DroQ *without* any target network and *without* a high UTD:

* **BatchRenorm** replaces LayerNorm inside the critic.  This removes the
  distribution-shift problem that normally forces target networks.
* The **current** and **next** transitions are passed through the critic in a
  *single* forward pass (concatenated along the batch axis), so both branches
  share the same batch statistics.
* No target networks are used at all, which halves the memory and removes the
  slow target-update hyper-parameter.

Consequently :meth:`update` is overridden: there is no
``_soft_update_targets`` step and :meth:`modules` only exposes the online nets.

Reference: https://openreview.net/forum?id=2VXyy9mIyU
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import BatchRenorm1d
from .sac import SAC

__all__ = ["CrossQ", "CrossQTwinCritic"]


class _RenormCritic(nn.Module):
    """``[BatchRenorm -> Linear -> ReLU] * n -> Linear -> BatchRenorm``."""

    def __init__(self, in_dim: int, hidden) -> None:
        super().__init__()
        self.norm_in = BatchRenorm1d(in_dim)
        layers = []
        last = in_dim
        for width in hidden:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.ReLU())
            last = int(width)
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(last, 1)
        self.norm_out = BatchRenorm1d(1, momentum=0.01)
        nn.init.orthogonal_(self.out.weight, 1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(torch.cat([s, a], dim=-1))
        return self.norm_out(self.out(self.trunk(x))).squeeze(-1)


class CrossQTwinCritic(nn.Module):
    """Twin batch-renormalised critics with the :class:`TwinCritic` interface."""

    def __init__(self, in_dim: int, hidden) -> None:
        super().__init__()
        self.q1 = _RenormCritic(in_dim, hidden)
        self.q2 = _RenormCritic(in_dim, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        return self.q1(s, a), self.q2(s, a)

    def q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class CrossQ(SAC):
    name = "crossq"

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
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, **kwargs,
        )
        # CrossQ has no target network; drop the copy SAC created.
        self.critic_target = None

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return CrossQTwinCritic(self.state_dim + self.act_dim, self.hidden).to(self.device)

    def modules(self) -> Dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        batch = self.sample()
        batch = self.normalize_batch(batch, self.obs_rms, self.state_rms)
        metrics = self._update_critic(batch)
        metrics.update(self._update_actor(batch))
        return metrics

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.rsample(batch["next_obs"].reshape(b, -1))

        # one joint forward pass keeps the BatchRenorm statistics consistent
        # between the current and the next transition (the key CrossQ trick)
        states = torch.cat([batch["state"], batch["next_state"]], dim=0)
        actions = torch.cat([batch["action"].reshape(b, -1), next_action], dim=0)
        q1_all, q2_all = self.critic(states, actions)
        q1, q1_next = q1_all[:b], q1_all[b:]
        q2, q2_next = q2_all[:b], q2_all[b:]

        with torch.no_grad():
            q_next = torch.min(q1_next, q2_next)
            target = batch["reward"].reshape(-1) + self.gamma * batch["discount"].reshape(-1) * (
                q_next - self.alpha.detach() * next_logp
            )
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action, logp, _ = self.actor.rsample(batch["obs"].reshape(b, -1))
        q = self.critic.q(batch["state"], action)
        actor_loss = (self.alpha.detach() * logp - q).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        out = {"actor_loss": float(actor_loss.detach()), "alpha": float(self.alpha.detach())}
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            out["alpha_loss"] = float(alpha_loss.detach())
        return out

    def _soft_update_targets(self) -> None:   # no target networks in CrossQ
        return None
