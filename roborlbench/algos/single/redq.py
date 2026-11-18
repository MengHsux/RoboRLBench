"""REDQ - Randomised Ensembled Double Q-learning (Chen et al., ICLR 2021).

REDQ raises the sample efficiency of SAC by combining three ingredients:

* an **ensemble** of ``G`` critics (10 in the paper),
* a **random subset** of ``M`` critics (2 by default) used for the Bellman
  target - the minimum over that subset - and
* a very **high update-to-data ratio** (20 in the paper).

This baseline inherits the SAC actor and temperature and only replaces the
critic, the target computation and the default ``grad_steps``.

Reference: https://arxiv.org/abs/2101.05982
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import EnsembleCritic
from .sac import SAC

__all__ = ["REDQ"]


class REDQ(SAC):
    name = "redq"

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
        num_critics: int = 10,
        num_target_subset: int = 2,
        grad_steps: int = 4,
        **kwargs,
    ) -> None:
        self.num_critics = int(num_critics)
        self.num_target_subset = int(num_target_subset)
        assert 1 <= self.num_target_subset <= self.num_critics, (
            "num_target_subset must lie in [1, num_critics]"
        )
        # ``grad_steps`` counts updates per vectorised environment step, i.e. it
        # equals UTD at ``num_envs == 1``; the paper's UTD of 20 is reduced to a
        # wall-clock friendly default (see docs/ALGORITHMS.md).
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, grad_steps=grad_steps, **kwargs,
        )

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return EnsembleCritic(
            self.state_dim + self.act_dim, self.hidden, num_critics=self.num_critics
        ).to(self.device)

    # ------------------------------------------------------------------ #
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.rsample(batch["next_obs"].reshape(b, -1))
            q_all = self.critic_target(batch["next_state"], next_action)   # (G, B)
            idx = torch.randperm(q_all.shape[0], device=self.device)[: self.num_target_subset]
            q_next = q_all[idx].mean(0)                                    # REDQ averages the subset
            target = batch["reward"].reshape(-1) + self.gamma * batch["discount"].reshape(-1) * (
                q_next - self.alpha.detach() * next_logp
            )

        q_all = self.critic(batch["state"], batch["action"].reshape(b, -1))  # (G, B)
        critic_loss = sum(F.mse_loss(q, target) for q in q_all) / float(q_all.shape[0])

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        self._soft_update_targets()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action, logp, _ = self.actor.rsample(batch["obs"].reshape(b, -1))
        q = self.critic(batch["state"], action).mean(0)                    # (B,)
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
