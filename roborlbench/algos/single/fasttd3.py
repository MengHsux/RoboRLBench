"""FastTD3 - a strong, fast and simple baseline for continuous control
(Bhattacharyya et al., 2025).

FastTD3 is TD3 with the settings that make it competitive with model-based
methods on humanoid control at a fraction of the compute:

* an **ensemble** of ``N`` critics whose Bellman target averages a random
  subset of ``M`` members (a milder, better calibrated version of REDQ's min),
* **LayerNorm** in the actor and in every critic hidden layer,
* a **large batch** (1024) and a **high update-to-data ratio**.

The actor, target policy smoothing and delayed policy updates are exactly TD3's,
which is why this baseline is a thin subclass of
:class:`~roborlbench.algos.single.td3.TD3`.

Reference: https://arxiv.org/abs/2502.06718
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import DeterministicActor, EnsembleCritic
from .td3 import TD3

__all__ = ["FastTD3"]


class FastTD3(TD3):
    name = "fasttd3"

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
        batch_size: int = 1024,
        grad_steps: int = 8,
        **kwargs,
    ) -> None:
        self.num_critics = int(num_critics)
        self.num_target_subset = int(num_target_subset)
        assert 1 <= self.num_target_subset <= self.num_critics
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, batch_size=batch_size, grad_steps=grad_steps, **kwargs,
        )

    # ------------------------------------------------------------------ #
    def build_actor(self) -> nn.Module:
        return DeterministicActor(
            self.obs_dim, self.act_dim, self.hidden, "relu", use_layer_norm=True
        ).to(self.device)

    def build_critic(self) -> nn.Module:
        return EnsembleCritic(
            self.state_dim + self.act_dim,
            self.hidden,
            num_critics=self.num_critics,
            use_layer_norm=True,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    def _target_q(self, next_state: torch.Tensor, next_action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_all = self.critic_target(next_state, next_action)            # (N, B)
            idx = torch.randperm(q_all.shape[0], device=self.device)[: self.num_target_subset]
            return q_all[idx].mean(0)

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        next_action = self._target_action(batch["next_obs"].reshape(b, -1))
        q_target = self._target_q(batch["next_state"], next_action)
        target = batch["reward"].reshape(-1) + self.gamma * batch["discount"].reshape(-1) * q_target

        q_all = self.critic(batch["state"], batch["action"].reshape(b, -1))  # (N, B)
        critic_loss = sum(F.mse_loss(q, target) for q in q_all) / float(q_all.shape[0])

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action = self.actor(batch["obs"].reshape(b, -1))
        q = self.critic(batch["state"], action).mean(0)
        actor_loss = -q.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()
        return {"actor_loss": float(actor_loss.detach())}
