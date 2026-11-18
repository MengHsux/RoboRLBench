"""HAPPO - Heterogeneous-Agent Proximal Policy Optimisation
(Kuba et al., ICLR 2022).

HAPPO is PPO with a **sequential, trust-region-equivalent** multi-agent update.
Instead of updating all actors simultaneously (MAPPO), the agents are updated
one at a time in a random order and each update is weighted by the joint
importance ratio ``M_i`` of the agents that were already updated.  This is what
guarantees monotonic improvement for the *joint* policy even when the agents are
heterogeneous.

The single-agent trust region is PPO's clipped surrogate; see
:class:`~roborlbench.algos.multi.hatrpo.HATRPO` for the KL-constrained variant.

Reference: https://arxiv.org/abs/2110.02793
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .sequential import SequentialOnPolicy

__all__ = ["HAPPO"]


class HAPPO(SequentialOnPolicy):
    name = "happo"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        clip_range: float = 0.2,
        num_critic_epochs: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )
        self.clip_range = float(clip_range)
        self.num_critic_epochs = int(num_critic_epochs)

    # ------------------------------------------------------------------ #
    def _agent_step(self, data: Dict[str, torch.Tensor], agent: int,
                    joint_ratio: torch.Tensor) -> float:
        obs_i = data["obs"][:, agent]
        act_i = data["action"][:, agent]
        adv_i = data["adv"][:, agent]

        dist = self.actor.distribution(obs_i)
        logp = dist.log_prob(act_i).sum(-1)
        ratio = torch.exp(logp - data["logp"][:, agent])

        surr1 = joint_ratio * ratio * adv_i
        surr2 = joint_ratio * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        ) * adv_i
        loss = -torch.min(surr1, surr2).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        return float(loss.detach())
