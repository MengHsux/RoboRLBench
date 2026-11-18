"""HASAC - Heterogeneous-Agent Soft Actor-Critic (Zhong et al., ICLR 2023).

HASAC is the maximum-entropy member of the heterogeneous-agent trust-region
family.  Its insight is that updating all actors *simultaneously* - as MASAC
does - ignores the fact that the centralised critic depends on every agent's
action.  HASAC therefore performs a **sequential** policy improvement:

1. draw a random permutation of the agents,
2. for each agent ``i_m`` in that order, build the joint action with the
   freshly sampled ``a_{i_m}`` and the buffer actions for the others, weighted
   by the **joint importance ratio**

   ``M = prod_{k<m} pi_{i_k}^{new}(a_{i_k}) / pi_{i_k}^{old}(a_{i_k})``,

3. take one gradient step for that agent, then fold its ratio into ``M``.

The frozen ``pi_old`` snapshot is taken at the start of every update, and the
ratio is clipped for numerical stability (documented in docs/ALGORITHMS.md).
The critic and the temperature are MASAC's.

Reference: https://arxiv.org/abs/2203.11487
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..nets import logprob_from_normal
from .base import joint_action
from .masac import MASAC

__all__ = ["HASAC"]


class HASAC(MASAC):
    name = "hasac"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        ratio_clip: float = 5.0,
        **kwargs,
    ) -> None:
        self.ratio_clip = float(ratio_clip)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )

    # ------------------------------------------------------------------ #
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs_all = batch["obs"]                       # (B, A, obs_dim)
        b, n = obs_all.shape[0], obs_all.shape[1]

        # frozen behaviour policy for the joint importance ratio
        old_actor = copy.deepcopy(self.actor)
        for p in old_actor.parameters():
            p.requires_grad_(False)

        order = torch.randperm(n, device=self.device).tolist()
        joint_ratio = torch.ones(b, device=self.device)
        total_loss = 0.0
        logps = []
        for i in order:
            own, logp_i, _ = self.actor.rsample(obs_all[:, i])
            with torch.no_grad():
                dist_old, _ = old_actor.distribution(obs_all[:, i])
                logp_old = logprob_from_normal(
                    dist_old.mean, dist_old.stddev, own.detach(), squashed=True
                )
                ratio = torch.exp(logp_i.detach() - logp_old).clamp(max=self.ratio_clip)

            joint = torch.cat(
                [batch["action"][:, :i], own.unsqueeze(1), batch["action"][:, i + 1:]], dim=1
            )
            q = self.critic.q(batch["state"], joint_action(joint))[:, i]
            loss = -(joint_ratio * (q - self.alpha.detach() * logp_i)).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optim.step()

            total_loss += float(loss.detach())
            logps.append(logp_i.detach())
            joint_ratio = joint_ratio * ratio

        out = {
            "actor_loss": total_loss / float(max(n, 1)),
            "alpha": float(self.alpha.detach()),
        }
        if self.autotune:
            logp_mean = torch.stack(logps, dim=-1).mean()
            alpha_loss = -(self.log_alpha * (logp_mean + self.target_entropy))
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            out["alpha_loss"] = float(alpha_loss.detach())
        return out
