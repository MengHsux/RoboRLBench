"""FACMAC - Factored Multi-Agent Centralised Policy Gradients
(Peng et al., ICLR 2021).

FACMAC's contribution is the **factored critic**: instead of one network that
sees the whole joint action, it learns
``Q_i(s, a_i, phi_i(a_{-i}))`` where ``phi_i`` is a learned embedding of the
other agents' actions and the mixing is *monotonic* (a weighted sum).  This
keeps the centralised information while making the joint action space
factorise, which improves scaling to many agents.

Everything else - deterministic actors, twin critics, target policy smoothing,
delayed actor updates - is MATD3's, hence the subclass.  The one-step joint
gradient of the original paper is replaced by the standard per-agent policy
gradient for comparability with the other deterministic baselines.

Reference: https://arxiv.org/abs/2006.14171
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .matd3 import MATD3
from .nets_marl import FactoredTwinQ

__all__ = ["FACMAC"]


class FACMAC(MATD3):
    name = "facmac"

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
        embed_dim: int = 32,
        **kwargs,
    ) -> None:
        # consumed by build_critic inside MADDPG.__init__
        self.embed_dim = int(embed_dim)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, **kwargs,
        )

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return FactoredTwinQ(
            self.state_dim,
            self.num_agents,
            self.act_dim,
            embed_dim=self.embed_dim,
            hidden=self.hidden,
        ).to(self.device)
