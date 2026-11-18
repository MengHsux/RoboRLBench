"""DroQ - Dropout Q-Functions for Doubly Efficient RL (Hiraoka et al., 2021).

DroQ produces the largest gains of any "dropout in the critic" recipe with a
three-line change to SAC:

* **LayerNorm** on every hidden layer of both critics,
* a small **dropout** (0.01) on every hidden layer, and
* a **high update-to-data ratio** (UTD = 20 in the paper).

The dropout makes the ensemble behave like a much larger implicit ensemble for
free, which is what stabilises the high UTD.  ``grad_steps`` defaults to 4 here;
it counts the gradient steps performed after every vectorised environment step,
so it equals UTD while ``num_envs == 1`` and has to be scaled with ``num_envs``
otherwise (see docs/ALGORITHMS.md).

Reference: https://arxiv.org/abs/2110.02034
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..nets import TwinCritic
from .sac import SAC

__all__ = ["DroQ"]


class DroQ(SAC):
    name = "droq"

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
        dropout: float = 0.01,
        grad_steps: int = 4,
        **kwargs,
    ) -> None:
        self.dropout = float(dropout)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, grad_steps=grad_steps, **kwargs,
        )

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return TwinCritic(
            self.state_dim + self.act_dim,
            self.hidden,
            "relu",
            use_layer_norm=True,
            dropout=self.dropout,
        ).to(self.device)
