"""SimBa - Simplicity Bias for Scaling Up Parameters in Deep RL
(Lee et al., NeurIPS 2024).

SimBa keeps the SAC (or PPO) objective untouched and only changes the
*architecture*: the usual 2-layer ReLU MLP is replaced by

``Linear -> [LayerNorm -> SiLU -> Linear -> SiLU -> residual] * n``

i.e. the residual block that made vision transformers scalable.  The paper
shows that this single change lets a 4x wider network train reliably where the
MLP version diverges, and improves every DMC / MyoSuite / HumanoidBench task.

Here both the actor and the twin critics are built from
:class:`~roborlbench.algos.nets.SimbaMLP`, so the *only* difference to SAC is
the network topology.

Reference: https://arxiv.org/abs/2410.09754
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..nets import SimbaSquashedGaussianActor, SimbaTwinCritic
from .sac import SAC

__all__ = ["SimBa"]


class SimBa(SAC):
    name = "simba"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        **kwargs,
    ) -> None:
        # consumed by build_actor / build_critic inside SAC.__init__
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )

    # ------------------------------------------------------------------ #
    def build_actor(self) -> nn.Module:
        return SimbaSquashedGaussianActor(
            self.obs_dim, self.act_dim, self.hidden_dim, self.num_blocks
        ).to(self.device)

    def build_critic(self) -> nn.Module:
        return SimbaTwinCritic(
            self.state_dim + self.act_dim, self.hidden_dim, self.num_blocks
        ).to(self.device)
