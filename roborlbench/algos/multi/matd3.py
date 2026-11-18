"""MATD3 - Multi-Agent Twin Delayed DDPG
(Ackermann et al., AAMAS 2019).

MATD3 applies the three TD3 fixes to MADDPG, which is what removes the
overestimation that makes plain MADDPG brittle:

* **clipped double-Q** - the centralised critic is doubled and the Bellman
  target takes the minimum,
* **target policy smoothing** - Gaussian noise is added to the target joint
  action and clipped,
* **delayed policy updates** - the actors are updated every ``policy_delay``
  critic updates.

Only the critic, the target and the update schedule change, so this is a small
subclass of :class:`~roborlbench.algos.multi.maddpg.MADDPG`.

Reference: https://arxiv.org/abs/1910.01465
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import joint_action
from .maddpg import MADDPG
from .nets_marl import TwinCentralQ

__all__ = ["MATD3"]


class MATD3(MADDPG):
    name = "matd3"
    policy_delay = 2

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
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        **kwargs,
    ) -> None:
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, **kwargs,
        )
        self.policy_delay = int(policy_delay)

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return TwinCentralQ(
            self.state_dim, self.num_agents, self.act_dim, hidden=self.hidden
        ).to(self.device)

    def _target_actions(self, next_obs: torch.Tensor) -> torch.Tensor:
        b = next_obs.shape[0]
        flat = next_obs.reshape(b * self.num_agents, -1)
        action = self.actor_target(flat).reshape(b, self.num_agents, self.act_dim)
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        return (action + noise).clamp(-1.0, 1.0)

    def _target_q(self, next_state: torch.Tensor, next_joint: torch.Tensor) -> torch.Tensor:
        return self.critic_target.q(next_state, next_joint)

    def _q(self, state: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        return self.critic.q(state, joint)
