"""MAPPO - Multi-Agent PPO (Yu et al., NeurIPS 2021).

MAPPO is the reference on-policy MARL baseline: a shared Gaussian policy per
agent plus a **centralised value function** ``V(s)`` that sees the global state.
The paper's main finding is that this simple recipe, with the right
implementation details (value normalisation, mini-batches, clipped value loss),
matches or beats far more complicated off-policy methods on cooperative
benchmarks.

Here the critic outputs one value per agent, which supports both shared and
individual rewards.  The clipped surrogate, advantage normalisation and
mini-batch schedule are shared with the PPO baseline.

Reference: https://arxiv.org/abs/2103.01955
"""

from __future__ import annotations

from typing import Optional

from .base import MARLOnPolicy

__all__ = ["MAPPO"]


class MAPPO(MARLOnPolicy):
    name = "mappo"
    centralized = True

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        value_clip: float = 0.2,
        **kwargs,
    ) -> None:
        # MAPPO's clipped value loss is one of the details the paper insists on
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            value_clip=value_clip, **kwargs,
        )
