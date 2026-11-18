"""IPPO - Independent Proximal Policy Optimisation (de Witt et al., 2020).

The simplest multi-agent on-policy baseline: every agent runs its own PPO with
its own critic ``V_i(o_i)`` and no information sharing.  It is the control
condition that tells you whether centralised training actually helps on your
task.

Implemented by setting ``centralized = False`` on the shared clustered-PPO
machinery of :class:`~roborlbench.algos.multi.base.MARLOnPolicy`.

Reference: https://arxiv.org/abs/2011.09533
"""

from __future__ import annotations

from typing import Optional

from .base import MARLOnPolicy

__all__ = ["IPPO"]


class IPPO(MARLOnPolicy):
    name = "ippo"
    centralized = False

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )
