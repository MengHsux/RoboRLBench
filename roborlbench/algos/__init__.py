"""Twenty deep-RL baselines for the RoboRLBench environments.

``single`` baselines run on every robot-single environment (``num_agents == 1``);
``multi`` baselines run on *every* environment, including the single-robot ones,
which is what makes the cross-simulator comparison possible.
"""

from __future__ import annotations

from .base import AgentBase, OffPolicyAgent, OnPolicyAgent
from .buffers import ReplayBuffer, RolloutBuffer
from .normalizer import RunningMeanStd
from .registry import (
    ALGOS,
    AlgoSpec,
    build_agent,
    get_algo_spec,
    list_algos,
    register_algo,
)

__all__ = [
    "AgentBase",
    "OnPolicyAgent",
    "OffPolicyAgent",
    "ReplayBuffer",
    "RolloutBuffer",
    "RunningMeanStd",
    "ALGOS",
    "AlgoSpec",
    "build_agent",
    "get_algo_spec",
    "list_algos",
    "register_algo",
]
