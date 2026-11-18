"""The ten single-agent baselines (single-robot environments)."""

from __future__ import annotations

from .crossq import CrossQ
from .droq import DroQ
from .fasttd3 import FastTD3
from .ppo import PPO
from .redq import REDQ
from .sac import SAC
from .simba import SimBa
from .td3 import TD3
from .td7 import TD7
from .tqc import TQC

__all__ = [
    "PPO",
    "SAC",
    "TD3",
    "TQC",
    "REDQ",
    "DroQ",
    "CrossQ",
    "SimBa",
    "FastTD3",
    "TD7",
]
