"""The ten multi-agent baselines (multi-robot environments).

All of them also run on the single-robot environments (``num_agents == 1``),
which is what lets the same benchmark table cover every simulator.
"""

from __future__ import annotations

from .base import MARLOffPolicy, MARLOnPolicy
from .facmac import FACMAC
from .hasac import HASAC
from .happo import HAPPO
from .hatrpo import HATRPO
from .ippo import IPPO
from .maddpg import MADDPG
from .mappo import MAPPO
from .masac import MASAC
from .mat import MAT
from .matd3 import MATD3

__all__ = [
    "MARLOnPolicy",
    "MARLOffPolicy",
    "IPPO",
    "MAPPO",
    "MADDPG",
    "MATD3",
    "MASAC",
    "FACMAC",
    "HASAC",
    "HAPPO",
    "HATRPO",
    "MAT",
]
