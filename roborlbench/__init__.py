"""RoboRLBench: a reproducible deep-RL benchmark for embodied robots.

The package is intentionally split into four layers:

``roborlbench.envs``
    A single vectorised, multi-agent-aware environment API (:class:`BenchEnv`)
    plus adapters for MuJoCo, Isaac Gym and Isaac Lab backends.  Simulator
    imports are lazy so this package is importable on a CPU-only machine.

``roborlbench.algos``
    Twenty self-contained deep-RL baselines (10 single-agent, 10 multi-agent)
    that only depend on ``torch`` + ``numpy``.

``roborlbench.runners``
    Training loops that wire an algorithm to an environment, handle logging,
    evaluation and checkpointing.

``roborlbench.cli``
    ``roborlbench train/eval`` command line entry points.
"""

from .version import __version__
from .envs.base import BenchEnv, EnvSpec
from .registry import ALGOS, ENVS, list_algos, list_envs

__all__ = [
    "__version__",
    "BenchEnv",
    "EnvSpec",
    "ALGOS",
    "ENVS",
    "list_algos",
    "list_envs",
]
