"""Convenience facade over the environment and algorithm registries.

``roborlbench.ENVS`` / ``roborlbench.ALGOS`` are the two dictionaries the CLI,
the tests and the docs enumerate.  Importing this module also triggers the
environment registrations (they live in :mod:`roborlbench.envs`).
"""

from __future__ import annotations

from .algos import (
    ALGOS,
    AlgoSpec,
    build_agent,
    get_algo_spec,
    list_algos,
    register_algo,
)
from .envs import (
    ENVS,
    BenchEnv,
    EnvSpec,
    get_env_spec,
    list_envs,
    make_env,
    probe_env,
    register_env,
)

__all__ = [
    "ALGOS",
    "AlgoSpec",
    "build_agent",
    "get_algo_spec",
    "list_algos",
    "register_algo",
    "ENVS",
    "BenchEnv",
    "EnvSpec",
    "get_env_spec",
    "list_envs",
    "make_env",
    "probe_env",
    "register_env",
]
