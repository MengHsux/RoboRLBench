"""``roborlbench doctor`` - diagnose the installation.

Prints the Python/Torch versions, where each upstream simulator repository was
found, which backends import cleanly, and finally instantiates (and briefly
probes) every environment that is available on this machine.
"""

from __future__ import annotations

import argparse
import importlib
import platform
from typing import Dict, List, Optional

from ..envs import ENVS, make_env, probe_env
from ..envs._simpaths import describe

__all__ = ["add_arguments", "run"]

# backend -> (adapter module, third-party modules it needs at run time).
# Checking the adapter alone is not enough: the adapters import their simulator
# lazily, so they always import cleanly on a CPU-only machine.
_BACKENDS = {
    "analytic": None,
    "mujoco": ("roborlbench.envs.mujoco_backend", ["mujoco"]),
    "isaacgym": ("roborlbench.envs.isaacgym_backend", ["isaacgym.gymapi"]),
    "isaaclab": ("roborlbench.envs.isaaclab_backend", ["isaaclab", "gymnasium"]),
}


def _probe_backend(backend: str) -> Optional[str]:
    """Return ``None`` when *backend* can really run here, else a reason."""
    adapter, runtime = _BACKENDS[backend]
    try:
        importlib.import_module(adapter)
    except Exception as exc:                           # noqa: BLE001 - report only
        return f"adapter import failed: {str(exc).splitlines()[0]}"

    modules = []
    for name in runtime:
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:                       # noqa: BLE001 - report only
            return f"{name} not installed: {str(exc).splitlines()[0]}"

    if backend == "isaacgym":
        # ``import isaacgym`` also succeeds against the CPU-only compatibility
        # shim that ships with legged_gym; only ``acquire_gym()`` proves that the
        # native simulator is present.
        try:
            modules[0].acquire_gym()
        except Exception as exc:                       # noqa: BLE001 - report only
            return str(exc).splitlines()[0]
    return None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=3,
                        help="random steps used to probe each available environment")
    parser.add_argument("--analytic-only", action="store_true",
                        help="skip every environment that needs a simulator")


def run(args: argparse.Namespace) -> int:
    import torch

    print(f"python      : {platform.python_version()} ({platform.system()})")
    print(f"torch       : {torch.__version__} (cuda available: {torch.cuda.is_available()})")

    print("\nsimulator repositories")
    for key, path in describe().items():
        print(f"  {key:20s} {path or '<not found>'}")

    print("\nbackends")
    status: Dict[str, bool] = {}
    for backend in _BACKENDS:
        if _BACKENDS[backend] is None:
            status[backend] = True
            print(f"  {backend:20s} built-in (pure NumPy)")
            continue
        reason = _probe_backend(backend)
        status[backend] = reason is None
        print(f"  {backend:20s} {'ready' if reason is None else 'unavailable: ' + reason}")

    print("\nenvironments")
    available: List[str] = []
    for name, spec in sorted(ENVS.items()):
        if args.analytic_only and spec.backend != "analytic":
            continue
        if not status.get(spec.backend, False):
            print(f"  {name:22s} {spec.backend:9s} skipped (backend unavailable)")
            continue
        try:
            env = make_env(name, num_envs=1)
            info = probe_env(env, steps=args.steps)
        except Exception as exc:                       # noqa: BLE001 - report only
            print(f"  {name:22s} {spec.backend:9s} FAILED: {str(exc).splitlines()[0]}")
            continue
        available.append(name)
        print(
            f"  {name:22s} {spec.backend:9s} ok  "
            f"obs={info['obs_dim']} state={info['state_dim']} "
            f"act={info['act_dim']} agents={info['num_agents']}"
        )

    print(f"\n{len(available)} environment(s) ready: {available}")
    return 0
