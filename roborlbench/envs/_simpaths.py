"""Locate the upstream simulator repositories that the adapters wrap.

RoboRLBench deliberately does **not** vendor the four simulator projects: they
carry their own licences and their own (mutually incompatible) dependency
stacks.  Instead the adapters import them by name and this module makes sure the
right directories are on ``sys.path``.

Resolution order (first hit wins), for a repository such as
``legged_hunt-main``:

1. explicit environment variable, e.g. ``ROBORLBENCH_LEGGED_HUNT=/code/legged_hunt-main``
2. ``simpaths.yaml`` in the current working directory or ``~/.roborlbench/simpaths.yaml``
   with a mapping ``legged_hunt: /code/legged_hunt-main``
3. a sibling of this project, searched upwards from the package directory
   (so the default layout ``workspace/{RoboRLBench,legged_hunt-main,...}`` works
   out of the box)

Example ``simpaths.yaml``::

    legged_hunt:        /data/code/legged_hunt-main
    legged_hunt_single: /data/code/legged_hunt-main-single
    go2_rl_lab:         /data/code/Go2_control/go2_rl_lab-lab3.0
    openso101:          /data/code/OpenSO-101-main
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["repo_path", "ensure_importable", "import_module", "describe"]


_REPOS: Dict[str, dict] = {
    "legged_hunt": {
        "env": "ROBORLBENCH_LEGGED_HUNT",
        "names": ["legged_hunt-main", "legged_hunt", "legged_hunter"],
        "subpaths": ["."],
    },
    "legged_hunt_single": {
        "env": "ROBORLBENCH_LEGGED_HUNT_SINGLE",
        "names": ["legged_hunt-main-single", "legged_hunt_single", "legged_hunter-single"],
        "subpaths": ["."],
    },
    "go2_rl_lab": {
        "env": "ROBORLBENCH_GO2_RL_LAB",
        "names": ["Go2_control/go2_rl_lab-lab3.0", "go2_rl_lab", "go2_rl_lab-lab3.0"],
        "subpaths": ["source/go2_rl_lab", "."],
    },
    "openso101": {
        "env": "ROBORLBENCH_OPENSO101",
        "names": ["OpenSO-101-main", "OpenSO-101", "openso101"],
        "subpaths": ["src", "."],
    },
}


def _config_candidates() -> List[Path]:
    here = Path(__file__).resolve()
    roots = [Path.cwd(), Path.home() / ".roborlbench"]
    roots += list(here.parents)[:5]
    return [r / "simpaths.yaml" for r in roots]


def _from_config() -> Dict[str, str]:
    try:
        import yaml
    except ImportError:                 # pyyaml is optional: env vars still work
        return {}
    for path in _config_candidates():
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    return {}


_CONFIG: Optional[Dict[str, str]] = None


def _config() -> Dict[str, str]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _from_config()
    return _CONFIG


def repo_path(key: str) -> Optional[Path]:
    """Return the root directory of an upstream repository, or ``None``."""
    if key not in _REPOS:
        raise KeyError(f"unknown simulator key {key!r}; known: {sorted(_REPOS)}")
    spec = _REPOS[key]

    env_value = os.environ.get(spec["env"])
    if env_value:
        p = Path(env_value).expanduser().resolve()
        if p.is_dir():
            return p

    cfg_value = _config().get(key)
    if cfg_value:
        p = Path(cfg_value).expanduser().resolve()
        if p.is_dir():
            return p

    roots: List[Path] = [Path.cwd()]
    roots += list(Path(__file__).resolve().parents)[:5]
    roots.append(Path.home())
    for root in roots:
        for name in spec["names"]:
            candidate = (root / name)
            if candidate.is_dir():
                return candidate.resolve()
    return None


def ensure_importable(key: str) -> Path:
    """Put an upstream repository's importable roots on ``sys.path``."""
    root = repo_path(key)
    if root is None:
        spec = _REPOS[key]
        raise ImportError(
            f"RoboRLBench could not find the {key!r} repository.\n"
            f"  * set {spec['env']}=/path/to/{spec['names'][0]}, or\n"
            f"  * add `{key}: /path/to/{spec['names'][0]}` to simpaths.yaml, or\n"
            f"  * clone it next to the RoboRLBench directory\n"
            f"See docs/ENVIRONMENTS.md for details."
        )
    for sub in _REPOS[key]["subpaths"]:
        directory = (root / sub).resolve()
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return root


def import_module(key: str, module: str):
    """``ensure_importable(key)`` followed by ``importlib.import_module``."""
    ensure_importable(key)
    return importlib.import_module(module)


def describe() -> Dict[str, Optional[str]]:
    """Human readable status of every known repository (used by ``doctor``)."""
    return {
        key: (str(p) if (p := repo_path(key)) else None) for key in _REPOS
    }
