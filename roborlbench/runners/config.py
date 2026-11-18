"""Run configuration: dataclass + YAML loading."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = ["TrainConfig", "load_config"]


@dataclass
class TrainConfig:
    """Everything needed to launch one training run."""

    env: str = "point-reach"
    algo: str = "sac"

    #: number of simulator copies stepped in parallel
    num_envs: int = 8
    #: total environment steps (transitions) to collect
    total_steps: int = 200_000
    seed: int = 0
    device: str = "cpu"

    #: evaluation schedule (in environment steps); 0 disables evaluation
    eval_interval: int = 20_000
    eval_episodes: int = 5
    #: checkpoint schedule (in environment steps); 0 disables saving
    save_interval: int = 100_000

    log_dir: str = "runs"
    run_name: str = ""
    tensorboard: bool = False

    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)

    def resolved_run_name(self) -> str:
        return self.run_name or f"{self.algo}-{self.env}-seed{self.seed}"

    def run_dir(self) -> str:
        return os.path.abspath(os.path.join(self.log_dir, self.resolved_run_name()))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: Optional[str] = None, **overrides: Any) -> TrainConfig:
    """Build a :class:`TrainConfig` from an optional YAML file plus overrides."""
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".json"):
                data = json.load(handle)
            else:
                try:
                    import yaml
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise ImportError(
                        "reading YAML configs requires pyyaml: pip install pyyaml"
                    ) from exc
                data = yaml.safe_load(handle) or {}
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise TypeError(f"unknown config key(s) {unknown}; accepted: {sorted(known)}")
    return TrainConfig(**data)
