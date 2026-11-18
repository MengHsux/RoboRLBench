"""``roborlbench train`` - launch one training run."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from ..runners import Trainer, load_config

__all__ = ["add_arguments", "run"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="YAML/JSON run configuration (CLI flags override it)")
    parser.add_argument("--env", default=None,
                        help="environment name, e.g. go2-flat / so101-lift / multi-hunt-2d")
    parser.add_argument("--algo", default=None,
                        help="baseline name, e.g. sac / crossq / mappo / hasac")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="number of parallel simulator copies")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="environment transitions to collect")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="evaluate every N env steps (0 disables)")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None,
                        help="checkpoint every N env steps (0 disables)")
    parser.add_argument("--tensorboard", action="store_true", default=None,
                        help="also write TensorBoard event files")
    parser.add_argument("--algo-kwargs", default=None,
                        help="JSON dict of algorithm hyper-parameter overrides")
    parser.add_argument("--env-kwargs", default=None,
                        help="JSON dict of environment keyword arguments")


def _json_arg(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    data = json.loads(value)
    if not isinstance(data, dict):
        raise SystemExit("--algo-kwargs/--env-kwargs must be a JSON object")
    return data


def run(args: argparse.Namespace) -> int:
    config = load_config(
        args.config,
        env=args.env,
        algo=args.algo,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        run_name=args.run_name,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        tensorboard=args.tensorboard,
    )
    for key, value in (("env_kwargs", _json_arg(args.env_kwargs)),
                       ("algo_kwargs", _json_arg(args.algo_kwargs))):
        if value:
            getattr(config, key).update(value)

    trainer = Trainer(config)
    print(f"[roborlbench] {config.algo} on {config.env} "
          f"({trainer.env.backend}, {trainer.env.num_agents} agent(s), "
          f"{config.num_envs} env(s), {config.total_steps} steps)")
    print(f"[roborlbench] writing to {config.run_dir()}")
    summary = trainer.train()
    print(json.dumps(summary, indent=2))
    return 0
