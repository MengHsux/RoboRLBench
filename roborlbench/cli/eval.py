"""``roborlbench eval`` - score a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from ..algos import build_agent
from ..envs import make_env
from ..runners.evaluate import evaluate_policy

__all__ = ["add_arguments", "run"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="path to a *.pt checkpoint")
    parser.add_argument("--env", default=None,
                        help="environment name (default: read from the run's config.json)")
    parser.add_argument("--algo", default=None,
                        help="baseline name (default: read from the run's config.json)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="parallel copies; the return is averaged over all of them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--algo-kwargs", default=None,
                        help="JSON dict of algorithm hyper-parameter overrides")
    parser.add_argument("--env-kwargs", default=None,
                        help="JSON dict of environment keyword arguments")


def _run_metadata(checkpoint: str) -> Dict[str, Any]:
    """Read ``config.json`` from the run directory, if it is still there."""
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_arg(value: Optional[str]) -> Dict[str, Any]:
    return json.loads(value) if value else {}


def run(args: argparse.Namespace) -> int:
    meta = _run_metadata(args.checkpoint)
    env_name = args.env or meta.get("env")
    algo_name = args.algo or meta.get("algo")
    if not env_name or not algo_name:
        raise SystemExit(
            "could not infer --env/--algo from the run directory; pass them explicitly"
        )

    env_kwargs = dict(meta.get("env_kwargs") or {})
    env_kwargs.update(_json_arg(args.env_kwargs))
    algo_kwargs = dict(meta.get("algo_kwargs") or {})
    algo_kwargs.update(_json_arg(args.algo_kwargs))

    env = make_env(env_name, num_envs=args.num_envs, seed=args.seed,
                   device=args.device, **env_kwargs)
    agent = build_agent(
        algo_name,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_agents=env.num_agents,
        state_dim=env.state_dim,
        num_envs=env.num_envs,
        device=args.device,
        seed=args.seed,
        **algo_kwargs,
    )
    agent.load(args.checkpoint, map_location=args.device)

    result = evaluate_policy(agent, env, episodes=args.episodes)
    result.update({"algo": algo_name, "env": env_name, "checkpoint": args.checkpoint})
    print(json.dumps(result, indent=2))
    return 0
