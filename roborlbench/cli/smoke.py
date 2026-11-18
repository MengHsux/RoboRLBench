"""``roborlbench smoke`` - train every baseline for a handful of steps.

This is the "does it actually run" check: for each baseline it builds the
environment, collects a few hundred transitions, performs at least one
gradient update, evaluates once and saves a checkpoint.  It is the command the
CI job and the contributors run before opening a pull request.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import traceback
from typing import Any, Dict, List

from ..algos import get_algo_spec, list_algos
from ..runners import TrainConfig, Trainer

__all__ = ["add_arguments", "run"]

#: Overrides that make a full update fit into a few hundred environment steps.
_OFF_POLICY = {"buffer_size": 8192, "warmup": 128, "batch_size": 32, "grad_steps": 1}
_ON_POLICY = {"rollout_steps": 8, "num_epochs": 1, "num_minibatches": 2}

#: Columns written by every run; anything else proves an update happened.
_BASE_COLUMNS = {"train_return", "steps_per_s", "eval_return"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algos", default=None,
                        help="comma separated subset (default: all twenty)")
    parser.add_argument("--env", default=None,
                        help="force one environment for every baseline")
    parser.add_argument("--env-single", default="point-reach")
    parser.add_argument("--env-multi", default="multi-hunt-2d")
    parser.add_argument("--steps", type=int, default=512,
                        help="environment transitions per baseline")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default=None,
                        help="keep the run directories here (default: a temp dir)")
    return None


def _train_one(
    name: str, args: argparse.Namespace, log_dir: str
) -> Dict[str, Any]:
    spec = get_algo_spec(name)
    env_name = args.env or (args.env_single if spec.family == "single" else args.env_multi)
    overrides = dict(_OFF_POLICY if spec.kind == "off_policy" else _ON_POLICY)

    config = TrainConfig(
        env=env_name,
        algo=name,
        num_envs=args.num_envs,
        total_steps=args.steps,
        seed=0,
        device=args.device,
        eval_interval=max(args.steps // 2, 1),
        eval_episodes=1,
        save_interval=0,
        log_dir=log_dir,
        run_name=f"smoke-{name}",
        algo_kwargs=overrides,
    )
    started = time.time()
    trainer = Trainer(config)
    summary = trainer.train()
    elapsed = time.time() - started

    trained = any(set(row) - _BASE_COLUMNS for row in trainer.history)
    value = summary.get("final_eval_return")
    return {
        "algorithm": name,
        "family": spec.family,
        "kind": spec.kind,
        "env": env_name,
        "steps": summary["steps"],
        "updates": trained,
        "eval_return": value,
        "seconds": elapsed,
        "ok": bool(trained) and value is not None and value == value,  # NaN-safe
        "error": "",
    }


def run(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.algos.split(",")] if args.algos else list_algos()
    log_dir = args.log_dir or os.path.join(tempfile.mkdtemp(prefix="roborlbench-smoke-"), "")
    os.makedirs(log_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for name in names:
        print(f"[smoke] {name} ... ", end="", flush=True)
        try:
            row = _train_one(name, args, log_dir)
        except Exception:                              # noqa: BLE001 - report below
            row = {
                "algorithm": name, "family": "-", "kind": "-", "env": args.env or "-",
                "steps": 0, "updates": False, "eval_return": None,
                "seconds": 0.0, "ok": False, "error": traceback.format_exc(limit=4),
            }
        rows.append(row)
        if row["ok"]:
            print(f"ok ({row['seconds']:.1f}s, eval={row['eval_return']:.3f})")
        elif row["error"]:
            print("FAILED")
            print(row["error"])
        else:
            print(f"no-update (updates={row['updates']}, eval={row['eval_return']})")

    _print_table(rows)
    failed = [r["algorithm"] for r in rows if not r["ok"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} baselines trained successfully")
    if failed:
        print(f"failed: {failed}")
        print(f"logs kept in {log_dir}")
        return 1
    print(f"logs kept in {log_dir}")
    return 0


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'algorithm':12s} {'family':7s} {'kind':10s} {'env':16s} " \
             f"{'steps':>7s} {'upd':>4s} {'eval':>10s} {'sec':>6s}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        value = row["eval_return"]
        shown = "n/a" if value is None else f"{value:.3f}"
        print(
            f"{row['algorithm']:12s} {row['family']:7s} {row['kind']:10s} "
            f"{row['env']:16s} {row['steps']:7d} {str(row['updates']):>4s} "
            f"{shown:>10s} {row['seconds']:6.1f}"
        )
