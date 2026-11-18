"""Command line interface.

``roborlbench train`` / ``eval`` / ``list`` / ``doctor`` / ``smoke`` are the five
sub-commands.  Every sub-command lives in its own module and exposes the same
``add_arguments(parser)`` / ``run(args) -> int`` pair, so adding one is a
three-line change here.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import doctor as doctor_cmd
from . import eval as eval_cmd
from . import smoke as smoke_cmd
from . import train as train_cmd
from ..algos import ALGOS
from ..envs import ENVS

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples
  roborlbench doctor
  roborlbench list --family single
  roborlbench train --algo crossq --env go2-flat --num-envs 4096 --total-steps 20e6
  roborlbench train --algo hasac --env hunt-multi-mujoco --num-envs 8
  roborlbench eval --checkpoint runs/crossq-go2-flat-seed0/final.pt --episodes 20
  roborlbench smoke --algos sac,crossq,mappo,hasac
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roborlbench",
        description="Train and evaluate the RoboRLBench deep-RL baselines.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train one baseline on one environment")
    train_cmd.add_arguments(p_train)
    p_train.set_defaults(func=train_cmd.run)

    p_eval = sub.add_parser("eval", help="evaluate a saved checkpoint")
    eval_cmd.add_arguments(p_eval)
    p_eval.set_defaults(func=eval_cmd.run)

    p_smoke = sub.add_parser("smoke", help="train every baseline for a few hundred steps")
    smoke_cmd.add_arguments(p_smoke)
    p_smoke.set_defaults(func=smoke_cmd.run)

    p_doctor = sub.add_parser("doctor", help="check installed backends and environments")
    doctor_cmd.add_arguments(p_doctor)
    p_doctor.set_defaults(func=doctor_cmd.run)

    p_list = sub.add_parser("list", help="list registered baselines and environments")
    p_list.add_argument("--family", choices=["single", "multi"], default=None,
                        help="only show baselines of this family")
    p_list.add_argument("--env-kind", choices=["single", "multi"], default=None,
                        help="only show environments with this number of robots")
    p_list.set_defaults(func=_cmd_list)

    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    names = sorted(n for n, s in ALGOS.items()
                   if args.family is None or s.family == args.family)
    header = f"{'algorithm':12s} {'family':7s} {'kind':10s} {'venue':26s} year  description"
    print("baselines")
    print(header)
    print("-" * len(header))
    for name in names:
        spec = ALGOS[name]
        print(f"{name:12s} {spec.family:7s} {spec.kind:10s} "
              f"{spec.venue[:26]:26s} {spec.year:4d}  {spec.description}")

    print("\nenvironments")
    header = f"{'environment':22s} {'robots':7s} {'backend':10s} description"
    print(header)
    print("-" * len(header))
    for env_name, spec in sorted(ENVS.items()):
        if args.env_kind is not None and spec.kind != args.env_kind:
            continue
        print(f"{env_name:22s} {spec.kind:7s} {spec.backend:10s} {spec.description}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _make_stdout_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:                            # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


def _make_stdout_safe() -> None:
    """Never crash on a console whose code page cannot encode a path."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):             # pragma: no cover
            pass


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())
