"""The CLI is the documented entry point, so it gets end-to-end coverage."""

from __future__ import annotations

import json
import os

import pytest

from roborlbench.cli import build_parser, main


def test_parser_exposes_every_subcommand():
    parser = build_parser()
    for command in ("train", "eval", "smoke", "doctor", "list"):
        args = parser.parse_args([command] + (["--checkpoint", "x.pt"] if command == "eval" else []))
        assert callable(args.func), command


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "baselines" in out and "environments" in out
    assert "crossq" in out and "go2-flat" in out


def test_list_filters(capsys):
    assert main(["list", "--family", "multi", "--env-kind", "multi"]) == 0
    out = capsys.readouterr().out
    baselines, environments = out.split("environments")
    first_column = [line.split()[0] for line in baselines.splitlines() if " " in line]
    assert "mappo" in first_column and "sac" not in first_column
    assert "multi-hunt-2d" in environments and "point-reach" not in environments


def test_doctor_analytic_only(capsys):
    assert main(["doctor", "--analytic-only"]) == 0
    out = capsys.readouterr().out
    assert "point-reach" in out and "multi-hunt-2d" in out


def test_train_then_eval(tmp_path, capsys):
    run_dir = tmp_path / "runs"
    assert main([
        "train",
        "--algo", "sac",
        "--env", "point-reach",
        "--num-envs", "2",
        "--total-steps", "256",
        "--eval-interval", "128",
        "--eval-episodes", "1",
        "--save-interval", "0",
        "--log-dir", str(run_dir),
        "--run-name", "cli-sac",
        "--algo-kwargs", json.dumps({"buffer_size": 8192, "warmup": 128,
                                     "batch_size": 32, "grad_steps": 1}),
    ]) == 0
    capsys.readouterr()

    summary_path = os.path.join(run_dir, "cli-sac", "summary.json")
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["algo"] == "sac" and summary["env"] == "point-reach"
    assert os.path.isfile(summary["checkpoint"])

    # ``eval`` must recover the algorithm/environment from config.json
    assert main(["eval", "--checkpoint", summary["checkpoint"],
                 "--episodes", "1"]) == 0
    out = capsys.readouterr().out
    assert "eval_return" in out


def test_eval_without_metadata_requires_explicit_names(tmp_path):
    checkpoint = tmp_path / "orphan.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit):
        main(["eval", "--checkpoint", str(checkpoint)])


def test_smoke_subset(capsys):
    assert main(["smoke", "--algos", "sac,mappo", "--steps", "192",
                 "--num-envs", "2"]) == 0
    out = capsys.readouterr().out
    assert "2/2 baselines trained successfully" in out
