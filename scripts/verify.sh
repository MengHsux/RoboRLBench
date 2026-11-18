#!/usr/bin/env bash
# Full local verification of a RoboRLBench installation.
#
#   ./scripts/verify.sh            # fast checks (~2 min)
#   ./scripts/verify.sh --full     # + learning-sanity tests (~5 min)
#
# Everything here runs on a CPU with no simulator installed.

set -euo pipefail

PYTHON="${PYTHON:-python}"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

cd "$(dirname "$0")/.."

section() { printf '\n=== %s\n' "$1"; }

section "interpreter"
"$PYTHON" - <<'PY'
import sys, torch, numpy
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}")
PY

section "backend diagnostics (analytic only)"
"$PYTHON" -m roborlbench doctor --analytic-only

section "registered baselines and environments"
"$PYTHON" -m roborlbench list

section "contract tests"
"$PYTHON" -m pytest -m "not slow"

section "every baseline performs a gradient update"
"$PYTHON" -m roborlbench smoke

if [[ "$FULL" == "1" ]]; then
    section "learning sanity (SAC on point-reach, MAPPO on multi-hunt-2d)"
    "$PYTHON" -m pytest -m slow
fi

printf '\nall checks passed\n'
