# Full local verification of a RoboRLBench installation (Windows).
#
#   .\scripts\verify.ps1            # fast checks (~2 min)
#   .\scripts\verify.ps1 -Full      # + learning-sanity tests (~5 min)
#
# Everything here runs on a CPU with no simulator installed.

param([switch]$Full)

$ErrorActionPreference = "Stop"
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Push-Location (Join-Path $PSScriptRoot "..")
try {
    function Section($name) { Write-Host "`n=== $name" -ForegroundColor Cyan }

    Section "interpreter"
    & $python -c "import sys, torch, numpy; print(f'python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}')"

    Section "backend diagnostics (analytic only)"
    & $python -m roborlbench doctor --analytic-only

    Section "registered baselines and environments"
    & $python -m roborlbench list

    Section "contract tests"
    & $python -m pytest -m "not slow"

    Section "every baseline performs a gradient update"
    & $python -m roborlbench smoke

    if ($Full) {
        Section "learning sanity (SAC on point-reach, MAPPO on multi-hunt-2d)"
        & $python -m pytest -m slow
    }

    Write-Host "`nall checks passed" -ForegroundColor Green
}
finally {
    Pop-Location
}
