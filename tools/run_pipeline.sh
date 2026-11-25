#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/runs artifacts/features artifacts/results figures report
PY=.venv/bin/python
"$PY" src/prepare.py > artifacts/prepare.log 2>&1
"$PY" -m unittest discover -s tests -v > artifacts/smoke_tests.log 2>&1
"$PY" src/train.py > artifacts/training.log 2>&1
"$PY" src/selection.py > artifacts/selection.log 2>&1
"$PY" src/evaluate.py > artifacts/evaluation.log 2>&1
"$PY" src/figures.py > artifacts/figures.log 2>&1
"$PY" src/verify.py > artifacts/verification.log 2>&1
