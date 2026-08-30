#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q .
python -m unittest discover -s tests -v
python pls_plan_train.py --config configs/study_180.yaml --explain-counts
python scripts/check_anonymity.py .
