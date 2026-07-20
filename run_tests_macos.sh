#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "radio-signal-intelligence-lab — macOS validation"
echo "============================================================"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
radio-intel-lab demo --config configs/demo.json

echo
echo "Validation complete."
echo "Plot: $(pwd)/results/reference/radio_signal_intelligence_summary.png"
