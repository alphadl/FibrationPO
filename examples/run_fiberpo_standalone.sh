#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python scripts/run_fiberpo_standalone.py
python scripts/run_fiberpo_standalone.py --apc
echo "Done."
