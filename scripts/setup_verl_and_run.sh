#!/bin/bash
# Clone verl (if not present), install deps, run minimal FiberPO check with verl policy loss.
# Run from FibrationPO repo root. Needs: pip, git.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERL_DIR="${VERL_DIR:-$ROOT/verl}"
if [ ! -d "$VERL_DIR" ]; then
  echo "Cloning verl into $VERL_DIR ..."
  git clone --depth 1 https://github.com/verl-project/verl.git "$VERL_DIR"
fi
pip install -e "$VERL_DIR" -q 2>/dev/null || true
pip install -r "$ROOT/requirements.txt" -q

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python -c "
from fiberpo.losses import register_fiberpo_with_verl
ok = register_fiberpo_with_verl()
assert ok, 'verl not available'
from verl.trainer.ppo.core_algos import get_policy_loss_fn
fn = get_policy_loss_fn('fiberpo')
print('FiberPO registered and get_policy_loss_fn(\"fiberpo\") OK')
"

python scripts/run_fiberpo_standalone.py
python scripts/run_fiberpo_standalone.py --apc
echo "Done."
