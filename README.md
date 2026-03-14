# Fibration Policy Optimization (FiberPO) — reproduction

**Not an official implementation.** Reproduction of [Fibration Policy Optimization](https://arxiv.org/abs/2603.08239) (arXiv:2603.08239).

## Contents

- **RGF, APC-Obj, FBG, FiberPO, FiberPO-Domain** — see [REPRODUCE.md](REPRODUCE.md).
- **verl integration**: policy loss plug-in; use with [verl](https://github.com/verl-project/verl) for full training.

## Setup

```bash
pip install -r requirements.txt
```

Optional (for training): install [verl](https://github.com/verl-project/verl), then before training:

```python
import fiberpo.losses
fiberpo.losses.register_fiberpo_with_verl()
```

Set `policy_loss.loss_mode: fiberpo` (or `fiberpo_domain`) in actor config.

## Quick check

```bash
python scripts/run_fiberpo_standalone.py
python scripts/run_fiberpo_standalone.py --apc
```

## Tests

```bash
python tests/run_tests.py
# or: pytest tests/ -v
```

## Data and runs

- Synthetic data: `python scripts/prepare_data.py --output data/train.parquet --num 200`
- Config snippet: `configs/fiberpo.yaml`
- With verl: use existing GRPO/PPO workflows and set loss_mode to `fiberpo`; see `scripts/README_VERL.md` for a minimal run.
