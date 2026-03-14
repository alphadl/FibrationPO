# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# Policy loss API compatible with verl.trainer.ppo.core_algos.PolicyLossFn.

from __future__ import annotations

from typing import Any

from fiberpo.fiber_po import compute_policy_loss_fiberpo
from fiberpo.fiber_po_domain import compute_policy_loss_fiberpo_domain


def _register_if_verl_available() -> None:
    try:
        from verl.trainer.ppo import core_algos
        if "fiberpo" not in core_algos.POLICY_LOSS_REGISTRY:
            core_algos.register_policy_loss("fiberpo")(compute_policy_loss_fiberpo)
        if "fiberpo_domain" not in core_algos.POLICY_LOSS_REGISTRY:
            core_algos.register_policy_loss("fiberpo_domain")(compute_policy_loss_fiberpo_domain)
    except Exception:
        pass


_register_if_verl_available()


def register_fiberpo_with_verl() -> bool:
    """Register FiberPO and FiberPO-Domain into verl POLICY_LOSS_REGISTRY. Returns True if verl is available."""
    try:
        from verl.trainer.ppo import core_algos
        core_algos.register_policy_loss("fiberpo")(compute_policy_loss_fiberpo)
        core_algos.register_policy_loss("fiberpo_domain")(compute_policy_loss_fiberpo_domain)
        return True
    except Exception:
        return False
