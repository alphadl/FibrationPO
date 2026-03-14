# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.

from fiberpo.rgf import rgf_surrogate
from fiberpo.apc_obj import apc_obj_surrogate, apc_obj_gated_ratios
from fiberpo.fbg import (
    base_aggregate_gate,
    logclip_fiber_gate,
    fbg_gated_ratios_trajectory_token,
)
from fiberpo.fiber_po import fiber_po_gated_ratios, compute_policy_loss_fiberpo
from fiberpo.fiber_po_domain import fiber_po_domain_gated_ratios, compute_policy_loss_fiberpo_domain
from fiberpo.losses import register_fiberpo_with_verl

__all__ = [
    "rgf_surrogate",
    "apc_obj_surrogate",
    "apc_obj_gated_ratios",
    "base_aggregate_gate",
    "logclip_fiber_gate",
    "fbg_gated_ratios_trajectory_token",
    "fiber_po_gated_ratios",
    "compute_policy_loss_fiberpo",
    "fiber_po_domain_gated_ratios",
    "compute_policy_loss_fiberpo_domain",
    "register_fiberpo_with_verl",
]
