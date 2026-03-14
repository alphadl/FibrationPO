# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# Paper Section 3.2, Definition 3.2, Eq (3); Theorem 3.3 (APC-Obj equiv. to sample-based TV-TRPO).

from __future__ import annotations

from typing import Optional

import torch


def _clip_symmetric(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Paper: clip(a, B) = clip(a, -B+, B+), B+ = max(B, 0)."""
    B_pos = B.clamp(min=0)
    return torch.clamp(x, -B_pos, B_pos)


def apc_obj_gated_ratios(
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float,
    trajectory_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-entry gated deviation for APC-Obj (Eq 3): clip(r_sa - 1, B_sa).
    B_sa = T_s*delta - sum_{other (a',tau',t') in same state} |r_sa' - 1|.
    In LLM we take state = trajectory (one per sequence), so T_s = token count of that trajectory.
    Returns gated deviation; gated_ratio = 1 + gated_deviation.
    """
    dev = ratio - 1.0
    abs_dev = (dev * response_mask).abs()

    if trajectory_index is None:
        T_s = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        sum_abs = abs_dev.sum(dim=-1, keepdim=True)
        B_per_token = T_s * delta - sum_abs + abs_dev
        B_per_token = B_per_token * response_mask
    else:
        B, T = ratio.shape
        device = ratio.device
        traj_ids = trajectory_index.view(B, 1).expand(-1, T)
        uniq = trajectory_index.unique()
        gated_dev = torch.zeros_like(dev, device=device)
        for uid in uniq:
            if uid < 0:
                continue
            sel = (traj_ids == uid) & (response_mask.bool())
            if sel.sum() == 0:
                continue
            T_s = sel.sum().float().clamp(min=1)
            sum_abs_per_traj = (abs_dev * sel.float()).sum()
            B_ij = (T_s * delta - sum_abs_per_traj + abs_dev) * sel.float()
            gated_dev = torch.where(sel, _clip_symmetric(dev, B_ij), gated_dev)
        return gated_dev

    gated_deviation = _clip_symmetric(dev, B_per_token)
    return gated_deviation * response_mask


def apc_obj_surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float,
    trajectory_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """APC-Obj objective (Eq 3): (1/T) sum [ clip(r-1, B)*A^ + A^ ]. Scalar to maximize."""
    gated_dev = apc_obj_gated_ratios(ratio, response_mask, delta, trajectory_index)
    gated_ratio = 1.0 + gated_dev
    T = response_mask.sum().clamp(min=1)
    return ((gated_ratio * advantages) * response_mask).sum() / T
