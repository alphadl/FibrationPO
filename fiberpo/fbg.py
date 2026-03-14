# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# Paper Section 4: FBG — base-level gate g^agg (trajectory budget delta), fiber-level logclip (epsilon). Theorem 4.5.

from __future__ import annotations

from typing import Optional

import torch


def base_aggregate_gate(aggregate_deviation: torch.Tensor, delta: float) -> torch.Tensor:
    """Base-level gate g^agg: clip aggregate deviation to [-delta, delta] (trajectory-level budget)."""
    return torch.clamp(aggregate_deviation, -delta, delta)


def logclip_fiber_gate(
    log_residual: torch.Tensor,
    epsilon: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fiber-level logclip: clip(log residual) to [-epsilon, epsilon]. Paper: bounds each token residual by epsilon."""
    out = torch.clamp(log_residual, -epsilon, epsilon)
    if mask is not None:
        out = out * mask
    return out


def fbg_gated_ratios_trajectory_token(
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float,
    epsilon: float,
    trajectory_index: Optional[torch.Tensor] = None,
    use_geometric_mean_base: bool = True,
) -> torch.Tensor:
    """
    FBG two-level (base=trajectory, fiber=token): pushforward to base -> g^agg(delta) -> reflect -> logclip(epsilon).
    Reflecting: on-policy r_base=1, r_fiber=1 => r_gated=1; base and fiber are orthogonal.
    """
    if trajectory_index is None:
        seq_lens = response_mask.sum(dim=-1).clamp(min=1)
        if use_geometric_mean_base:
            log_ratio = torch.log(ratio.clamp(min=1e-8))
            log_ratio_masked = torch.where(response_mask.bool(), log_ratio, torch.zeros_like(log_ratio))
            log_r_agg = log_ratio_masked.sum(dim=-1) / seq_lens
            agg_dev = log_r_agg
        else:
            dev = (ratio - 1.0) * response_mask
            agg_dev = dev.sum(dim=-1) / seq_lens
        gated_agg_dev = base_aggregate_gate(agg_dev, delta)
        if use_geometric_mean_base:
            r_base_gated = torch.exp(gated_agg_dev)
        else:
            r_base_gated = 1.0 + gated_agg_dev
        r_base_gated = r_base_gated.unsqueeze(-1)
        log_residual = torch.log(ratio.clamp(min=1e-8)) - torch.log(r_base_gated.clamp(min=1e-8))
        gated_log_res = logclip_fiber_gate(log_residual, epsilon, response_mask)
        r_fiber_gated = torch.exp(gated_log_res)
        gated_ratio = (r_base_gated * r_fiber_gated) * response_mask + (1.0 - response_mask)
        return gated_ratio

    B, T = ratio.shape
    device = ratio.device
    traj_ids = trajectory_index.view(B, 1).expand(-1, T)
    uniq = trajectory_index.unique()
    gated_ratio = torch.ones_like(ratio, device=device)
    for uid in uniq:
        if uid < 0:
            continue
        sel = (traj_ids == uid) & (response_mask.bool())
        if sel.sum() == 0:
            continue
        seq_len = sel.sum().float().clamp(min=1)
        if use_geometric_mean_base:
            log_r = torch.log(ratio.clamp(min=1e-8))
            log_r_agg = (log_r * sel.float()).sum() / seq_len
            agg_dev = log_r_agg
        else:
            dev = (ratio - 1.0) * sel.float()
            agg_dev = dev.sum() / seq_len
        gated_agg_dev = base_aggregate_gate(agg_dev.unsqueeze(0), delta).squeeze(0)
        if use_geometric_mean_base:
            r_base_gated = torch.exp(gated_agg_dev)
        else:
            r_base_gated = 1.0 + gated_agg_dev
        log_residual = torch.log(ratio.clamp(min=1e-8)) - torch.log(r_base_gated.clamp(min=1e-8) + 1e-10)
        gated_log_res = logclip_fiber_gate(log_residual, epsilon, sel.float())
        r_fiber = torch.exp(gated_log_res)
        gated_ratio = torch.where(sel, r_base_gated * r_fiber, gated_ratio)
    return gated_ratio * response_mask + (1.0 - response_mask)
