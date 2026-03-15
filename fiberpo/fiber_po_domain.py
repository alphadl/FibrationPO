# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# FiberPO-Domain: four-level FGH (domain, prompt_group, trajectory, token); g^agg at each level, residual = deviation from next-coarser aggregate.

from __future__ import annotations

from typing import Any, Optional

import torch


def _base_gate(x: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.clamp(x, -delta, delta)


def _logclip(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x, -eps, eps)


def fiber_po_domain_gated_ratios(
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    delta_domain: float,
    delta_prompt_group: float,
    delta_trajectory: float,
    epsilon_token: float,
    domain_index: Optional[torch.Tensor] = None,
    prompt_group_index: Optional[torch.Tensor] = None,
    trajectory_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Four-level FGH: apply g^agg at domain, prompt_group, trajectory; logclip at token.
    Gated residual at each level = deviation from next-coarser aggregate (paper: no new primitives).
    """
    B, T = ratio.shape
    device = ratio.device
    mask = response_mask.float()

    if trajectory_index is None:
        trajectory_index = torch.arange(B, device=device)
    r_base_traj = _aggregate_and_gate(ratio, mask, trajectory_index, delta_trajectory)
    # Save token-level residual (deviation from trajectory aggregate) before coarser levels overwrite it.
    token_log_res = torch.log(ratio.clamp(min=1e-8)) - torch.log(r_base_traj.clamp(min=1e-8))
    if prompt_group_index is not None:
        r_base_pg = _aggregate_and_gate(r_base_traj, mask, prompt_group_index, delta_prompt_group)
        r_base_traj = r_base_pg
    if domain_index is not None:
        r_base_dom = _aggregate_and_gate(r_base_traj, mask, domain_index, delta_domain)
        r_base_traj = r_base_dom
    gated_log_res = _logclip(token_log_res, epsilon_token) * mask
    r_fiber = torch.exp(gated_log_res)
    gated_ratio = r_base_traj * r_fiber
    return gated_ratio * mask + (1.0 - mask)


def _aggregate_and_gate(
    r: torch.Tensor,
    mask: torch.Tensor,
    group_id: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Per-group geometric-mean ratio, then g^agg(delta); broadcast to (B,T)."""
    B, T = r.shape
    device = r.device
    uniq = group_id.unique()
    out_base = torch.ones(B, T, device=device, dtype=r.dtype)
    for g in uniq:
        if g < 0:
            continue
        sel = (group_id.view(B, 1).expand(-1, T) == g) & mask.bool()
        if sel.sum() == 0:
            continue
        n = sel.sum().float().clamp(min=1)
        log_r = torch.log(r.clamp(min=1e-8))
        agg = (log_r * sel.float()).sum() / n
        gated = _base_gate(agg, delta)
        r_base = torch.exp(gated)
        out_base = torch.where(sel, r_base, out_base)
    return out_base


def compute_policy_loss_fiberpo_domain(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Policy loss for FiberPO-Domain; config may provide domain_index, prompt_group_index, trajectory_index and delta/epsilon params."""
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    delta_d = 0.1
    delta_pg = 0.15
    delta_t = 0.2
    epsilon = 0.2
    if config is not None:
        delta_d = getattr(config, "fiberpo_delta_domain", delta_d)
        delta_pg = getattr(config, "fiberpo_delta_prompt_group", delta_pg)
        delta_t = getattr(config, "fiberpo_delta_trajectory", delta_t)
        epsilon = getattr(config, "fiberpo_epsilon", epsilon)
        if hasattr(config, "policy_loss") and isinstance(getattr(config, "policy_loss"), dict):
            pl = config.policy_loss
            delta_d = pl.get("fiberpo_delta_domain", delta_d)
            delta_pg = pl.get("fiberpo_delta_prompt_group", delta_pg)
            delta_t = pl.get("fiberpo_delta_trajectory", delta_t)
            epsilon = pl.get("fiberpo_epsilon", epsilon)

    domain_index = getattr(config, "domain_index", None) if config else None
    prompt_group_index = getattr(config, "prompt_group_index", None) if config else None
    trajectory_index = getattr(config, "trajectory_index", None) if config else None

    gated_ratio = fiber_po_domain_gated_ratios(
        ratio=ratio,
        response_mask=response_mask.float(),
        delta_domain=delta_d,
        delta_prompt_group=delta_pg,
        delta_trajectory=delta_t,
        epsilon_token=epsilon,
        domain_index=domain_index,
        prompt_group_index=prompt_group_index,
        trajectory_index=trajectory_index,
    )

    pg_losses = -advantages * gated_ratio
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss_agg_mode = loss_agg_mode or "seq-mean-token-mean"
    if loss_agg_mode == "token-mean":
        pg_loss = (pg_losses * response_mask).sum() / response_mask.sum().clamp(min=1)
    else:
        seq_losses = (pg_losses * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
        seq_mask = (response_mask.sum(dim=-1) > 0).float()
        pg_loss = (seq_losses * seq_mask).sum() / seq_mask.sum().clamp(min=1)

    if config and getattr(config, "global_batch_info", None):
        pg_loss = pg_loss * config.global_batch_info.get("dp_size", 1)

    ppo_kl = (negative_approx_kl * response_mask).sum() / response_mask.sum().clamp(min=1)
    clipfrac = (torch.abs(ratio - gated_ratio) > 1e-5).float() * response_mask
    clipfrac = clipfrac.sum() / response_mask.sum().clamp(min=1)
    pg_metrics = {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, pg_metrics
