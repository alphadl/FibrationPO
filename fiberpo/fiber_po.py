# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# FiberPO = FBG instantiation (trajectory + token): g^agg(delta) + logclip(epsilon). Block-diagonal Jacobian; identity at on-policy.

from __future__ import annotations

from typing import Any, Optional

import torch

from fiberpo.fbg import fbg_gated_ratios_trajectory_token


def fiber_po_gated_ratios(
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float,
    epsilon: float,
    trajectory_index: Optional[torch.Tensor] = None,
    use_geometric_mean_base: bool = True,
) -> torch.Tensor:
    """FiberPO gated ratio: FBG with base budget delta, fiber budget epsilon."""
    return fbg_gated_ratios_trajectory_token(
        ratio=ratio,
        response_mask=response_mask,
        delta=delta,
        epsilon=epsilon,
        trajectory_index=trajectory_index,
        use_geometric_mean_base=use_geometric_mean_base,
    )


def compute_policy_loss_fiberpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Policy loss compatible with verl: L = -E[ advantage * gated_ratio ], FiberPO gating.
    config may have policy_loss.fiberpo_delta, policy_loss.fiberpo_epsilon (default 0.2).
    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    delta = 0.2
    epsilon = 0.2
    if config is not None:
        delta = getattr(config, "fiberpo_delta", None) or getattr(config, "clip_ratio", 0.2)
        epsilon = getattr(config, "fiberpo_epsilon", None) or 0.2
        if hasattr(config, "policy_loss"):
            pl = config.policy_loss
            if isinstance(pl, dict):
                delta = pl.get("fiberpo_delta", delta)
                epsilon = pl.get("fiberpo_epsilon", epsilon)
            elif hasattr(pl, "fiberpo_delta"):
                delta = getattr(pl, "fiberpo_delta", delta)
                epsilon = getattr(pl, "fiberpo_epsilon", epsilon)

    gated_ratio = fiber_po_gated_ratios(
        ratio=ratio,
        response_mask=response_mask.float(),
        delta=delta,
        epsilon=epsilon,
        trajectory_index=None,
        use_geometric_mean_base=True,
    )

    pg_losses = -advantages * gated_ratio
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    loss_agg_mode = loss_agg_mode or "seq-mean-token-mean"
    if loss_agg_mode == "token-mean":
        batch_num_tokens = response_mask.sum().clamp(min=1)
        pg_loss = (pg_losses * response_mask).sum() / batch_num_tokens
    elif loss_agg_mode in ("seq-mean-token-mean", "seq-mean-token-sum"):
        seq_losses = (pg_losses * response_mask).sum(dim=-1)
        seq_lens = response_mask.sum(dim=-1).clamp(min=1)
        seq_losses = seq_losses / seq_lens
        seq_mask = (response_mask.sum(dim=-1) > 0).float()
        pg_loss = (seq_losses * seq_mask).sum() / seq_mask.sum().clamp(min=1)
    else:
        batch_num_tokens = response_mask.sum().clamp(min=1)
        pg_loss = (pg_losses * response_mask).sum() / batch_num_tokens

    if config is not None and getattr(config, "global_batch_info", None):
        info = config.global_batch_info
        dp_size = info.get("dp_size", 1)
        batch_num_tokens_global = info.get("batch_num_tokens")
        if batch_num_tokens_global is not None and loss_agg_mode == "token-mean":
            pg_loss = pg_loss * dp_size
        elif info.get("global_batch_size") and loss_agg_mode != "token-mean":
            pg_loss = pg_loss * dp_size

    ppo_kl = (negative_approx_kl * response_mask).sum() / response_mask.sum().clamp(min=1)
    clipfrac = (torch.abs(ratio - gated_ratio) > 1e-5).float() * response_mask
    clipfrac = clipfrac.sum() / response_mask.sum().clamp(min=1)

    pg_metrics = {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, pg_metrics
