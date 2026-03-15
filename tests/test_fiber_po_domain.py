# Tests for FiberPO-Domain (four-level FGH).

import torch

from fiberpo.fiber_po_domain import (
    fiber_po_domain_gated_ratios,
    compute_policy_loss_fiberpo_domain,
)


def test_fiber_po_domain_shape() -> None:
    B, T = 4, 6
    ratio = torch.exp(torch.randn(B, T) * 0.1).clamp(0.5, 2.0)
    response_mask = torch.ones(B, T)
    trajectory_index = torch.tensor([0, 0, 1, 1])
    gated = fiber_po_domain_gated_ratios(
        ratio, response_mask,
        delta_domain=0.1, delta_prompt_group=0.15, delta_trajectory=0.2, epsilon_token=0.2,
        domain_index=None,
        prompt_group_index=None,
        trajectory_index=trajectory_index,
    )
    assert gated.shape == (B, T)


def test_fiber_po_domain_token_variation() -> None:
    """With domain+pg+trajectory indices, gated output must vary per-token (not collapse to per-group constant)."""
    B, T = 4, 6
    torch.manual_seed(7)
    ratio = torch.exp(torch.randn(B, T) * 0.15)
    mask = torch.ones(B, T)
    domain_index = torch.tensor([0, 0, 1, 1])
    prompt_group_index = torch.tensor([0, 0, 1, 1])
    trajectory_index = torch.arange(B)

    gated = fiber_po_domain_gated_ratios(
        ratio, mask,
        delta_domain=0.1, delta_prompt_group=0.15, delta_trajectory=0.2, epsilon_token=0.2,
        domain_index=domain_index,
        prompt_group_index=prompt_group_index,
        trajectory_index=trajectory_index,
    )
    # Each row must have per-token variation (token logclip of token residual must differ across tokens)
    for b in range(B):
        assert not torch.allclose(gated[b], gated[b, 0].expand(T)), \
            f"Row {b} has no token variation — token residual likely overwritten"


def test_compute_policy_loss_fiberpo_domain() -> None:
    B, T = 4, 6
    torch.manual_seed(42)
    old_log_prob = torch.randn(B, T) * 0.1
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T) * 0.3
    response_mask = torch.ones(B, T)

    loss, metrics = compute_policy_loss_fiberpo_domain(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=None,
    )
    assert loss.dim() == 0
    assert loss.isfinite().item()
    assert "actor/ppo_kl" in metrics
