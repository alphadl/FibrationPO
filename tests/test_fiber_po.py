# Tests for FiberPO policy loss.

import torch

from fiberpo.fiber_po import fiber_po_gated_ratios, compute_policy_loss_fiberpo


def test_fiberpo_gated_ratios_on_policy() -> None:
    B, T = 2, 4
    ratio = torch.ones(B, T)
    response_mask = torch.ones(B, T)
    gated = fiber_po_gated_ratios(ratio, response_mask, delta=0.2, epsilon=0.2)
    assert torch.allclose(gated * response_mask, response_mask)


def test_compute_policy_loss_fiberpo() -> None:
    B, T = 4, 8
    torch.manual_seed(123)
    old_log_prob = torch.randn(B, T) * 0.1
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T) * 0.3
    response_mask = (torch.rand(B, T) > 0.3).float()
    response_mask[:, -1] = 1.0

    class Config:
        clip_ratio = 0.2
        global_batch_info = {}

    loss, metrics = compute_policy_loss_fiberpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=Config(),
    )
    assert loss.dim() == 0
    assert loss.isfinite().item()
    assert "actor/pg_clipfrac" in metrics
    assert "actor/ppo_kl" in metrics


def test_compute_policy_loss_fiberpo_backward() -> None:
    B, T = 2, 4
    torch.manual_seed(1)
    old_log_prob = torch.randn(B, T) * 0.1
    log_prob = torch.randn(B, T) * 0.1
    log_prob.requires_grad_(True)
    advantages = torch.randn(B, T) * 0.2
    response_mask = torch.ones(B, T)

    loss, _ = compute_policy_loss_fiberpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=None,
    )
    loss.backward()
    assert log_prob.grad is not None
