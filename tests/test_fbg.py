# Tests for FBG (Sec 4: g^agg, logclip).

import torch

from fiberpo.fbg import (
    base_aggregate_gate,
    logclip_fiber_gate,
    fbg_gated_ratios_trajectory_token,
)


def test_base_aggregate_gate() -> None:
    x = torch.tensor([-0.5, 0.0, 0.5, 2.0])
    out = base_aggregate_gate(x, delta=0.3)
    assert abs(out[0].item() - (-0.3)) < 1e-5
    assert abs(out[1].item() - 0.0) < 1e-5
    assert abs(out[2].item() - 0.3) < 1e-5
    assert abs(out[3].item() - 0.3) < 1e-5


def test_logclip_fiber_gate() -> None:
    log_res = torch.tensor([[-1.0, 0.0, 1.0]])
    out = logclip_fiber_gate(log_res, epsilon=0.5)
    assert out[0, 0].item() == -0.5
    assert out[0, 1].item() == 0.0
    assert out[0, 2].item() == 0.5


def test_fbg_on_policy() -> None:
    B, T = 3, 5
    ratio = torch.ones(B, T)
    response_mask = torch.ones(B, T)
    gated = fbg_gated_ratios_trajectory_token(
        ratio, response_mask, delta=0.2, epsilon=0.2, trajectory_index=None
    )
    assert torch.allclose(gated * response_mask, response_mask)


def test_fbg_gated_ratio_shape() -> None:
    B, T = 4, 8
    ratio = torch.exp(torch.randn(B, T) * 0.1).clamp(0.3, 3.0)
    response_mask = (torch.rand(B, T) > 0.2).float()
    response_mask[:, -1] = 1.0
    gated = fbg_gated_ratios_trajectory_token(
        ratio, response_mask, delta=0.2, epsilon=0.2, trajectory_index=None
    )
    assert gated.shape == (B, T)
    assert (gated >= 0).all()
