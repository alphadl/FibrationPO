# Tests for APC-Obj (Eq 3, Def 3.2).

import torch

from fiberpo.apc_obj import apc_obj_gated_ratios, apc_obj_surrogate, _clip_symmetric


def test_clip_symmetric() -> None:
    x = torch.tensor([-2.0, 0.5, 3.0])
    B = torch.tensor([1.0, 1.0, 1.0])
    out = _clip_symmetric(x, B)
    assert out[0].item() == -1.0
    assert out[1].item() == 0.5
    assert out[2].item() == 1.0


def test_apc_obj_on_policy() -> None:
    B, T = 4, 6
    ratio = torch.ones(B, T)
    response_mask = (torch.rand(B, T) > 0.3).float()
    response_mask[:, 0] = 1.0
    gated = apc_obj_gated_ratios(ratio, response_mask, delta=0.2, trajectory_index=None)
    assert torch.allclose(gated, torch.zeros_like(gated))


def test_apc_obj_surrogate_shape() -> None:
    B, T = 2, 4
    ratio = torch.exp(torch.randn(B, T) * 0.1).clamp(0.5, 2.0)
    response_mask = torch.ones(B, T)
    advantages = torch.randn(B, T) * 0.5
    s = apc_obj_surrogate(ratio, advantages, response_mask, delta=0.2, trajectory_index=None)
    assert s.dim() == 0
    assert s.isfinite().item()
