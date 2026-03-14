# Tests for RGF (Eq 2).

import torch

from fiberpo.rgf import rgf_surrogate


def test_rgf_surrogate() -> None:
    gated_ratios = torch.ones(2, 4)
    advantages = torch.randn(2, 4) * 0.5
    mask = torch.ones(2, 4)
    s = rgf_surrogate(gated_ratios, advantages, mask)
    assert s.dim() == 0
    assert s.isfinite()
