# Fibration Policy Optimization (arXiv:2603.08239) — not an official implementation.
# Paper Section 3.1, Definition 3.1, Eq (2): RGF surrogate J^ = sum mu * G(r)_{s,a,I} * A^.

from __future__ import annotations

from typing import Optional

import torch


def rgf_surrogate(
    gated_ratios: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RGF objective (Eq 2): sum over (s,a,I) of mu * G(r)_{s,a,I} * A^. Maximize this scalar."""
    if weights is None:
        weights = torch.ones_like(advantages, dtype=advantages.dtype, device=advantages.device)
    term = weights * gated_ratios * advantages
    return (term * mask).sum() / mask.sum().clamp(min=1)
