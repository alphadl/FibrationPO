#!/usr/bin/env python3
# One-step FiberPO check (no verl). Run from repo root or with PYTHONPATH.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fiberpo.fiber_po import compute_policy_loss_fiberpo
from fiberpo.apc_obj import apc_obj_gated_ratios, apc_obj_surrogate
from fiberpo.fbg import fbg_gated_ratios_trajectory_token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apc", action="store_true", help="Run APC-Obj and FBG checks.")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    B, T = 4, 8
    torch.manual_seed(42)
    response_mask = (torch.rand(B, T) > 0.2).float()
    response_mask[:, -1] = 1.0

    old_log_prob = torch.randn(B, T, device=device) * 0.1
    log_prob = old_log_prob + torch.randn(B, T, device=device) * 0.2
    log_prob.requires_grad_(True)
    advantages = (torch.randn(B, T, device=device) * 0.5) * response_mask

    loss, metrics = compute_policy_loss_fiberpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=None,
    )
    print("FiberPO loss:", loss.item())
    print("FiberPO metrics:", metrics)
    loss.backward()
    print("FiberPO backward OK.")

    if args.apc:
        ratio = torch.exp(log_prob.detach() - old_log_prob)
        delta = 0.2
        _ = apc_obj_gated_ratios(ratio, response_mask, delta, None)
        obj_apc = apc_obj_surrogate(ratio, advantages, response_mask, delta, None)
        print("APC-Obj surrogate (maximize):", obj_apc.item())
        gated_fbg = fbg_gated_ratios_trajectory_token(
            ratio, response_mask, delta=0.2, epsilon=0.2, trajectory_index=None
        )
        print("FBG gated ratio shape:", gated_fbg.shape, "mean:", gated_fbg.float().mean().item())
        print("APC/FBG checks OK.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
