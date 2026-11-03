"""M5: L2-SP, a proximity penalty to S1 on the trainable parameters."""

from __future__ import annotations

import torch

from vlaretain.config import RunConfig
from vlaretain.methods.sft import FullSFT


def l2sp_penalty(policy: torch.nn.Module, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean over all trainable scalars of (theta - theta_S1)^2."""
    total = None
    count = 0
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        term = torch.sum((param - reference[name]) ** 2)
        total = term if total is None else total + term
        count += param.numel()
    if total is None:
        raise RuntimeError("L2-SP found no trainable parameters")
    return total / count


class L2SP(FullSFT):
    name = "l2sp"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self.lam = float(cfg.method.params["lam"])
        self.reference: dict[str, torch.Tensor] = {}

    def build_model(self, s1_ckpt, cfg: RunConfig) -> torch.nn.Module:
        policy = super().build_model(s1_ckpt, cfg)
        # Captured before any resume happens, so the reference is always S1.
        self.reference = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
        return policy

    def extra_loss(self, policy: torch.nn.Module, batch: dict, step: int) -> tuple[torch.Tensor, dict]:
        penalty = l2sp_penalty(policy, self.reference)
        return self.lam * penalty, {"l2sp_penalty": penalty.item()}
