"""M6 (optional): feature anchoring to the frozen S1 hidden states on the LM's mid and last layers."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F  # noqa: N812

from vlaretain.config import RunConfig
from vlaretain.methods.sft import FullSFT
from vlaretain.models.load import lm_hidden_states
from vlaretain.models.modules import num_lm_layers


def cosine_distance(student: torch.Tensor, teacher: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean of 1 - cos over non-padding tokens."""
    distance = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    mask = pad_mask.to(distance.dtype)
    return (distance * mask).sum() / mask.sum().clamp_min(1.0)


class FeatureAnchor(FullSFT):
    name = "anchor"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self.lam = float(cfg.method.params["lam"])
        self.reference: torch.nn.Module | None = None

    def build_model(self, s1_ckpt, cfg: RunConfig) -> torch.nn.Module:
        policy = super().build_model(s1_ckpt, cfg)
        self.reference = copy.deepcopy(policy).eval()
        for param in self.reference.parameters():
            param.requires_grad_(False)
        return policy

    def anchor_layers(self, policy: torch.nn.Module) -> tuple[int, int]:
        n = num_lm_layers(policy)
        return n // 2, n - 1

    def extra_loss(self, policy: torch.nn.Module, batch: dict, step: int) -> tuple[torch.Tensor, dict]:
        layers = self.anchor_layers(policy)
        student, pad_mask = lm_hidden_states(policy, batch)
        with torch.no_grad():
            teacher, _ = lm_hidden_states(self.reference, batch)
        distance = sum(cosine_distance(student[i], teacher[i], pad_mask) for i in layers) / len(layers)
        return self.lam * distance, {"anchor_distance": distance.item()}
