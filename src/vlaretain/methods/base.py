"""The Method interface. The training loop only ever calls these three functions."""

from __future__ import annotations

from pathlib import Path

import torch

from vlaretain.config import RunConfig
from vlaretain.models.load import load_policy_for_training


class Method:
    """A retention method: how to build the model, an extra loss term, and any post-training work."""

    name = "base"
    train_expert_only: bool | None = None  # None keeps the value from the config

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def build_model(self, s1_ckpt: str | Path | None, cfg: RunConfig) -> torch.nn.Module:
        return load_policy_for_training(s1_ckpt, cfg, train_expert_only=self.train_expert_only)

    def extra_loss(self, policy: torch.nn.Module, batch: dict, step: int) -> tuple[torch.Tensor | None, dict]:
        return None, {}

    def post_train(self, ckpt_dir: str | Path, cfg: RunConfig) -> None:
        return None

    def checkpoint_kind(self) -> str:
        return "full"
