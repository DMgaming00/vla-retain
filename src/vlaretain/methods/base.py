"""The Method interface. The training loop only ever calls these functions."""

from __future__ import annotations

from pathlib import Path

import torch

from vlaretain.config import RunConfig
from vlaretain.models import load as model_load
from vlaretain.models.modules import GROUPS, group_counts, num_lm_layers
from vlaretain.utils.io import write_json


class Method:
    """A retention method: how to build the model, an extra loss term, and any post-training work."""

    name = "base"
    train_expert_only: bool | None = None  # None keeps the value from the config
    expected_groups: tuple[str, ...] | None = None  # module groups that must have trainable parameters

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    # The three calls of the interface.
    def build_model(self, s1_ckpt: str | Path | None, cfg: RunConfig) -> torch.nn.Module:
        return model_load.load_policy_for_training(s1_ckpt, cfg, train_expert_only=self.train_expert_only)

    def extra_loss(self, policy: torch.nn.Module, batch: dict, step: int) -> tuple[torch.Tensor | None, dict]:
        return None, {}

    def post_train(self, ckpt_dir: str | Path, cfg: RunConfig) -> None:
        return None

    # Checkpoint format, which LoRA overrides.
    def save_weights(self, policy: torch.nn.Module, ckpt_dir: Path, meta: dict) -> None:
        policy.save_pretrained(ckpt_dir)
        write_json(ckpt_dir / model_load.CKPT_META, {"kind": "full", **meta})

    def load_weights(self, policy: torch.nn.Module, ckpt_dir: Path) -> None:
        from safetensors.torch import load_file

        state = load_file(ckpt_dir / "model.safetensors")
        result = policy.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in {ckpt_dir}: {result.unexpected_keys[:5]}")

    def check_trainable(self, policy: torch.nn.Module) -> dict[str, int]:
        """Fail early if the trainable set is not what this method claims. Returns counts per module group."""
        layers = num_lm_layers(policy)
        counts = group_counts(((n, p) for n, p in policy.named_parameters() if p.requires_grad), layers)
        if counts["vision"]:
            raise RuntimeError(f"{self.name}: the vision encoder has {counts['vision']} trainable parameters")
        if self.expected_groups is not None:
            for group in GROUPS:
                expected = group in self.expected_groups
                if expected and counts[group] == 0:
                    raise RuntimeError(f"{self.name}: expected {group} to train, but it has 0 trainable parameters")
                if not expected and counts[group] > 0:
                    raise RuntimeError(f"{self.name}: {group} should be frozen, but has {counts[group]} trainable")
        return counts


class Stage1(Method):
    """Stage-1 fine-tuning of the base model. The trainable set comes from configs/stage1.yaml."""

    name = "stage1"
