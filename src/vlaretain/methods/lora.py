"""M2: LoRA fine-tuning on the attention and MLP linears of the LM and the expert."""

from __future__ import annotations

from pathlib import Path

import torch

from vlaretain.config import RunConfig
from vlaretain.methods.base import Method
from vlaretain.utils.io import write_json


def lora_target_regex() -> str:
    """PEFT matches this against full module names."""
    return (
        r".*vlm_with_expert\.(vlm\.model\.text_model|lm_expert)\.layers\.\d+\."
        r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    )


class LoraSFT(Method):
    name = "lora"
    train_expert_only = True  # the base model stays frozen; PEFT marks only adapter weights as trainable

    def build_model(self, s1_ckpt, cfg: RunConfig) -> torch.nn.Module:
        from peft import LoraConfig, inject_adapter_in_model

        policy = super().build_model(s1_ckpt, cfg)
        params = cfg.method.params
        config = LoraConfig(
            r=int(params.get("r", 16)),
            lora_alpha=int(params.get("alpha", 32)),
            lora_dropout=float(params.get("dropout", 0.05)),
            target_modules=lora_target_regex(),
        )
        for param in policy.parameters():
            param.requires_grad_(False)
        inject_adapter_in_model(config, policy)
        return policy

    def checkpoint_kind(self) -> str:
        return "lora"

    def post_train(self, ckpt_dir: str | Path, cfg: RunConfig) -> None:
        params = cfg.method.params
        write_json(
            Path(ckpt_dir) / "lora_summary.json",
            {"r": params.get("r", 16), "alpha": params.get("alpha", 32), "dropout": params.get("dropout", 0.05)},
        )
