"""M2: LoRA fine-tuning on the attention and MLP linears of the LM and the expert."""

from __future__ import annotations

from pathlib import Path

import torch

from vlaretain.config import RunConfig
from vlaretain.methods.base import Method
from vlaretain.models import load as model_load
from vlaretain.models.modules import LM_GROUPS
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
    expected_groups = (*LM_GROUPS, "expert")

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

    def save_weights(self, policy: torch.nn.Module, ckpt_dir: Path, meta: dict) -> None:
        """Save the adapter only. The evaluator bakes it onto S1 with `lora_merged_state_dict`."""
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        adapter = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(policy).items()}
        save_file(adapter, ckpt_dir / "adapter_model.safetensors")
        policy.config._save_pretrained(ckpt_dir)
        params = self.cfg.method.params
        write_json(
            ckpt_dir / model_load.CKPT_META,
            {
                "kind": "lora",
                "lora_r": int(params.get("r", 16)),
                "lora_alpha": int(params.get("alpha", 32)),
                "base": str(self.cfg.s1_ckpt) if self.cfg.s1_ckpt else None,
                **meta,
            },
        )

    def load_weights(self, policy: torch.nn.Module, ckpt_dir: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        result = set_peft_model_state_dict(policy, load_file(ckpt_dir / "adapter_model.safetensors"))
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:5]}")

    def post_train(self, ckpt_dir: str | Path, cfg: RunConfig) -> None:
        params = cfg.method.params
        write_json(
            Path(ckpt_dir) / "lora_summary.json",
            {"r": params.get("r", 16), "alpha": params.get("alpha", 32), "dropout": params.get("dropout", 0.05)},
        )
