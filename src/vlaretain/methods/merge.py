"""M4 and M8: weight merging. Merges live in memory at eval time and are never written to disk."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vlaretain.methods.sft import FullSFT
from vlaretain.models.modules import LM_GROUPS, module_group


@dataclass(frozen=True)
class AlphaSpec:
    """Interpolation weights theta(a) = (1-a) theta_S1 + a theta_FT, per module group.

    Forms: `global=0.3`, `lm=0.25,expert=0.5`, or `lm_early=0.25,lm_mid=0.5,lm_late=1,expert=1`.
    The connector follows the LM (or `lm_early`), since it feeds the LM. The vision encoder never
    trains, so its weight is irrelevant; we keep S1's copy.
    """

    weights: tuple[tuple[str, float], ...]

    @classmethod
    def parse(cls, text: str) -> AlphaSpec:
        weights = {}
        for part in text.split(","):
            key, _, value = part.partition("=")
            key = key.strip()
            alpha = float(value)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha for {key!r} is outside [0, 1]: {alpha}")
            weights[key] = alpha
        keys = set(weights)
        allowed = ({"global"}, {"lm", "expert"}, {*LM_GROUPS, "expert"})
        if keys not in allowed:
            raise ValueError(f"Alpha spec {text!r} must set exactly one of {[sorted(a) for a in allowed]}")
        return cls(tuple(sorted(weights.items())))

    @property
    def canonical(self) -> str:
        """The string form used in results keys."""
        return ",".join(f"{k}={v:.2f}" for k, v in self.weights)

    def alpha_for(self, group: str) -> float:
        weights = dict(self.weights)
        if "global" in weights:
            return weights["global"]
        if group == "vision":
            return 0.0
        if group == "expert":
            return weights["expert"]
        if "lm" in weights:
            return weights["lm"]
        return weights["lm_early"] if group == "connector" else weights[group]


NO_MERGE = "none"


def merge_state_dicts(
    s1: Mapping[str, torch.Tensor], ft: Mapping[str, torch.Tensor], spec: AlphaSpec, num_layers: int = 16
) -> dict[str, torch.Tensor]:
    """Interpolate two full state dicts. At alpha 0 or 1 the result equals one endpoint exactly."""
    if set(s1) != set(ft):
        raise ValueError(f"State dicts differ in keys: {sorted(set(s1) ^ set(ft))[:5]}")
    merged = {}
    for name, a in s1.items():
        b = ft[name]
        if not torch.is_floating_point(a) or "rotary_emb" in name:
            merged[name] = a.clone()
            continue
        alpha = spec.alpha_for(module_group(name, num_layers))
        if alpha == 0.0:
            merged[name] = a.clone()
        elif alpha == 1.0:
            merged[name] = b.clone()
        else:
            merged[name] = torch.lerp(a.to(b.dtype), b, alpha)
    return merged


def lora_merged_state_dict(
    base: Mapping[str, torch.Tensor], adapter: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    """Bake LoRA deltas into a base state dict: W + scale * B @ A, with scale = alpha / r.

    Adapter keys follow PEFT's saved form: `<module>.lora_A.weight` and `<module>.lora_B.weight`.
    """
    merged = {k: v.clone() for k, v in base.items()}
    modules = {k.removesuffix(".lora_A.weight") for k in adapter if k.endswith(".lora_A.weight")}
    for module in sorted(modules):
        key = module.removeprefix("base_model.model.") + ".weight"
        if key not in merged:
            raise KeyError(f"LoRA module {module!r} has no base weight {key!r}")
        delta = adapter[f"{module}.lora_B.weight"] @ adapter[f"{module}.lora_A.weight"]
        merged[key] = merged[key] + scale * delta.to(merged[key].dtype)
    return merged


def global_alpha_grid() -> list[str]:
    return [AlphaSpec.parse(f"global={i / 10:.1f}").canonical for i in range(11)]


def module_alpha_grid(values: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> list[str]:
    return [AlphaSpec.parse(f"lm={a},expert={b}").canonical for a in values for b in values]


def layer_alpha_grid(values: tuple[float, ...] = (0.25, 0.5, 1.0), expert: float = 1.0) -> list[str]:
    return [
        AlphaSpec.parse(f"lm_early={e},lm_mid={m},lm_late={la},expert={expert}").canonical
        for e in values for m in values for la in values
    ]  # fmt: skip


class MergeMethod(FullSFT):
    """Training is identical to M1. Merging happens only at eval time."""

    name = "merge"


class LayerMerge(MergeMethod):
    """M8 (optional): merging with a separate alpha for the early, mid, and late LM groups."""

    name = "layer_merge"
