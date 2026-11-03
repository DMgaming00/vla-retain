"""Maps every SmolVLA parameter to exactly one module group."""

from __future__ import annotations

import re
from collections.abc import Iterable

GROUPS = ("vision", "connector", "lm_early", "lm_mid", "lm_late", "expert")
LM_GROUPS = ("lm_early", "lm_mid", "lm_late")

_LM_LAYER = re.compile(r"text_model\.layers\.(\d+)\.")
_EXPERT_TOP = ("model.state_proj.", "model.action_in_proj.", "model.action_out_proj.",
               "model.action_time_mlp_in.", "model.action_time_mlp_out.")  # fmt: skip


def lm_layer_group(layer: int, num_layers: int = 16) -> str:
    """Thirds of the LM. For 16 layers: early 0-4, mid 5-10, late 11-15."""
    if not 0 <= layer < num_layers:
        raise ValueError(f"Layer {layer} is outside [0, {num_layers})")
    if layer < round(num_layers / 3):
        return "lm_early"
    if layer < round(2 * num_layers / 3):
        return "lm_mid"
    return "lm_late"


def module_group(name: str, num_layers: int = 16) -> str:
    """Return the group of a parameter name. Raises on anything unknown, so no parameter is dropped silently."""
    name = name.removeprefix("module.")
    if "vision_model." in name:
        return "vision"
    if ".connector." in name:
        return "connector"
    if match := _LM_LAYER.search(name):
        return lm_layer_group(int(match.group(1)), num_layers)
    if "text_model.embed_tokens." in name:
        return "lm_early"
    if "text_model.norm." in name or "vlm.lm_head." in name:
        return "lm_late"
    if ".lm_expert." in name or name.startswith(_EXPERT_TOP):
        return "expert"
    raise KeyError(f"Parameter {name!r} does not belong to any module group")


def group_counts(named_params: Iterable[tuple[str, object]], num_layers: int = 16) -> dict[str, int]:
    counts = dict.fromkeys(GROUPS, 0)
    for name, param in named_params:
        counts[module_group(name, num_layers)] += param.numel()
    return counts


def num_lm_layers(policy) -> int:
    return len(policy.model.vlm_with_expert.get_vlm_model().text_model.layers)
