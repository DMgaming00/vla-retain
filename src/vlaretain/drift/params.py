"""Relative weight drift per module group."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from vlaretain.models.modules import GROUPS, module_group


def relative_l2_per_group(
    state: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], num_layers: int = 16
) -> dict[str, float]:
    """||theta - theta_ref||_2 / ||theta_ref||_2 per group, over parameters present in both state dicts.

    Non-parameter buffers such as rotary frequencies are skipped, because `module_group` does not know them.
    """
    sq_diff = dict.fromkeys(GROUPS, 0.0)
    sq_ref = dict.fromkeys(GROUPS, 0.0)
    for name, ref in reference.items():
        if name not in state or not torch.is_floating_point(ref) or "rotary_emb" in name:
            continue
        group = module_group(name, num_layers)
        ref64 = ref.detach().to(dtype=torch.float64, device="cpu")
        cur64 = state[name].detach().to(dtype=torch.float64, device="cpu")
        sq_diff[group] += torch.sum((cur64 - ref64) ** 2).item()
        sq_ref[group] += torch.sum(ref64**2).item()
    return {g: (sq_diff[g] ** 0.5 / sq_ref[g] ** 0.5 if sq_ref[g] > 0 else 0.0) for g in GROUPS}
