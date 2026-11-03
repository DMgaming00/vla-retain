"""M0 and M1: plain supervised fine-tuning."""

from __future__ import annotations

from vlaretain.methods.base import Method
from vlaretain.models.modules import LM_GROUPS


class ExpertOnlySFT(Method):
    name = "sft_expert"
    train_expert_only = True
    expected_groups = ("expert",)


class FullSFT(Method):
    name = "sft_full"
    train_expert_only = False
    expected_groups = ("connector", *LM_GROUPS, "expert")
