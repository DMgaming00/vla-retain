"""M0 and M1: plain supervised fine-tuning."""

from __future__ import annotations

from vlaretain.methods.base import Method


class ExpertOnlySFT(Method):
    name = "sft_expert"
    train_expert_only = True


class FullSFT(Method):
    name = "sft_full"
    train_expert_only = False
