"""Stateless seeding. Each step's randomness depends only on (seed, step), which makes resume exact."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def derive_seed(*parts: object) -> int:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def seed_step(seed: int, step: int) -> None:
    torch.manual_seed(derive_seed("train", seed, step))
