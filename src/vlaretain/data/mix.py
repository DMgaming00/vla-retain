"""A seeded, stateless batch sampler that mixes target and replay frames.

Every batch is a pure function of (seed, step), so a resumed run sees exactly the batches it would have seen.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np


class StatelessPool:
    """Without-replacement draws from range(size), reshuffled per epoch, addressable by absolute position."""

    def __init__(self, size: int, seed: int, tag: str):
        self.size = size
        self.seed = seed
        self.tag = 0 if tag == "target" else 1
        self._epoch = -1
        self._perm: np.ndarray | None = None

    def at(self, position: int) -> int:
        epoch, offset = divmod(position, self.size)
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.seed, self.tag, epoch]).permutation(self.size)
            self._epoch = epoch
        return int(self._perm[offset])


class MixedBatchSampler:
    """Yields lists of indices into ConcatDataset([target, replay]).

    Replay indices are offset by `n_target`. Batch sizes are integers, so the replay count alternates by error
    diffusion: with batch 32 and fraction 0.2 a batch has 6 or 7 replay frames, and the long-run share is exactly 0.2.
    """

    def __init__(
        self, n_target: int, n_replay: int, batch_size: int, replay_fraction: float, seed: int,
        total_steps: int, start_step: int = 0,
    ):  # fmt: skip
        if replay_fraction > 0 and n_replay == 0:
            raise ValueError("replay_fraction > 0 needs a non-empty replay pool")
        if not 0 <= replay_fraction < 1:
            raise ValueError(f"replay_fraction must be in [0, 1), got {replay_fraction}")
        self.n_target = n_target
        self.batch_size = batch_size
        self.replay_fraction = replay_fraction
        self.total_steps = total_steps
        self.start_step = start_step
        self.target = StatelessPool(n_target, seed, "target")
        self.replay = StatelessPool(n_replay, seed, "replay") if n_replay else None

    def _replay_seen(self, step: int) -> int:
        """Replay frames drawn before `step`. Integer arithmetic keeps this exact over long runs."""
        scaled = round(self.replay_fraction * 1_000_000)
        return math.floor(scaled * self.batch_size * step / 1_000_000)

    def batch(self, step: int) -> list[int]:
        replay_start = self._replay_seen(step)
        n_replay = self._replay_seen(step + 1) - replay_start
        target_start = step * self.batch_size - replay_start
        indices = [self.target.at(target_start + i) for i in range(self.batch_size - n_replay)]
        if n_replay:
            indices += [self.n_target + self.replay.at(replay_start + i) for i in range(n_replay)]
        return indices

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.total_steps):
            yield self.batch(step)

    def __len__(self) -> int:
        return self.total_steps - self.start_step
