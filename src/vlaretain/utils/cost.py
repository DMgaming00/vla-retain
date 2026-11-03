"""GPU-hour accounting. Totals are stored in checkpoints and survive preemption."""

from __future__ import annotations

import time

import torch


def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpu_type() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CostMeter:
    """Accumulates wall-clock hours scaled by GPU count. CPU-only runs count as one device."""

    def __init__(self, hours: float = 0.0, devices: int | None = None, clock=time.monotonic):
        self.hours = hours
        self.devices = devices if devices is not None else max(gpu_count(), 1)
        self._clock = clock
        self._start = clock()

    def tick(self) -> float:
        now = self._clock()
        self.hours += (now - self._start) / 3600.0 * self.devices
        self._start = now
        return self.hours
