"""Preemption handling: a signal sets a flag, and the training loop saves and exits at the next step boundary."""

from __future__ import annotations

import signal


class PreemptionFlag:
    def __init__(self, signals: tuple[int, ...] = (signal.SIGUSR1, signal.SIGTERM)):
        self.triggered = False
        self.signum: int | None = None
        self._previous = {}
        for signum in signals:
            self._previous[signum] = signal.signal(signum, self._handle)

    def _handle(self, signum, _frame) -> None:
        self.triggered = True
        self.signum = signum

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous = {}
