"""M3 and M7: experience replay. Mixing happens in the data layer, driven by `data.replay_*` config fields."""

from __future__ import annotations

from vlaretain.config import RunConfig
from vlaretain.methods.sft import FullSFT


class ExperienceReplay(FullSFT):
    name = "replay"

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        if cfg.data.replay_fraction <= 0:
            raise ValueError(f"{self.name} needs data.replay_fraction > 0")


class SelfReplay(ExperienceReplay):
    """M7: the replay pool holds S1's own successful rollouts, written by `vlaretain.data.self_replay`."""

    name = "self_replay"
