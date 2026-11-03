"""Method registry. Names here are the `method.name` values in configs/methods/*.yaml."""

from __future__ import annotations

from vlaretain.config import RunConfig


def get_method(cfg: RunConfig):
    from vlaretain.methods.anchor import FeatureAnchor
    from vlaretain.methods.base import Stage1
    from vlaretain.methods.l2sp import L2SP
    from vlaretain.methods.lora import LoraSFT
    from vlaretain.methods.merge import LayerMerge, MergeMethod
    from vlaretain.methods.replay import ExperienceReplay, SelfReplay
    from vlaretain.methods.sft import ExpertOnlySFT, FullSFT

    registry = {
        cls.name: cls
        for cls in (Stage1, ExpertOnlySFT, FullSFT, LoraSFT, ExperienceReplay, SelfReplay, MergeMethod, LayerMerge,
                    L2SP, FeatureAnchor)
    }  # fmt: skip
    if cfg.method.name not in registry:
        raise KeyError(f"Unknown method {cfg.method.name!r}. Known: {sorted(registry)}")
    return registry[cfg.method.name](cfg)
