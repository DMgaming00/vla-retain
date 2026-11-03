"""Dataclass configs loaded from YAML. The run id is a short hash of the resolved config."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ModelConfig:
    base: str = "lerobot/smolvla_base"
    base_revision: str = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True


@dataclass
class DataConfig:
    repo_id: str = "lerobot/libero"
    revision: str = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
    suites: list[str] = field(default_factory=lambda: ["libero_10"])
    splits_file: str = "splits/episodes.json"
    replay_fraction: float = 0.0  # share of each batch drawn from the replay pool
    replay_pool_fraction: float = 0.0  # share of Stage-1 episodes kept in the pool
    replay_source: str = "demos"  # demos | self_rollouts
    self_replay_root: str = ""
    num_workers: int = 8
    video_backend: str | None = None


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 3e-5
    min_lr: float = 2.5e-6
    warmup_steps: int = 1_000
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    save_every: int = 1_000
    milestones: list[int] = field(default_factory=lambda: [2_000, 5_000, 10_000, 20_000])
    log_every: int = 100
    use_amp: bool = False
    device: str = "cuda"


@dataclass
class MethodConfig:
    name: str = "sft_full"
    id: str = "M1"
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathsConfig:
    run_root: str = "runs"
    results_dir: str = "results"
    probe_file: str = "probes/probe_v1.pt"


@dataclass
class RunConfig:
    stage: int = 2
    seed: int = 0
    s1_run: str = ""  # run id of the Stage-1 reference model
    s1_step: int = 40_000
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    tuning: list[dict[str, Any]] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return run_id_of(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.run_root) / self.run_id

    @property
    def s1_ckpt(self) -> Path | None:
        if self.stage == 1 or not self.s1_run:
            return None
        return Path(self.paths.run_root) / self.s1_run / "checkpoints" / f"step_{self.s1_step:06d}"


# Fields that change where or how fast a run executes, but not what it computes.
_NON_SEMANTIC = {
    "paths": None,
    "tuning": None,
    "data": {"num_workers", "video_backend", "splits_file", "self_replay_root"},
    "train": {"log_every", "device", "save_every"},
    "method": {"enabled"},
}


def _coerce(value: Any, hint: Any) -> Any:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        if value is None:
            return None
        non_none = [a for a in get_args(hint) if a is not type(None)]
        return _coerce(value, non_none[0])
    if dataclasses.is_dataclass(hint):
        return from_dict(hint, value)
    if hint is float and value is not None:
        return float(value)
    if hint is int and value is not None:
        return int(value)
    if origin is tuple:
        return tuple(_coerce(v, a) for v, a in zip(value, get_args(hint), strict=True))
    if origin is list and get_args(hint):
        return [_coerce(v, get_args(hint)[0]) for v in value]
    return value


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict. Unknown keys raise, so a typo cannot silently change an experiment."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _coerce(v, hints[k]) for k, v in data.items()})


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "params":
            out[key] = deep_merge(out[key], value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def apply_override(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            value = float(value)  # YAML reads 1e-5 as a string
        except ValueError:
            pass
    return key, value


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> RunConfig:
    """Merge YAML files in order, then apply `key=value` overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            merged = deep_merge(merged, yaml.safe_load(f) or {})
    for text in overrides or []:
        key, value = parse_override(text)
        apply_override(merged, key, value)
    return from_dict(RunConfig, merged)


def resolved_dict(cfg: RunConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def semantic_dict(cfg: RunConfig) -> dict[str, Any]:
    """The part of the config that determines the run's outputs."""
    data = resolved_dict(cfg)
    for section, keys in _NON_SEMANTIC.items():
        if keys is None:
            data.pop(section, None)
        else:
            for key in keys:
                data[section].pop(key, None)
    return data


def run_id_of(cfg: RunConfig) -> str:
    canonical = json.dumps(semantic_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]
