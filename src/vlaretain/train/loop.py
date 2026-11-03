"""The training loop. It never branches on the method name: methods act only through their interface."""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vlaretain.config import RunConfig
from vlaretain.data.mix import MixedBatchSampler
from vlaretain.utils.cost import CostMeter, gpu_type
from vlaretain.utils.io import append_jsonl, write_run_meta
from vlaretain.utils.seed import seed_step
from vlaretain.utils.signals import PreemptionFlag

log = logging.getLogger(__name__)

LATEST = "latest"
TRAINING_STATE = "training_state.pt"


@dataclass
class TrainData:
    """A map-style dataset laid out as [target frames, replay frames], and how to prepare its batches."""

    dataset: Any
    n_target: int
    n_replay: int
    preprocessor: Callable[[dict], dict]
    save_processors: Callable[[Path], None] | None = None


@dataclass
class TrainResult:
    status: str  # finished | preempted | stopped | already_done
    step: int
    gpu_hours: float
    trainable: dict[str, int] | None = None


def lr_lambda(step: int, *, peak: float, floor: float, warmup: int, total: int) -> float:
    """Linear warmup to the peak learning rate, then cosine decay to the floor. Returns a factor of the peak."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    floor_factor = min(floor / peak, 1.0)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor_factor + (1 - floor_factor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(policy: torch.nn.Module, cfg: RunConfig) -> tuple[torch.optim.Optimizer, Any]:
    params = [p for p in policy.parameters() if p.requires_grad]
    t = cfg.train
    optimizer = torch.optim.AdamW(params, lr=t.lr, betas=t.betas, eps=t.eps, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_lambda(s, peak=t.lr, floor=t.min_lr, warmup=t.warmup_steps, total=t.steps),
    )
    return optimizer, scheduler


def _replace_dir(tmp: Path, final: Path) -> None:
    """Swap a finished directory into place, so a kill never leaves a half-written checkpoint as the latest one."""
    old = final.with_name(final.name + ".old")
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def save_checkpoint(
    method, policy, data: TrainData, ckpt_root: Path, name: str, step: int, gpu_hours: float, training_state: dict | None
) -> Path:
    final = ckpt_root / name
    tmp = ckpt_root / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    method.save_weights(policy, tmp, {"step": step, "gpu_hours_train": gpu_hours})
    if data.save_processors is not None:
        data.save_processors(tmp)
    if training_state is not None:
        torch.save(training_state, tmp / TRAINING_STATE)
    _replace_dir(tmp, final)
    return final


def find_resume_dir(ckpt_root: Path) -> Path | None:
    for name in (LATEST, f"{LATEST}.old"):
        candidate = ckpt_root / name
        if (candidate / TRAINING_STATE).exists():
            return candidate
    return None


def milestone_name(step: int) -> str:
    return f"step_{step:06d}"


def train(
    cfg: RunConfig,
    method,
    data: TrainData,
    *,
    stop_after: int | None = None,
    on_step: Callable[[int], None] | None = None,
) -> TrainResult:
    """Train to cfg.train.steps, resuming from the latest checkpoint if one exists.

    `stop_after` stops early, as a test or a time-boxed run would. A signal saves and exits cleanly.
    """
    run_dir = cfg.run_dir
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "DONE"
    if done_marker.exists():
        return TrainResult("already_done", cfg.train.steps, _read_done_hours(done_marker))

    write_run_meta(run_dir, cfg, gpu_type())
    policy = method.build_model(cfg.s1_ckpt, cfg)
    trainable = method.check_trainable(policy)
    log.info("Trainable parameters per group: %s", trainable)
    optimizer, scheduler = make_optimizer(policy, cfg)

    step = 0
    meter = CostMeter()
    resume_dir = find_resume_dir(ckpt_root)
    if resume_dir is not None:
        method.load_weights(policy, resume_dir)
        state = torch.load(resume_dir / TRAINING_STATE, map_location=cfg.train.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        meter = CostMeter(hours=state["gpu_hours"])
        log.info("Resumed from %s at step %d", resume_dir, step)

    sampler = MixedBatchSampler(
        data.n_target, data.n_replay, cfg.train.batch_size, cfg.data.replay_fraction, cfg.seed,
        total_steps=cfg.train.steps, start_step=step,
    )  # fmt: skip
    loader = DataLoader(
        data.dataset,
        batch_sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.train.device == "cuda",
        persistent_workers=cfg.data.num_workers > 0,
    )

    flag = PreemptionFlag()
    log_path = run_dir / "train_log.jsonl"
    device_type = cfg.train.device.split(":")[0]
    amp = torch.autocast(device_type, dtype=torch.bfloat16) if cfg.train.use_amp else nullcontext()
    policy.train()

    def snapshot() -> dict:
        return {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "gpu_hours": meter.tick(),
        }

    status = "finished"
    try:
        for raw_batch in loader:
            seed_step(cfg.seed, step)
            batch = data.preprocessor(raw_batch)
            with amp:
                loss, info = policy.forward(batch)
                extra, extra_info = method.extra_loss(policy, batch, step)
                total = loss if extra is None else loss + extra
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], cfg.train.grad_clip_norm
            )
            optimizer.step()
            scheduler.step()
            step += 1
            hours = meter.tick()

            if step % cfg.train.log_every == 0 or step == cfg.train.steps or step == stop_after:
                row = {
                    "step": step, "loss": loss.item(), "total_loss": total.item(), "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0], "gpu_hours": hours, **extra_info,
                }  # fmt: skip
                append_jsonl(log_path, [row])
                log.info("step %d %s", step, json.dumps(row))

            if on_step is not None:
                on_step(step)

            if step in cfg.train.milestones:
                save_checkpoint(method, policy, data, ckpt_root, milestone_name(step), step, hours, None)
            interrupted = flag.triggered or step == stop_after
            if step % cfg.train.save_every == 0 or interrupted:
                save_checkpoint(method, policy, data, ckpt_root, LATEST, step, hours, snapshot())
            if flag.triggered:
                log.warning("Signal %s received; saved at step %d and exiting", flag.signum, step)
                status = "preempted"
                break
            if step == stop_after:
                status = "stopped"
                break
    finally:
        flag.restore()
        del loader

    hours = meter.tick()
    if status == "finished":
        final = save_checkpoint(method, policy, data, ckpt_root, milestone_name(step), step, hours, None)
        method.post_train(final, cfg)
        done_marker.write_text(json.dumps({"step": step, "gpu_hours": hours}) + "\n")
    return TrainResult(status, step, hours, trainable)


def _read_done_hours(marker: Path) -> float:
    try:
        return float(json.loads(marker.read_text())["gpu_hours"])
    except (ValueError, KeyError):
        return 0.0


def _subset_dataset(cfg: RunConfig, policy_config, episodes: list[int] | None, repo_id=None, root=None, revision=None):
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    repo_id = repo_id or cfg.data.repo_id
    revision = revision if root is not None else (revision or cfg.data.revision)
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    deltas = resolve_delta_timestamps(policy_config, meta)
    return LeRobotDataset(
        repo_id, root=root, episodes=episodes, delta_timestamps=deltas, revision=revision,
        video_backend=cfg.data.video_backend,
    )  # fmt: skip


def make_processors(cfg: RunConfig, policy, device: str | None = None):
    """Stage-1 builds fresh processors from dataset-level statistics; Stage-2 reuses S1's unchanged."""
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors

    device = device or cfg.train.device
    if cfg.s1_ckpt is not None:
        return make_pre_post_processors(
            policy.config,
            pretrained_path=str(cfg.s1_ckpt),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    stats = LeRobotDatasetMetadata(cfg.data.repo_id, revision=cfg.data.revision).stats
    return make_pre_post_processors(policy.config, dataset_stats=stats)


def build_train_data(cfg: RunConfig, policy_config, processors) -> TrainData:
    """Build [target, replay] from the episode index and the data.* config fields."""
    from torch.utils.data import ConcatDataset

    from vlaretain.data.splits import episodes_for_suites, load_episode_index, replay_pool

    index = load_episode_index(cfg.data.splits_file)
    target = _subset_dataset(cfg, policy_config, episodes_for_suites(index, cfg.data.suites))
    parts = [target]
    n_replay = 0
    if cfg.data.replay_fraction > 0:
        if cfg.data.replay_source == "self_rollouts":
            replay = _subset_dataset(
                cfg, policy_config, None, repo_id="local/self_replay", root=cfg.data.self_replay_root
            )
        else:
            pool = replay_pool(index, cfg.data.replay_pool_fraction, cfg.seed)
            replay = _subset_dataset(cfg, policy_config, pool)
        parts.append(replay)
        n_replay = len(replay)
    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    preprocessor, postprocessor = processors

    def save(path: Path) -> None:
        preprocessor.save_pretrained(path)
        postprocessor.save_pretrained(path)

    return TrainData(dataset, len(target), n_replay, preprocessor, save)
