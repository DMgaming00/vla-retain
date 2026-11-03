#!/usr/bin/env python
"""Cluster smoke test: fetch artifacts, check EGL and CUDA, then time rollouts at 1, 4, and 8 parallel envs.

The seconds-per-episode figure sets the eval budget, because evaluation dominates this project's compute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

SMOKE_CKPT = "HuggingFaceVLA/smolvla_libero"
SMOKE_REV = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
ROLLOUTS_PER_CHECKPOINT = 750


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true", help="fetch models and data, then exit (login node)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--env-counts", default="1,4,8")
    parser.add_argument("--episodes", type=int, default=8, help="episodes per env count")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def download() -> None:
    from huggingface_hub import snapshot_download

    from vlaretain.config import DataConfig, ModelConfig

    data, model = DataConfig(), ModelConfig()
    print(snapshot_download(model.base, revision=model.base_revision))
    print(snapshot_download(SMOKE_CKPT, revision=SMOKE_REV))
    print(snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", allow_patterns=["*.json", "*.txt", "*.jinja"]))
    print(snapshot_download(data.repo_id, repo_type="dataset", revision=data.revision))


# UNVERIFIED: needs GPU/LIBERO
def timing_table(args) -> list[tuple[int, float, float]]:
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from vlaretain.eval.rollout import EpisodeSpec, run_episodes
    from vlaretain.eval.suites import EVAL_SEED

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this node")
    policy = SmolVLAPolicy.from_pretrained(SMOKE_CKPT, revision=SMOKE_REV).to(args.device, dtype=torch.float32).eval()
    processors = make_pre_post_processors(
        policy.config, pretrained_path=SMOKE_CKPT, pretrained_revision=SMOKE_REV,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )  # fmt: skip
    rows = []
    for n_envs in [int(x) for x in args.env_counts.split(",")]:
        specs = [EpisodeSpec(args.suite, args.task_id, i, EVAL_SEED) for i in range(args.episodes)]
        start = time.monotonic()
        successes = run_episodes(policy, processors, specs, n_envs=n_envs, n_action_steps=args.n_action_steps)
        elapsed = time.monotonic() - start
        rows.append((n_envs, elapsed / len(specs), float(successes.mean())))
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"checkpoint: {SMOKE_CKPT}@{SMOKE_REV}")
        print(f"timing: {args.episodes} episodes of {args.suite}/{args.task_id} at envs={args.env_counts}")
        print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(unset; set it to egl)')}")
        return 0
    if args.download_only:
        download()
        return 0
    if os.environ.get("MUJOCO_GL") != "egl":
        print("Warning: MUJOCO_GL is not egl; rendering may fail on headless nodes")
    rows = timing_table(args)
    print(f"\n{'envs':>5} {'sec/episode':>12} {'episodes/hour':>14} {'hours/checkpoint':>17} {'SR':>6}")
    for n_envs, seconds, rate in rows:
        hours = seconds * ROLLOUTS_PER_CHECKPOINT / 3600
        print(f"{n_envs:>5} {seconds:>12.1f} {3600 / seconds:>14.0f} {hours:>17.2f} {rate:>6.2f}")
    best = min(rows, key=lambda r: r[1])
    print(f"\nBest: {best[0]} envs. One full eval of {ROLLOUTS_PER_CHECKPOINT} rollouts takes about "
          f"{best[1] * ROLLOUTS_PER_CHECKPOINT / 3600:.2f} GPU-hours.")  # fmt: skip
    print("The SR column checks that the pipeline works, on a small sample; it is not a result.")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
