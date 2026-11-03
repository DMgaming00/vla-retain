"""The frozen probe set: 256 frames from libero_10 demos and 256 from Stage-1 demos."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from vlaretain.data.splits import STAGE1_SUITES, STAGE2_SUITES
from vlaretain.utils.io import write_json

PROBE_VERSION = "probe_v1"
PROBE_SEED = 20261102
FRAMES_PER_SPLIT = 256
_ACTION_KEYS = ("action", "action_is_pad")


def allocate_per_task(n_tasks: int, n_frames: int) -> list[int]:
    base, extra = divmod(n_frames, n_tasks)
    return [base + (1 if i < extra else 0) for i in range(n_tasks)]


def select_probe_episodes(
    index: dict[str, dict[str, list[int]]], episode_lengths: dict[int, int], suites, n_frames: int, seed: int
) -> list[tuple[int, int]]:
    """Choose (episode, frame) pairs, evenly across tasks, with at most one frame per episode where possible."""
    tasks = sorted((t, eps) for s in suites for t, eps in index[s].items())
    per_task = allocate_per_task(len(tasks), n_frames)
    chosen = []
    for task_index, ((_, episodes), count) in enumerate(zip(tasks, per_task, strict=True)):
        rng = np.random.default_rng([seed, task_index])
        order = rng.permutation(sorted(episodes))
        for i in range(count):
            episode = int(order[i % len(order)])
            frame = int(rng.integers(0, episode_lengths[episode]))
            chosen.append((episode, frame))
    return sorted(set(chosen)) if len(set(chosen)) == n_frames else chosen


def probe_manifest(index, episode_lengths, seed: int = PROBE_SEED, n_frames: int = FRAMES_PER_SPLIT) -> dict:
    return {
        "version": PROBE_VERSION,
        "seed": seed,
        "new": select_probe_episodes(index, episode_lengths, STAGE2_SUITES, n_frames, seed),
        "old": select_probe_episodes(index, episode_lengths, STAGE1_SUITES, n_frames, seed + 1),
    }


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def collate_frames(items: list[dict]) -> dict:
    """Stack raw dataset items, storing images as uint8 to keep the file small. Temporal dims are squeezed."""
    out: dict = {"task": [item["task"] for item in items]}
    for key in items[0]:
        if key == "task" or not torch.is_tensor(items[0][key]):
            continue
        stacked = torch.stack([item[key] for item in items])
        if key.startswith("observation.images"):
            if stacked.ndim == 5:
                stacked = stacked[:, -1]
            stacked = _to_uint8(stacked)
        elif key == "observation.state" and stacked.ndim == 3:
            stacked = stacked[:, -1]
        elif key not in _ACTION_KEYS:
            continue
        out[key] = stacked
    return out


def freeze_probe(path: str | Path, new: dict, old: dict, manifest: dict) -> str:
    """Write the probe file once. A frozen probe is never overwritten, because every metric refers to it."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen. Delete it by hand only if you mean to change every drift result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"version": manifest["version"], "new": new, "old": old}, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:16]
    tmp.rename(path)
    write_json(path.with_suffix(".json"), {**manifest, "sha256_16": digest})
    return digest


def load_probe(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def raw_probe_batches(split: dict, batch_size: int) -> list[dict]:
    """Slice a probe split into raw batches, with images back in float [0, 1], ready for the preprocessor."""
    n = len(split["task"])
    batches = []
    for start in range(0, n, batch_size):
        batch: dict = {}
        for key, value in split.items():
            piece = value[start : start + batch_size]
            if key.startswith("observation.images"):
                piece = piece.to(torch.float32) / 255.0
            batch[key] = piece
        batches.append(batch)
    return batches


# UNVERIFIED: needs GPU/LIBERO
def build_probe_file(
    out: str | Path, splits_file: str | Path, repo_id: str, revision: str, chunk_size: int = 50
) -> str:
    """Read the chosen frames from the dataset and freeze them. Needs the dataset on disk, but not a GPU."""
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    from vlaretain.data.splits import load_episode_index

    index = load_episode_index(splits_file)
    meta = LeRobotDatasetMetadata(repo_id, revision=revision)
    episodes_table = meta.episodes
    lengths = {int(e): int(n) for e, n in zip(episodes_table["episode_index"], episodes_table["length"], strict=True)}
    starts = {
        int(e): int(n) for e, n in zip(episodes_table["episode_index"], episodes_table["dataset_from_index"], strict=True)
    }
    manifest = probe_manifest(index, lengths)
    deltas = {"action": [i / meta.fps for i in range(chunk_size)]}
    dataset = LeRobotDataset(repo_id, delta_timestamps=deltas, revision=revision)
    new = collate_frames([dataset[starts[e] + f] for e, f in manifest["new"]])
    old = collate_frames([dataset[starts[e] + f] for e, f in manifest["old"]])
    return freeze_probe(out, new, old, manifest)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and freeze the probe set.")
    parser.add_argument("--out", default="probes/probe_v1.pt")
    parser.add_argument("--splits", default="splits/episodes.json")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    if args.dry_run:
        print(f"would freeze {FRAMES_PER_SPLIT} new-task and {FRAMES_PER_SPLIT} old-task frames into {args.out}")
        return
    print(build_probe_file(args.out, args.splits, args.repo_id, args.revision))


if __name__ == "__main__":
    main()
