"""Episode splits: Stage-1 and Stage-2 episode lists and seeded replay pools."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from vlaretain.utils.io import write_json

STAGE1_SUITES = ("libero_spatial", "libero_object", "libero_goal")
STAGE2_SUITES = ("libero_10",)
ALL_SUITES = STAGE1_SUITES + STAGE2_SUITES


def load_task_table() -> dict[str, list[dict]]:
    """Per-suite task tables, in the task-id order of the LIBERO benchmark."""
    text = resources.files("vlaretain.data").joinpath("libero_tasks.json").read_text()
    return json.loads(text)


def language_to_suite() -> dict[str, tuple[str, int]]:
    return {t["language"]: (suite, t["task_id"]) for suite, tasks in load_task_table().items() for t in tasks}


def build_episode_index(episode_tasks: dict[int, str]) -> dict[str, dict[str, list[int]]]:
    """Group episodes as {suite: {task_language: [episode_index, ...]}}. Unknown task strings raise."""
    lookup = language_to_suite()
    out: dict[str, dict[str, list[int]]] = {}
    for episode, language in sorted(episode_tasks.items()):
        if language not in lookup:
            raise KeyError(f"Episode {episode} has a task string outside the LIBERO tables: {language!r}")
        suite, _ = lookup[language]
        out.setdefault(suite, {}).setdefault(language, []).append(int(episode))
    return out


def episodes_for_suites(index: dict[str, dict[str, list[int]]], suites: tuple[str, ...] | list[str]) -> list[int]:
    return sorted(e for s in suites for eps in index[s].values() for e in eps)


def replay_pool(index: dict[str, dict[str, list[int]]], fraction: float, seed: int) -> list[int]:
    """Select round(fraction * N) Stage-1 episodes, spread across tasks by largest remainder.

    When the pool is smaller than the task count (2% of Stage-1 is), some tasks receive no episode.
    The selection depends only on the index, the fraction, and the seed.
    """
    by_task = {t: eps for s in STAGE1_SUITES for t, eps in sorted(index[s].items())}
    total = sum(len(e) for e in by_task.values())
    target = round(fraction * total)
    rng = np.random.default_rng([seed, round(fraction * 1_000_000)])
    tasks = list(by_task)
    order = rng.permutation(len(tasks))
    exact = np.array([fraction * len(by_task[t]) for t in tasks])
    counts = np.floor(exact).astype(int)
    remainder = exact - counts
    # Largest remainder first; the seeded permutation breaks ties.
    ranked = sorted(order, key=lambda i: -round(remainder[i], 9))
    for i in ranked[: target - counts.sum()]:
        counts[i] += 1
    chosen: list[int] = []
    for i, task in enumerate(tasks):
        task_rng = np.random.default_rng([seed, round(fraction * 1_000_000), i])
        chosen.extend(int(e) for e in task_rng.choice(by_task[task], size=counts[i], replace=False))
    return sorted(chosen)


def read_episode_tasks(repo_id: str, revision: str) -> dict[int, str]:
    """Read the episode-to-task map from the dataset's data parquet files, without videos."""
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["data/*"]))
    tasks = pd.read_parquet(hf_hub_download(repo_id, "meta/tasks.parquet", repo_type="dataset", revision=revision))
    names = {int(row.task_index): str(language) for language, row in tasks.iterrows()}
    out: dict[int, str] = {}
    for path in sorted(root.glob("data/*/*.parquet")):
        frame = pd.read_parquet(path, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
        for row in frame.itertuples():
            out[int(row.episode_index)] = names[int(row.task_index)]
    return out


def write_episode_index(path: str | Path, repo_id: str, revision: str) -> dict:
    index = build_episode_index(read_episode_tasks(repo_id, revision))
    payload = {"repo_id": repo_id, "revision": revision, "index": index}
    write_json(path, payload)
    return payload


def load_episode_index(path: str | Path) -> dict[str, dict[str, list[int]]]:
    with open(path) as f:
        return json.load(f)["index"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the episode index used by every split.")
    parser.add_argument("--out", default="splits/episodes.json")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    args = parser.parse_args()
    payload = write_episode_index(args.out, args.repo_id, args.revision)
    for suite, tasks in payload["index"].items():
        print(suite, len(tasks), "tasks", sum(len(v) for v in tasks.values()), "episodes")


if __name__ == "__main__":
    main()
