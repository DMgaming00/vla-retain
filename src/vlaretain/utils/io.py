"""Append-only JSONL results, atomic JSON writes, and run metadata."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vlaretain.eval.stats import wilson_interval

LEROBOT_PINNED_SHA = "7e241bd630a3719a56157a497ce5d08f244784f1"

RESULT_FIELDS = (
    "run_id", "method", "seed", "step", "alpha_spec", "eval_set", "task_id", "n_episodes", "n_success", "sr",
    "wilson_lo", "wilson_hi", "gpu_hours_train", "gpu_hours_eval", "git_sha", "lerobot_sha", "timestamp",
)  # fmt: skip
DRIFT_FIELDS = ("probe_split", "module", "layer", "metric", "value")
EVAL_KEY = ("run_id", "step", "alpha_spec", "eval_set", "task_id")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Append rows under an exclusive lock, so concurrent array tasks do not interleave lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a killed job; the skip logic re-runs that work
    return rows


def read_results(results_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def completed_eval_keys(results_dir: str | Path) -> set[tuple]:
    return {tuple(r.get(k) for k in EVAL_KEY) for r in read_results(results_dir) if "n_success" in r}


def make_eval_row(
    *, run_id: str, method: str, seed: int, step: int, alpha_spec: str, eval_set: str, task_id: str,
    n_episodes: int, n_success: int, gpu_hours_train: float | None, gpu_hours_eval: float,
    git_sha: str, lerobot_sha: str,
) -> dict[str, Any]:  # fmt: skip
    lo, hi = wilson_interval(n_success, n_episodes)
    return {
        "run_id": run_id, "method": method, "seed": seed, "step": step, "alpha_spec": alpha_spec,
        "eval_set": eval_set, "task_id": task_id, "n_episodes": n_episodes, "n_success": n_success,
        "sr": n_success / n_episodes if n_episodes else None, "wilson_lo": lo, "wilson_hi": hi,
        "gpu_hours_train": gpu_hours_train, "gpu_hours_eval": gpu_hours_eval, "git_sha": git_sha,
        "lerobot_sha": lerobot_sha, "timestamp": datetime.now(UTC).isoformat(),
    }  # fmt: skip


def make_drift_row(
    *, run_id: str, method: str, seed: int, step: int, alpha_spec: str, probe_split: str, module: str,
    layer: int | None, metric: str, value: float, gpu_hours_train: float | None, gpu_hours_eval: float,
    git_sha: str, lerobot_sha: str,
) -> dict[str, Any]:  # fmt: skip
    return {
        "run_id": run_id, "method": method, "seed": seed, "step": step, "alpha_spec": alpha_spec,
        "eval_set": "drift", "task_id": None, "n_episodes": None, "n_success": None, "sr": None,
        "wilson_lo": None, "wilson_hi": None, "gpu_hours_train": gpu_hours_train,
        "gpu_hours_eval": gpu_hours_eval, "git_sha": git_sha, "lerobot_sha": lerobot_sha,
        "timestamp": datetime.now(UTC).isoformat(), "probe_split": probe_split, "module": module,
        "layer": layer, "metric": metric, "value": value,
    }  # fmt: skip


def git_sha(repo: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo or Path(__file__).resolve().parents[3],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def lerobot_sha() -> str:
    """The commit LeRobot was actually installed from, falling back to the pinned value."""
    try:
        raw = importlib.metadata.distribution("lerobot").read_text("direct_url.json")
        if raw:
            return json.loads(raw).get("vcs_info", {}).get("commit_id", LEROBOT_PINNED_SHA)
    except importlib.metadata.PackageNotFoundError:
        pass
    return LEROBOT_PINNED_SHA


def pip_freeze_hash() -> str:
    lines = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def write_run_meta(run_dir: str | Path, cfg: Any, gpu: str) -> dict[str, Any]:
    from vlaretain.config import run_id_of

    meta = {
        "run_id": run_id_of(cfg),
        "git_sha": git_sha(),
        "lerobot_sha": lerobot_sha(),
        "pip_freeze_hash": pip_freeze_hash(),
        "gpu_type": gpu,
        "seed": cfg.seed,
        "config": dataclasses.asdict(cfg),
        "written_at": datetime.now(UTC).isoformat(),
    }
    write_json(Path(run_dir) / "run_meta.json", meta)
    return meta
