"""Reads the JSONL results into tables. Every figure and table starts here, and nothing is ever filled in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlaretain.utils.io import read_results

N_RETENTION_TASKS = 30
N_TARGET_TASKS = 10
STAGE1_METHOD = "stage1"
MISSING = "missing"

# Display name, color, and marker per method. Color follows the method and never its rank, and a marker always
# accompanies it, so no method is told apart by color alone. M6-M8 are optional, and reuse hues with new markers.
METHOD_STYLE = {
    "sft_expert": ("M0 Expert-only SFT", "#2a78d6", "o"),
    "sft_full": ("M1 Full SFT", "#eb6834", "s"),
    "lora": ("M2 LoRA", "#1baf7a", "^"),
    "replay": ("M3 Replay", "#eda100", "D"),
    "merge_global": ("M4 Merge (global)", "#e87ba4", "v"),
    "merge_module": ("M4 Merge (module)", "#008300", "P"),
    "l2sp": ("M5 L2-SP", "#6250d6", "X"),
    "anchor": ("M6 Anchoring", "#e34948", "*"),
    "self_replay": ("M7 Self-replay", "#2a78d6", "<"),
    "layer_merge": ("M8 Layer merge", "#eb6834", ">"),
}
METHOD_ORDER = list(METHOD_STYLE)


def method_name(method: str) -> str:
    return METHOD_STYLE[method][0] if method in METHOD_STYLE else method


def load_tables(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split results into (eval rows, drift rows). Selection bookkeeping rows are ignored."""
    rows = read_results(results_dir)
    frame = pd.DataFrame(rows)
    if frame.empty or "eval_set" not in frame:
        return pd.DataFrame(), pd.DataFrame()
    frame = frame[frame["eval_set"].notna()]
    return frame[frame["eval_set"] != "drift"].copy(), frame[frame["eval_set"] == "drift"].copy()


def s1_task_rates(eval_df: pd.DataFrame, eval_set: str = "retention") -> pd.Series:
    """S1's success rate per task at its last step."""
    rows = eval_df[(eval_df["method"] == STAGE1_METHOD) & (eval_df["eval_set"] == eval_set)]
    if rows.empty:
        return pd.Series(dtype=float)
    rows = rows[rows["step"] == rows["step"].max()]
    return rows.drop_duplicates("task_id").set_index("task_id")["sr"]


def forgetting(run_rows: pd.DataFrame, s1_rates: pd.Series) -> float:
    """Mean over shared tasks of S1 SR minus this checkpoint's SR. Positive means skills were lost."""
    rates = run_rows.drop_duplicates("task_id").set_index("task_id")["sr"]
    shared = rates.index.intersection(s1_rates.index)
    return float((s1_rates[shared] - rates[shared]).mean()) if len(shared) else float("nan")


def checkpoint_table(
    eval_df: pd.DataFrame, retention_set: str = "retention", target_set: str = "target",
    n_retention: int = N_RETENTION_TASKS, n_target: int = N_TARGET_TASKS,
) -> pd.DataFrame:  # fmt: skip
    """One row per evaluated checkpoint, only where both eval sets are complete. Partial evals are left out."""
    columns = [
        "run_id", "method", "seed", "step", "alpha_spec", "target_sr", "retention_sr", "forgetting",
        "gpu_hours_train", "compute_hours",
    ]  # fmt: skip
    if eval_df.empty:
        return pd.DataFrame(columns=columns)
    s1_rates = s1_task_rates(eval_df, retention_set)
    records = []
    stage2 = eval_df[eval_df["method"] != STAGE1_METHOD]
    for key, rows in stage2.groupby(["run_id", "method", "seed", "step", "alpha_spec"], dropna=False):
        retention = rows[rows["eval_set"] == retention_set].drop_duplicates("task_id")
        target = rows[rows["eval_set"] == target_set].drop_duplicates("task_id")
        if len(retention) < n_retention or len(target) < n_target:
            continue
        run_id, method, seed, step, alpha = key
        hours = rows["gpu_hours_train"].dropna()
        train_hours = float(hours.iloc[0]) if len(hours) else float("nan")
        # A merge's compute also includes its validation sweep, over every alpha of its label.
        sweep = eval_df[
            (eval_df["run_id"] == run_id) & (eval_df["method"] == method) & (eval_df["alpha_spec"] != "none")
            & eval_df["eval_set"].astype(str).str.startswith("validation_")
        ]["gpu_hours_eval"].sum()  # fmt: skip
        records.append({
            "run_id": run_id, "method": method, "seed": seed, "step": step, "alpha_spec": alpha,
            "target_sr": float(target["n_success"].sum() / target["n_episodes"].sum()),
            "retention_sr": float(retention["n_success"].sum() / retention["n_episodes"].sum()),
            "forgetting": forgetting(retention, s1_rates),
            "gpu_hours_train": train_hours,
            "compute_hours": train_hours + float(sweep),
        })  # fmt: skip
    return pd.DataFrame(records, columns=columns)


def drift_table(drift_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, step, alpha), with a column per drift or divergence summary."""
    if drift_df.empty:
        return pd.DataFrame()
    records = []
    for (run_id, method, seed, step, alpha), rows in drift_df.groupby(
        ["run_id", "method", "seed", "step", "alpha_spec"], dropna=False
    ):
        record = {"run_id": run_id, "method": method, "seed": seed, "step": step, "alpha_spec": alpha}
        weights = rows[(rows["metric"] == "relative_l2") & (rows["module"] != "vision")]
        record["weight_drift"] = float(weights["value"].mean()) if len(weights) else np.nan
        for split in ("new", "old"):
            part = rows[rows["probe_split"] == split]
            for metric in ("velocity_distance", "action_chunk_distance", "action_mmd"):
                values = part[part["metric"] == metric]["value"]
                record[f"{split}_{metric}"] = float(values.iloc[0]) if len(values) else np.nan
            cka = part[part["metric"] == "cka_distance"]["value"]
            record[f"{split}_feature_drift"] = float(cka.mean()) if len(cka) else np.nan
        records.append(record)
    return pd.DataFrame(records)
