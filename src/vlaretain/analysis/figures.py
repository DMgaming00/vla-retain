"""Fig 2, Fig 3, and Table 1. All values come from the results tables, and absent values print as missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlaretain.analysis.common import METHOD_ORDER, METHOD_STYLE, MISSING, N_RETENTION_TASKS, method_name
from vlaretain.analysis.correlate import PRIMARY
from vlaretain.analysis.pareto import style_axis, write_missing_figure
from vlaretain.models.modules import GROUPS


def plot_forgetting_vs_divergence(joined: pd.DataFrame, path: str | Path, suffix: str = "") -> None:
    """Fig 2: forgetting against new-state divergence, one mark per checkpoint."""
    import matplotlib.pyplot as plt

    path = Path(path)
    title = f"Forgetting vs new-state divergence{suffix}"
    if joined.empty or PRIMARY not in joined or joined[PRIMARY].isna().all():
        write_missing_figure(path, title, "missing: no checkpoint has both retention and drift results")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for method in METHOD_ORDER:
        part = joined[joined["method"] == method]
        if part.empty:
            continue
        _, color, marker = METHOD_STYLE[method]
        ax.scatter(
            part[PRIMARY], part["forgetting"] * 100, color=color, marker=marker, s=44, linewidths=1.0,
            edgecolors="#fcfcfb", label=method_name(method), zorder=3,
        )  # fmt: skip
    style_axis(ax)
    ax.set_xscale("log" if (joined[PRIMARY] > 0).all() else "linear")
    ax.set_xlabel("Velocity-field distance to S1, new-task probe states", fontsize=9, color="#52514e")
    ax.set_ylabel("Forgetting (S1 retention SR minus SR, points)", fontsize=9, color="#52514e")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def plot_module_drift(drift_df: pd.DataFrame, path: str | Path, suffix: str = "", probe_split: str = "new") -> None:
    """Fig 3: relative weight L2 per module group, and 1 - CKA per LM layer, at each method's last step."""
    import matplotlib.pyplot as plt

    path = Path(path)
    title = f"Per-module weight drift and per-layer feature drift{suffix}"
    if drift_df.empty:
        write_missing_figure(path, title, "missing: no drift results")
        return
    last = drift_df[drift_df["step"] == drift_df.groupby("run_id")["step"].transform("max")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    methods = [m for m in METHOD_ORDER if m in set(last["method"])]
    weights = last[last["metric"] == "relative_l2"]
    groups = [g for g in GROUPS if g != "vision"]
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        means = weights[weights["method"] == method].groupby("module")["value"].mean().reindex(groups)
        _, color, _ = METHOD_STYLE[method]
        offset = (i - (len(methods) - 1) / 2) * width
        # A small floor keeps frozen groups visible on the log axis; they show as the shortest bars.
        left.bar(np.arange(len(groups)) + offset, means.fillna(0).clip(lower=1e-6), width=width * 0.9, color=color,
                 label=method_name(method), zorder=3)  # fmt: skip
    left.set_yscale("log")
    left.set_xticks(np.arange(len(groups)), groups, fontsize=8)
    left.set_ylabel("Relative weight L2 to S1", fontsize=9, color="#52514e")
    left.set_title("Weight drift by module group", fontsize=9, loc="left", color="#52514e")
    cka = last[(last["metric"] == "cka_distance") & (last["probe_split"] == probe_split)]
    for method in methods:
        means = cka[cka["method"] == method].groupby("layer")["value"].mean()
        if means.empty:
            continue
        _, color, marker = METHOD_STYLE[method]
        right.plot(means.index, means.values, color=color, marker=marker, markersize=5, linewidth=1.5, zorder=3)
    right.set_xlabel("LM layer", fontsize=9, color="#52514e")
    right.set_ylabel("1 - linear CKA to S1", fontsize=9, color="#52514e")
    right.set_title(f"Feature drift by layer, {probe_split}-task probe states", fontsize=9, loc="left", color="#52514e")
    for ax in (left, right):
        style_axis(ax)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def _fmt(values: pd.Series, scale: float = 100.0, digits: int = 1) -> str:
    values = values.dropna()
    if values.empty:
        return MISSING
    if len(values) == 1:
        return f"{values.iloc[0] * scale:.{digits}f}"
    return f"{values.mean() * scale:.{digits}f} ± {values.std() * scale:.{digits}f}"


TABLE_COLUMNS = [
    "Method", "Seeds", "Target SR", "Retention SR", "Forgetting", "Robust camera", "Robust lighting",
    "Robust background", "Robust noise", "libero_goal SR", "Swap followed", "GPU-h train", "GPU-h eval",
]  # fmt: skip


def table_one(eval_df: pd.DataFrame, checkpoints: pd.DataFrame, final_step: int = 20_000) -> pd.DataFrame:
    """Table 1, at the final step. Rates are percentages, mean ± SD over seeds. Absent cells read `missing`."""
    if eval_df.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    rows = []
    finals = eval_df[eval_df["step"] == final_step]
    for method in METHOD_ORDER:
        part = finals[finals["method"] == method]
        if part.empty:
            continue

        def per_run(mask, part=part) -> pd.Series:
            selected = part[mask]
            if selected.empty:
                return pd.Series(dtype=float)
            grouped = selected.groupby(["run_id", "alpha_spec"])
            return grouped["n_success"].sum() / grouped["n_episodes"].sum()

        chosen = checkpoints[(checkpoints["method"] == method) & (checkpoints["step"] == final_step)]
        row = {
            "Method": method_name(method),
            "Seeds": part["seed"].nunique(),
            "Target SR": _fmt(chosen["target_sr"]) if len(chosen) else MISSING,
            "Retention SR": _fmt(chosen["retention_sr"]) if len(chosen) else MISSING,
            "Forgetting": _fmt(chosen["forgetting"]) if len(chosen) else MISSING,
            "libero_goal SR": _fmt(per_run((part["eval_set"] == "retention") & part["task_id"].str.startswith("libero_goal/"))),
            "Swap followed": _fmt(per_run(part["eval_set"] == "language_swap")),
        }
        for axis in ("camera", "lighting", "background", "noise"):
            row[f"Robust {axis}"] = _fmt(per_run(part["eval_set"] == f"robustness_{axis}"))
        train = part.groupby("run_id")["gpu_hours_train"].first()
        full_sets = ~part["eval_set"].str.startswith("validation_")
        evals = part[full_sets].groupby("run_id")["gpu_hours_eval"].sum()
        row["GPU-h train"] = _fmt(train, scale=1.0)
        row["GPU-h eval"] = _fmt(evals, scale=1.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def to_latex(table: pd.DataFrame) -> str:
    def escape(value) -> str:
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")

    lines = [r"\begin{tabular}{l" + "r" * (len(table.columns) - 1) + "}", r"\toprule"]
    lines.append(" & ".join(escape(c) for c in table.columns) + r" \\")
    lines.append(r"\midrule")
    for row in table.itertuples(index=False):
        lines.append(" & ".join(escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def completeness_note(eval_df: pd.DataFrame) -> str:
    if eval_df.empty:
        return "No evaluation rows found."
    stage1 = eval_df[(eval_df["method"] == "stage1") & (eval_df["eval_set"] == "retention")]
    return f"S1 retention tasks present: {stage1['task_id'].nunique()} of {N_RETENTION_TASKS}."
