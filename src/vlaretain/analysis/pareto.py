"""Fig 1: target SR against retention SR, at matched steps and at matched GPU-hours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlaretain.analysis.common import METHOD_ORDER, METHOD_STYLE, MISSING, method_name


def pareto_front(points: pd.DataFrame, x: str = "target_sr", y: str = "retention_sr") -> pd.DataFrame:
    """The points no other point beats on both axes, with x ascending."""
    ordered = points.sort_values([x, y], ascending=[False, False])
    keep, best_y = [], -np.inf
    for index, row in ordered.iterrows():
        if row[y] > best_y:
            keep.append(index)
            best_y = row[y]
    return points.loc[keep].sort_values(x)


def method_means(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, per method and step."""
    if table.empty:
        return pd.DataFrame()
    grouped = table.groupby(["method", "step"])
    out = grouped[["target_sr", "retention_sr", "compute_hours"]].mean()
    out["target_sd"] = grouped["target_sr"].std().fillna(0.0)
    out["retention_sd"] = grouped["retention_sr"].std().fillna(0.0)
    out["n_seeds"] = grouped["seed"].nunique()
    return out.reset_index()


def matched_budget(table: pd.DataFrame) -> float | None:
    """The largest GPU-hour budget that every method reaches."""
    if table.empty or table["compute_hours"].isna().all():
        return None
    return float(table.groupby("method")["compute_hours"].max().min())


def at_budget(table: pd.DataFrame, budget: float) -> pd.DataFrame:
    """For each run, its latest checkpoint within the budget."""
    within = table[table["compute_hours"] <= budget * (1 + 1e-9)]
    if within.empty:
        return within
    return within.sort_values("compute_hours").groupby(["method", "run_id"], as_index=False).last()


def _plot_methods(ax, rows: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        part = rows[rows["method"] == method]
        if part.empty:
            continue
        _, color, marker = METHOD_STYLE[method]
        part = part.sort_values("step" if "step" in part else "target_sr")  # a curve follows training time
        ax.plot(part["target_sr"], part["retention_sr"], color=color, linewidth=1.5, alpha=0.55, zorder=2)
        ax.errorbar(
            part["target_sr"], part["retention_sr"], xerr=part.get("target_sd"), yerr=part.get("retention_sd"),
            fmt="none", ecolor=color, elinewidth=0.8, alpha=0.6, zorder=2,
        )  # fmt: skip
        ax.scatter(
            part["target_sr"], part["retention_sr"], color=color, marker=marker, s=48, linewidths=1.0,
            edgecolors="#fcfcfb", label=method_name(method), zorder=3,
        )  # fmt: skip


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(which="both", colors="#52514e", labelsize=8)
    ax.grid(color="#e8e7e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def write_missing_figure(path: Path, title: str, note: str = MISSING) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", fontsize=10, color="#0b0b0b")
    ax.text(0.5, 0.3, note, ha="center", fontsize=10, color="#52514e")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(table: pd.DataFrame, path: str | Path, suffix: str = "") -> dict:
    """Write Fig 1 and return what it shows."""
    import matplotlib.pyplot as plt

    path = Path(path)
    title = f"Target SR vs retention SR{suffix}"
    if table.empty:
        write_missing_figure(path, title, "missing: no checkpoint has complete retention and target results")
        return {"budget": None, "methods": []}
    means = method_means(table)
    budget = matched_budget(table)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
    _plot_methods(axes[0], means)
    axes[0].set_title("Matched steps: each mark is one checkpoint step", fontsize=9, color="#52514e", loc="left")
    if budget is not None:
        matched = at_budget(table, budget)
        matched_means = (
            matched.groupby("method")
            .agg(target_sr=("target_sr", "mean"), retention_sr=("retention_sr", "mean"),
                 target_sd=("target_sr", "std"), retention_sd=("retention_sr", "std"))
            .fillna(0.0).reset_index()
        )  # fmt: skip
        _plot_methods(axes[1], matched_means)
        axes[1].set_title(f"Matched compute: within {budget:.1f} GPU-hours", fontsize=9, color="#52514e", loc="left")
    else:
        axes[1].text(0.5, 0.5, "missing: no GPU-hour data", ha="center", transform=axes[1].transAxes, color="#52514e")
    for ax in axes:
        style_axis(ax)
        ax.set_xlabel("Target SR (libero_10)", fontsize=9, color="#52514e")
    axes[0].set_ylabel("Retention SR (Stage-1 tasks)", fontsize=9, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return {"budget": budget, "methods": sorted(means["method"].unique())}
