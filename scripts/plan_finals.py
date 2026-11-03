#!/usr/bin/env python
"""Pick each method's best tuning trial on validation, and write the finals plan: 3 fresh seeds per method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vlaretain.config import load_config, run_id_of, trial_config
from vlaretain.utils.io import read_jsonl, read_results, write_json

TRAINED_METHODS = ("sft_expert", "sft_full", "lora", "replay", "l2sp")
FINAL_SEEDS = (1, 2, 3)
TUNING_SEED = 0
GRID_LABELS = {"global": "merge_global", "module": "merge_module"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="plans/finals.json")
    parser.add_argument("--step", type=int, default=20_000)
    parser.add_argument("--methods", default=",".join(TRAINED_METHODS))
    parser.add_argument("--best", default=None, help="print only this method's best tuning run id, then exit")
    parser.add_argument("--list", action="store_true", help="print the plan in the existing file as shell-friendly lines")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing it")
    return parser.parse_args(argv)


def validation_score(rows: list[dict], run_id: str, step: int) -> float | None:
    """mean(target SR, retention SR) on validation, or None if either part is absent."""
    parts = []
    for eval_set in ("validation_target", "validation_retention"):
        match = [r for r in rows if r.get("run_id") == run_id and r.get("step") == step
                 and r.get("eval_set") == eval_set and r.get("alpha_spec") == "none"]  # fmt: skip
        if not match:
            return None
        parts.append(sum(r["n_success"] for r in match) / sum(r["n_episodes"] for r in match))
    return sum(parts) / 2


def s1_run_id(root: Path) -> str:
    return run_id_of(trial_config([root / "base.yaml", root / "stage1.yaml"]))


def choose_trial(rows, root: Path, method: str, s1_run: str, step: int) -> dict:
    files = [root / "base.yaml", root / "methods" / f"{method}.yaml"]
    n_trials = len(load_config(files).tuning)
    scores = []
    for trial in range(n_trials):
        run_id = run_id_of(trial_config(files, trial, [f"s1_run={s1_run}", f"seed={TUNING_SEED}"]))
        scores.append({"trial": trial, "run_id": run_id, "score": validation_score(rows, run_id, step)})
    known = [s for s in scores if s["score"] is not None]
    if not known:
        return {"method": method, "best": None, "scores": scores}
    # Ties go to the earlier trial, which max() does for us.
    return {"method": method, "best": max(known, key=lambda s: s["score"]), "scores": scores}


def latest_selections(results_dir: str | Path) -> dict[tuple[str, str], dict]:
    """The last recorded alpha choice per (fine-tuned run, grid)."""
    path = Path(results_dir) / "merge_selection.jsonl"
    if not path.exists():
        return {}
    return {(r["run_id"], r["grid"]): r for r in read_jsonl(path)}


def build_plan(rows, root: Path, s1_run: str, methods: list[str], step: int, selections: dict) -> dict:
    plan: dict = {"s1_run": s1_run, "seeds": list(FINAL_SEEDS), "trained": [], "merges": [], "unresolved": []}
    for method in methods:
        choice = choose_trial(rows, root, method, s1_run, step)
        if choice["best"] is None:
            plan["unresolved"].append(method)
            continue
        plan["trained"].append({
            "method": method, "config": f"methods/{method}.yaml", "trial": choice["best"]["trial"],
            "tuning_run": choice["best"]["run_id"], "score": choice["best"]["score"],
        })  # fmt: skip
    m1 = next((t for t in plan["trained"] if t["method"] == "sft_full"), None)
    if m1 is not None:
        for grid in GRID_LABELS:
            selected = selections.get((m1["tuning_run"], grid))
            if selected is None:
                plan["unresolved"].append(f"merge_{grid}")
            else:
                plan["merges"].append({"grid": grid, "alpha_spec": selected["alpha_spec"], "score": selected["score"]})
    return plan


def plan_lines(plan: dict) -> list[str]:
    """`train <config> <trial> <seed>` and `merge <grid> <alpha> <seed>` lines, for the launch script."""
    lines = []
    for item in plan["trained"]:
        lines += [f"train {item['config']} {item['trial']} {seed}" for seed in plan["seeds"]]
    for item in plan["merges"]:
        lines += [f"merge {item['grid']} {item['alpha_spec']} {seed}" for seed in plan["seeds"]]
    return lines


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        for line in plan_lines(json.loads(Path(args.out).read_text())):
            print(line)
        return 0
    root = Path(args.configs)
    rows = read_results(args.results)
    s1_run = s1_run_id(root)
    if args.best:
        choice = choose_trial(rows, root, args.best, s1_run, args.step)
        if choice["best"] is None:
            print(f"No validation results for {args.best}", file=sys.stderr)
            return 2
        print(choice["best"]["run_id"])
        return 0
    methods = [m for m in args.methods.split(",") if m]
    plan = build_plan(rows, root, s1_run, methods, args.step, latest_selections(args.results))
    for item in plan["trained"]:
        print(f"{item['method']:12s} trial {item['trial']}  validation score {item['score']:.4f}")
    for item in plan["merges"]:
        print(f"merge {item['grid']:6s} {item['alpha_spec']}  validation score {item['score']:.4f}")
    if plan["unresolved"]:
        print(f"Not resolved, because results are missing: {plan['unresolved']}")
    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        write_json(args.out, plan)
        print(f"Wrote {args.out}")
    return 0 if not plan["unresolved"] else 2


if __name__ == "__main__":
    sys.exit(main())
