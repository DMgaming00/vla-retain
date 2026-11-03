#!/usr/bin/env python
"""Sweep merge weights on validation, pick the best, then run the full eval at that weight.

Array tasks call this with --alpha-index. A final call with --select reads the results and evaluates the chosen alpha.
"""

from __future__ import annotations

import argparse
import sys

from vlaretain.eval.suites import expand_eval_set
from vlaretain.methods.merge import global_alpha_grid, layer_alpha_grid, module_alpha_grid, select_alpha
from vlaretain.utils.io import append_jsonl, read_results

GRIDS = {"global": global_alpha_grid, "module": module_alpha_grid, "layer": layer_alpha_grid}
LABELS = {"global": "merge_global", "module": "merge_module", "layer": "layer_merge"}
SELECTION_FILE = "merge_selection.jsonl"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="the fine-tuned run to merge with S1")
    parser.add_argument("--step", type=int, default=20_000)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="global")
    parser.add_argument("--alpha-index", type=int, default=None, help="evaluate one grid entry on validation")
    parser.add_argument("--select", action="store_true", help="choose the best alpha from validation results")
    parser.add_argument("--full-eval", action="store_true", help="with --select: also run the full eval at that alpha")
    parser.add_argument("--with-drift", action="store_true", help="with --select or --alpha: also measure drift")
    parser.add_argument("--alpha", default=None, help="skip selection and run the full eval at this alpha")
    parser.add_argument("--list", action="store_true", help="print the grid size and exit")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser.parse_args(argv)


def _eval(args, step: int, alpha: str, eval_sets: list[str]) -> int:
    # Sibling script; scripts/ is on sys.path when this file runs.
    from eval import main as eval_main

    argv = [
        "--run-id", args.run_id, "--steps", str(step), "--alpha", alpha, "--method-label", LABELS[args.grid],
        "--run-root", args.run_root, "--results", args.results, "--n-envs", str(args.n_envs),
        "--n-action-steps", str(args.n_action_steps), "--device", args.device,
    ]  # fmt: skip
    for name in eval_sets:
        argv += ["--eval-set", name]
    return eval_main(argv)


def _drift(args, alpha: str) -> int:
    from drift import main as drift_main

    return drift_main([
        "--run-id", args.run_id, "--steps", str(args.step), "--alpha", alpha, "--method-label", LABELS[args.grid],
        "--run-root", args.run_root, "--results", args.results, "--device", args.device,
    ])  # fmt: skip


FULL_SETS = ["retention", "target", "robustness", "language"]


def main(argv=None) -> int:
    args = parse_args(argv)
    grid = GRIDS[args.grid]()
    if args.list:
        print(len(grid))
        return 0
    if args.dry_run:
        per_alpha = sum(j.n_episodes for j in expand_eval_set("validation"))
        print(f"run={args.run_id} step={args.step} grid={args.grid}: {len(grid)} alphas, {per_alpha} rollouts each")
        print(f"validation sweep total: {len(grid) * per_alpha} rollouts")
        for spec in grid:
            print(f"  {spec}")
        return 0
    if args.alpha is not None:
        code = _eval(args, args.step, args.alpha, FULL_SETS)
        return _drift(args, args.alpha) if args.with_drift and code == 0 else code
    if args.alpha_index is not None:
        return _eval(args, args.step, grid[args.alpha_index], ["validation"])
    if args.select:
        choice = select_alpha(read_results(args.results), args.run_id, args.step, grid)
        if choice["missing"]:
            print(f"Warning: {len(choice['missing'])} of {len(grid)} grid entries have no validation results")
        if choice["best"] is None:
            print("No validation results yet; nothing selected.")
            return 1
        best = choice["best"]
        print(f"selected {best['alpha_spec']} score={best['score']:.4f}")
        append_jsonl(
            f"{args.results}/{SELECTION_FILE}",
            [{"selection": True, "run_id": args.run_id, "step": args.step, "grid": args.grid, **best}],
        )
        code = _eval(args, args.step, best["alpha_spec"], FULL_SETS) if args.full_eval else 0
        return _drift(args, best["alpha_spec"]) if args.with_drift and code == 0 else code
    raise SystemExit("Pass one of --list, --dry-run, --alpha-index, --alpha, or --select")


if __name__ == "__main__":
    sys.exit(main())
