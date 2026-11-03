#!/usr/bin/env python
"""Gate 1, after the pilots: is there forgetting to study, and does replay reduce it?

Exit code: 0 on PASS, 1 on FAIL, 2 when results are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlaretain.config import run_id_of, trial_config
from vlaretain.utils.io import read_results

S1_THRESHOLD = 0.80
FORGET_THRESHOLD = 0.10
STAGE1_SUITES = ("libero_spatial", "libero_object", "libero_goal")
FIXES = {
    "s1": "S1 is too weak to forget from. Train Stage-1 longer, or unfreeze the LM in configs/stage1.yaml, "
          "and confirm --n-action-steps and the control mode match training.",
    "m1": "M1 barely forgets, so retention methods have little to show. Try a larger learning rate (trial 2, 1e-4), "
          "longer Stage-2, or check that the M1 trainable counts include the LM.",
    "m3": "Replay does not help over M1. Check the replay ratio in the train log, then try the 10% pool. "
          "If it persists, the forgetting may be in the expert: compare M0 and M1.",
}  # fmt: skip


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--s1-run", default=None)
    parser.add_argument("--m1-run", default=None)
    parser.add_argument("--m3-run", default=None)
    parser.add_argument("--step", type=int, default=20_000)
    return parser.parse_args(argv)


def default_runs(configs: str | Path) -> tuple[str, str, str]:
    root = Path(configs)
    s1 = run_id_of(trial_config([root / "base.yaml", root / "stage1.yaml"]))
    pin = [f"s1_run={s1}"]
    m1 = run_id_of(trial_config([root / "base.yaml", root / "methods/sft_full.yaml"], 1, pin))
    m3 = run_id_of(trial_config([root / "base.yaml", root / "methods/replay.yaml"], 0, pin))
    return s1, m1, m3


def task_rates(rows: list[dict], run_id: str, step: int | None = None) -> dict[str, float]:
    """Retention SR per task. Without a step, use the run's last evaluated step."""
    retained = [r for r in rows if r.get("run_id") == run_id and r.get("eval_set") == "retention"
                and r.get("alpha_spec") == "none"]  # fmt: skip
    if not retained:
        return {}
    chosen = step if step is not None else max(r["step"] for r in retained)
    return {r["task_id"]: r["sr"] for r in retained if r["step"] == chosen}


def suite_mean(rates: dict[str, float]) -> float | None:
    means = []
    for suite in STAGE1_SUITES:
        values = [v for k, v in rates.items() if k.startswith(f"{suite}/")]
        if not values:
            return None
        means.append(sum(values) / len(values))
    return sum(means) / len(means)


def forgetting(s1: dict[str, float], run: dict[str, float]) -> float | None:
    shared = set(s1) & set(run)
    return sum(s1[t] - run[t] for t in shared) / len(shared) if shared else None


def evaluate(rows: list[dict], s1_run: str, m1_run: str, m3_run: str, step: int) -> list[dict]:
    s1 = task_rates(rows, s1_run)
    m1, m3 = task_rates(rows, m1_run, step), task_rates(rows, m3_run, step)
    s1_mean = suite_mean(s1)
    forget_m1, forget_m3 = forgetting(s1, m1), forgetting(s1, m3)

    def check(name, key, value, passed, detail):
        status = "MISSING" if value is None else ("PASS" if passed else "FAIL")
        return {"name": name, "key": key, "status": status, "detail": detail if value is not None else "no results yet"}

    return [
        check("S1 averages >= 80% across Stage-1 suites", "s1", s1_mean, s1_mean is not None and s1_mean >= S1_THRESHOLD,
              f"S1 mean SR = {(s1_mean or 0) * 100:.1f}%"),
        check("M1 forgets >= 10 points", "m1", forget_m1, forget_m1 is not None and forget_m1 >= FORGET_THRESHOLD,
              f"M1 forgetting = {(forget_m1 or 0) * 100:.1f} points"),
        check("M3 forgets less than M1", "m3", forget_m3 if forget_m1 is not None else None,
              forget_m1 is not None and forget_m3 is not None and forget_m3 < forget_m1,
              f"M3 forgetting = {(forget_m3 or 0) * 100:.1f} points"),
    ]  # fmt: skip


def main(argv=None) -> int:
    args = parse_args(argv)
    s1_run, m1_run, m3_run = default_runs(args.configs)
    s1_run, m1_run, m3_run = args.s1_run or s1_run, args.m1_run or m1_run, args.m3_run or m3_run
    print(f"runs: S1={s1_run} M1={m1_run} M3={m3_run} step={args.step}")
    checks = evaluate(read_results(args.results), s1_run, m1_run, m3_run, args.step)
    for item in checks:
        print(f"{item['status']:8s}{item['name']}: {item['detail']}")
        if item["status"] == "FAIL":
            print(f"        Suggested fix: {FIXES[item['key']]}")
    statuses = {c["status"] for c in checks}
    verdict = "MISSING" if "MISSING" in statuses else ("FAIL" if "FAIL" in statuses else "PASS")
    print(f"GATE 1: {verdict}")
    return {"PASS": 0, "FAIL": 1, "MISSING": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
