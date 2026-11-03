#!/usr/bin/env python
"""Gate 2, after tuning: rho of new-state divergence against forgetting, with its bootstrap CI.

Forgetting here uses validation episodes, so that tuning decisions never touch the test init states.
Exit code: 0 if the CI excludes zero with positive rho, 1 otherwise, 2 when results are missing.
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser.parse_args(argv)


def gate_report(eval_df: pd.DataFrame, drift_df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    from vlaretain.analysis.common import checkpoint_table, drift_table
    from vlaretain.analysis.correlate import correlation_report, join_tables

    checkpoints = checkpoint_table(eval_df, retention_set="validation_retention", target_set="validation_target")
    joined = join_tables(checkpoints, drift_table(drift_df))
    return correlation_report(joined, n_boot=n_boot) if not joined.empty else pd.DataFrame()


def main(argv=None) -> int:
    args = parse_args(argv)
    from vlaretain.analysis.common import load_tables
    from vlaretain.analysis.correlate import PRIMARY

    report = gate_report(*load_tables(args.results), n_boot=args.n_boot)
    if report.empty or PRIMARY not in set(report["predictor"]):
        print("GATE 2: MISSING (no checkpoint has both validation and drift results)")
        return 2
    for row in report.itertuples():
        print(f"{row.label:50s} rho={row.rho:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] "
              f"LOMO R2={row.lomo_r2:+.3f} n={row.n} runs={row.n_runs}")  # fmt: skip
    primary = report[report["predictor"] == PRIMARY].iloc[0]
    if math.isnan(primary["rho"]) or math.isnan(primary["ci_lo"]):
        print("GATE 2: MISSING (too few runs for an interval)")
        return 2
    passed = primary["rho"] > 0 and primary["ci_lo"] > 0
    print(f"New-state divergence vs forgetting: rho={primary['rho']:+.3f}, 95% CI "
          f"[{primary['ci_lo']:+.3f}, {primary['ci_hi']:+.3f}]")  # fmt: skip
    print(f"GATE 2: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Suggested check: a null here is a finding. Look at per-method scatter and old-state divergence before "
              "changing the plan; do not tune the metric toward the claim.")  # fmt: skip
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
