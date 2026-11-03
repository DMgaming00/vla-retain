#!/usr/bin/env python
"""Measure weight drift, feature drift, and policy divergence against S1, on the frozen probe set."""

from __future__ import annotations

import argparse
import sys
import time

from vlaretain.methods.merge import NO_MERGE, AlphaSpec
from vlaretain.models.modules import GROUPS
from vlaretain.utils.io import append_jsonl, git_sha, lerobot_sha, make_drift_row, read_results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", required=True, help="comma-separated checkpoint steps")
    parser.add_argument("--alpha", default=NO_MERGE)
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--probe", default="probes/probe_v1.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser.parse_args(argv)


def already_done(results_dir, run_id: str, step: int, alpha: str) -> bool:
    return any(
        r.get("eval_set") == "drift" and r.get("run_id") == run_id and r.get("step") == step
        and r.get("alpha_spec") == alpha
        for r in read_results(results_dir)
    )  # fmt: skip


def metric_rows(weight_drift: dict, probe_metrics: dict[str, dict]) -> list[dict]:
    """Flatten metrics into (probe_split, module, layer, metric, value) records."""
    rows = [
        {"probe_split": "all", "module": g, "layer": None, "metric": "relative_l2", "value": weight_drift[g]}
        for g in GROUPS
    ]
    for split, metrics in probe_metrics.items():
        for name in ("action_chunk_distance", "velocity_distance", "action_mmd"):
            rows.append({"probe_split": split, "module": "policy", "layer": None, "metric": name, "value": metrics[name]})
        for layer, value in metrics["cka_distance"].items():
            rows.append({"probe_split": split, "module": "lm", "layer": layer, "metric": "cka_distance", "value": value})
    return rows


# UNVERIFIED: needs GPU/LIBERO
def measure(args, step: int, alpha: str) -> None:
    from lerobot.policies import make_pre_post_processors

    from vlaretain.data.probes import load_probe, raw_probe_batches
    from vlaretain.drift.divergence import DivergenceConfig, compute_probe_metrics
    from vlaretain.drift.params import relative_l2_per_group
    from vlaretain.eval.runner import build_state, source_from_run
    from vlaretain.models.load import load_policy

    source = source_from_run(args.run_root, args.run_id, step, alpha, args.method_label)
    if source.s1_ckpt is None:
        raise SystemExit(f"{args.run_id} has no S1 reference; drift is measured for Stage-2 runs")
    start = time.monotonic()
    state = build_state(source)
    reference = load_policy(str(source.s1_ckpt), device=args.device).eval()
    policy = load_policy(str(source.s1_ckpt), device=args.device).eval()
    policy.load_state_dict(state, strict=False)
    weight_drift = relative_l2_per_group(state, {k: v.cpu() for k, v in reference.state_dict().items()})
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(source.s1_ckpt),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )  # fmt: skip
    probe = load_probe(args.probe)
    cfg = DivergenceConfig()
    probe_metrics = {}
    for split in ("new", "old"):
        batches = [preprocessor(b) for b in raw_probe_batches(probe[split], args.batch_size)]
        probe_metrics[split] = compute_probe_metrics(policy, reference, batches, cfg)
    hours = (time.monotonic() - start) / 3600.0
    code_sha, lib_sha = git_sha(), lerobot_sha()
    rows = [
        make_drift_row(
            run_id=source.run_id, method=source.method, seed=source.seed, step=step, alpha_spec=alpha,
            gpu_hours_train=source.gpu_hours_train, gpu_hours_eval=hours, git_sha=code_sha, lerobot_sha=lib_sha, **record,
        )  # fmt: skip
        for record in metric_rows(weight_drift, probe_metrics)
    ]
    # All rows go in one append, so a killed job leaves either the whole set or nothing.
    append_jsonl(f"{args.results}/drift_{args.run_id}.jsonl", rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    alpha = args.alpha if args.alpha == NO_MERGE else AlphaSpec.parse(args.alpha).canonical
    steps = [int(s) for s in args.steps.split(",")]
    if args.dry_run:
        print(f"run={args.run_id} steps={steps} alpha={alpha} probe={args.probe}")
        return 0
    for step in steps:
        if already_done(args.results, args.run_id, step, alpha):
            print(f"step {step}: nothing to do")
            continue
        measure(args, step, alpha)
        print(f"step {step}: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
