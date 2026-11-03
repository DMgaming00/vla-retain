#!/usr/bin/env python
"""Evaluate checkpoints of a run, or an in-memory merge, on the eval sets. Safe to re-run: finished work is skipped."""

from __future__ import annotations

import argparse
import logging
import sys

from vlaretain.eval.runner import (
    expand_all,
    make_libero_task_runner,
    pending_jobs,
    run_jobs,
    select_jobs,
    source_from_run,
)
from vlaretain.eval.suites import FULL_EVAL
from vlaretain.methods.merge import NO_MERGE, AlphaSpec


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", required=True, help="comma-separated checkpoint steps")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--eval-set", action="append", default=None, help=f"repeatable; default: {FULL_EVAL}")
    parser.add_argument("--alpha", default=NO_MERGE, help="merge spec such as global=0.5; default: no merge")
    parser.add_argument("--method-label", default=None, help="the method name written to results")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--soft-reset", action="store_true", help="faster, but not bit-identical to hard resets")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print pending work and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    alpha = args.alpha if args.alpha == NO_MERGE else AlphaSpec.parse(args.alpha).canonical
    names = tuple(args.eval_set or FULL_EVAL)
    jobs = select_jobs(expand_all(names), args.shard, args.num_shards)
    steps = [int(s) for s in args.steps.split(",")]
    if args.dry_run:
        rollouts = sum(j.n_episodes for j in jobs)
        print(f"run={args.run_id} steps={steps} alpha={alpha} sets={names}")
        print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} task jobs, {rollouts} rollouts per checkpoint")
        return 0

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    for step in steps:
        source = source_from_run(args.run_root, args.run_id, step, alpha, args.method_label)
        if not pending_jobs(jobs, source, args.results):
            print(f"step {step}: nothing to do")
            continue

        def factory(source=source):
            return make_libero_task_runner(
                source, device=args.device, n_envs=args.n_envs, n_action_steps=args.n_action_steps,
                hard_reset=not args.soft_reset, use_amp=args.use_amp, video_dir=args.video_dir,
            )  # fmt: skip

        done = run_jobs(source, jobs, args.results, factory)
        print(f"step {step}: ran {done} task jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
